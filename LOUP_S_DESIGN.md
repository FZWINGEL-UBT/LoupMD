# Loup-S: a per-system student architecture built from the engine's own measurements

Stage 0 implementation status: engine sampling, two-phase stress training, gated export and the label report have run end to end on SrTiO3. Both FP64 seed-1 students passed the 3× engine gate; round-0 force errors on their own trajectories were 13.18% (`s64l0`) and 12.65% (`s64r8`) against the original labels, versus the 2% target. These are not closed-loop results. **§6.5 now reports the two measured RTX 4000 Ada baselines**, not 5060 Ti timings; generic/fused targets remain estimates. See STATUS §6.70 and `plans/loup-s-stage0.md`.
Label-integrity update: the stale cell in a captured neighbor rebuild was fixed (`048489b907`). The **versioned 300/30/60 SrTiO3 base splits** passed the full 3× teacher gate and direct fresh-vs-reused agreement on every frame. The original labels, students and scores remain historical; the Ada measurements describe those archived bundles, not an adoption. Fidelity thresholds are unchanged.
Training backend decision: Stage 0 stays on mace-torch. Corrected centered-FP32 `s64l0` ran all 150 epochs, passed its 3× export gate and the 300 K label bar on its own trajectories (1.579% force RMSE). L4 medians were 17.47 s/epoch in both phases, versus 48.18 s/epoch so far for the matched FP64 first phase. This is a precision comparison, not a custom-engine speedup or a Stage 0 exit. See STATUS §6.70. Section 10's training-engine plan starts only after Stage 0.
Stage 0 physics and timing update: centered FP32 passed relaxed lattice but missed 8/960 phonon frequencies at both displacement sizes; it is not qualified for adoption. The 320-atom grouping experiment in §6.5 retained a small-cell fusion target on Ada, not a new production default.
Oxide fixed-set update (§§4, 8): 60 nominal student-visited and 40 independent hot/strained frames are frozen and teacher-gated. Retain the 5× target tail factor. The 3% oxide frame-audit threshold is an offline proposal, not a new general operating default. No universal per-atom trigger qualified.
§6.5 provenance clarification: the 5% paired-order analysis criterion was in the SHA-pinned experiment script before the first timed output. No timing or adoption result changed.

Original design status (23 November 2026, before implementation): the owner approved the direction and the outline of sections 4 to 10 on
23 November 2026. At approval, nothing here had been built. Every original figure is either quoted from `OPTIMISATION_LEDGER.md` or
`STATUS_AND_PROFILE.md` (entry or section named), marked as an estimate or a target, or cited from a named paper
and marked as not checked. No measurement was made for the original design.
Sections 6.4 and 12 and the pair-term, first-hop, message-form and harmonics rows of section 5.2's
table were added in discussion after that approval, and a second review the same day corrected sections 5.3, 6.2
and 6.5. A third, external review the same day moved the stop rule's baseline to `s64r8`; split and tightened the
fidelity bar (per species, molecular centre-of-mass terms, a tail cap, fixed tolerances for deterministic
observables, NVE at a conserving step); corrected launch 2's staging budget and set the layer-0 launches first in
Stage 2; and added stress to training, the virial to the lean buffers, the `h_i` audit trigger, the pair term's
safeguards, a radial-input row, a four-species system and falsifiers 9 and 10. A fourth review the same day added
the curvature loss on force differences (section 7), the trigger candidates and how one is chosen and the local
harvest for large cells (section 8), the one-layer correlation-4 and dual-cutoff variants (section 5.2), the
evaluation sets on all three systems and the rattled mix (Stage 0), and the second screen on water and the
four-species system (Stage 1a). Its literature is cited, but no figure from it was checked. A fifth review the same
day led to the harvest radius of `2 L r_c` beyond the core, which corrects the fourth revision's 15 A (section 8);
the atom-local step before layer 0's backward, `U` out of launch 2, `m1` in registers and the occupancy table
(section 6.2); the dual cutoff's outer shell in the compaction pass and the one-layer model's single sweep (section
5.2); dense curvature probes (section 7); the energy-range baseline trigger (section 8); and Stage 1a's screening
rules. A sixth set of ideas added normalised trigger variants, molecular and tail loss terms, resuming from the
checkpoint before the second phase, and labelled replay with a weight-interpolation sweep for the harvest
(sections 7 and 8, falsifier 2); a check of `eps` against anharmonicity was judged unnecessary (section 7).

It replaces, for this project, the external design "StrataEQ-MD: A Hardware-Co-Designed Multi-Scale Equivariant
Interatomic Potential" that the owner supplied on 23 November 2026 (not in the tree). Section 2 reviews that design.

## 1. Summary

The goal is a potential that is much faster than MACE at MACE's accuracy. The first product is a model for one
system, trained on a MACE teacher's labels. A foundation model is undecided and not designed here.

- **StrataEQ-MD's speed case does not hold against the ledger.** It measures itself against MACE at 32 us per
  atom-step and takes its gain from INT8 tensor-core emulation, SO(2) edge-frame convolutions, half-edge tiles and a
  three-level r-RESPA. LoupMD runs MACE-OFF23 small at 0.57 us per atom-step on the RTX 5060 Ti, and each of those
  levers has been measured here as a loss, a tie or a few per cent (section 2).
- **The ledger already holds the route that works.** A small MACE distilled from MH-1 on one system (`s64l0`) sits
  9.7 meV/A from its teacher at 0.094 of the teacher's call. The kernel census says where the rest of the time goes:
  equivariant senders, wide channels on edges, the last layer's l >= 1 outputs, and dense maps between kernels.
- **Loup-S** keeps MACE's algebra and removes those four costs by construction: scalar messages narrowed to 16
  channels, an 8-function radial head, no two dense maps back to back, and widths small enough that a layer's weights
  fit shared memory, so a force call can be three or four fused launches.
- **Bars.** Force error within 2% of the RMS force against the teacher, overall and per species, on frames the
  student's own MD visits; at most 0.15 us per atom-step at 5,000 atoms on the 5060 Ti (a target); a stop rule against
  `s64r8`, the stock student that needs no new kernel, which falls back to lean MACE students.
- **First step.** Stage 0 (section 10): bring the distillation into the tree and take the existing students' kernel
  census, which replaces this document's estimates with measurements.
- **What may improve it.** Section 5.2's ablations include changes to the model itself (harmonics to l = 2,
  species in the radial input, factorised messages, a radial first hop, a learned pair term, one layer of
  correlation order 4, a short many-body cutoff with a longer pair term), and section 7 uses the teacher's nearly
  free labels to trade data for model size, with a curvature loss if phonons miss. Section 12 lists what lies
  beyond. None of it is tested.

## 2. Review of StrataEQ-MD against LoupMD's measurements

### 2.1 The baseline it compares with

StrataEQ-MD quotes MACE-OFF23 small at 395.3 ms per step on 12,255 atoms (32.25 us per atom, one H100) and four larger
MACE models out of memory on 80 GB, and projects itself at 1.39 us per atom single-step and 0.34 us under r-RESPA.

| Quantity | StrataEQ-MD's figure | LoupMD, RTX 5060 Ti 16 GB | Source |
| --- | --- | --- | --- |
| MACE-OFF23 small | 32.25 us per atom (H100) | 0.57 us (water, 5,184 atoms, 2.966 ms per step) | STATUS 5.3, S40 |
| MACE-OFF23 medium | not listed | 1.42 us (water, 5,184 atoms, 7.35 ms) | STATUS 5.3, S20 |
| OMAT-0 medium | not listed | 1.5-1.7 us (SrTiO3, 5,000 atoms; 300 K on the Ada 8.65 ms) | STATUS 5.3, `meas-rest-vs-300k` |
| MH-1 | out of memory at 12,255 atoms | 4.7 us (SrTiO3, 5,000 atoms, 23.35 ms per call) | STATUS 6.62 |
| Largest system | 5,300-12,255 atoms | OMAT-0 medium: 121,945 atoms whole, 10,985,000 in tiles | STATUS 6.35 |
| Its own projection | 1.39 us single-step (H100) | slower than OFF23 small above, on a smaller card | – |

Its "23-95x over MACE-OFF23 small" therefore becomes 0.4-1.7x against the engine in this repository, before the
difference between the two GPUs. In the same LAMMPS job Symmetrix runs OMAT-0 medium SrTiO3 at 13.5 ms per step where
`run_style loupmd` takes 7.85 (STATUS 5.3, S145), so the gap is not a property of LoupMD's benchmark alone.

### 2.2 Its speed levers

| StrataEQ-MD | Measured here | Entry | Verdict |
| --- | --- | --- | --- |
| Tensor cores sit idle; 15-56x headroom over FP32 | 5060 Ti dense matmul: FP32 17.1, FP16 49-50 TFLOP/s. At FP32-level accuracy (the two-half kernels, on by default) a call takes 0.77 of its time on MH-1 and OMOL-0 XL and 0.97 on OMAT-0 medium (`sm_89`). The convolutions, 61% of OMAT-0 medium's step, are bound by issue rate and table traffic, which tensor cores do not touch | `meas-radial-precision-screens`, `ideas-20261121-ws`, `meas-conv-ablation-bounds` | built already; no further lever |
| "Emu2": two 7-bit slices, three INT8 products, accurate to 5.3e-5 | 588x / 1,498x the float32 model's force error on MACE (rattled SrTiO3) | `meas-tc-overlap-ozaki-locality` | fails |
| "Emu3": six INT8 products as the fallback | Passes its screens in the `oz6_v2` form. Sustained on the Ada 21.8-25.3 TFLOP/s against the two-half kernels' 18.5-20.9; ceiling 1.10x (MH-1), 1.12x (OMOL-0 XL); loses with its own pack pass from 1,500 atoms | `meas-ada-power-sustained`, lists 32, 33 | closed |
| SO(2) edge-frame convolutions replace Clebsch-Gordan paths | A bond-frame backward is bounded at 452 us, 5.8% of OMAT-0 medium's kernel time; rotating each adjoint into the edge frame costs more than the contractions it saves | `meas-conv-bwd-algebra-bound` | not built |
| Two-chart pole-free atlas | Needed only by edge frames. The engine evaluates real harmonics of the Cartesian edge vector and their gradient in closed form | STATUS 5.1 | not needed |
| Undirected half-edge list | 6.4 KB more reads against 10.2 KB saved per bond before cache reuse; adds reductions for the second endpoint | list 17 item 6 | not built |
| Edge-parallel kernels lose over 30% to `atomicAdd` | Sender-gradient atomics 5% of the backward conv (159 us per step); the six FP64 force atomics per edge 29.5 us per call | `meas-conv-ablation-bounds`, `meas-s142-bounds` | not a bottleneck |
| Hilbert order, L2-resident tiles of 4,096 atoms | Perfect sender locality is worth under 1% of the step; L2-sized receiver chunks lose 0.2-2.8%; the tiled call costs 1.24-1.5x the whole call | `meas-tc-overlap-ozaki-locality`, `int1-l2-chunks`, STATUS 6.35, list 17 item 7 | closed |
| A 3.5 A stratum holds the stiff motion; the rest moves slowly | The same model split at 3.5 A: outer 4 fs drifts 0.24 meV/atom/ps at RMS 0.063, against plain Verlet 4 fs at -0.011 and 0.037 | list 17 item 4 | rejected |
| Layers separate by frequency | OMAT-0 medium's layer 1 holds 93% of the stiffest modes' curvature; a layer split gives 0.95-1.25x (estimate) | list 12 item 1 | not built |
| r-RESPA 1:4:8, about 4x | Only the dual-model exact form gains: 1.5-1.8x on SrTiO3 and 1.3-1.7x on water with a foundation model as the fast force, 2.3-2.9x on SrTiO3 with a distilled student. Water's outer step caps near 2 fs | lists 13, 15 | kept in that form |
| Hydrogen schedules with mass repartitioning | Repartitioned water holds 2 fs, not 3. Rigid water holds 3 fs but is a different model from the one MACE-OFF learned | list 17 item 5 | the owner's call |
| Mesh long range at 0.40 us per atom | The D3 mesh (order 8, 0.8 A) costs about 5 ms at 5,000 atoms. POLAR-1's k-space kernels are 5.8% (M) and 9.3% (S) of a call | STATUS 6.31, list 17 item 9 | optional stage |
| Low-rank bottlenecks; lower degree at long range | Post-hoc truncation fails (rank 112 of 128: 443-1,128x). The high-degree paths do not decay first. A rank *trained* in works: an 8-function radial head costs nothing | `meas-s142-bounds`, `meas-degree-channel-shell-screens`, list 16 item 5 | train it in |
| NVE gate of 0.1 meV/atom/ns | Float32 MACE with plain Verlet at 1 fs: -0.11 meV/atom/ns on SrTiO3 (std 0.36 over 1 ps), -50 on OFF23 small | STATUS 1, 3.2 | gate against the teacher instead |

One number sums up the mismatch. StrataEQ-MD counts 114.5 MFLOP per atom for its medium-range stratum. At the rate
the Ada sustains for FP32-accurate tensor-core GEMMs (18-25 TFLOP/s) that is 4.6-6.4 us per atom, about three times
OMAT-0 medium's whole step on that card. The design spends more arithmetic per edge than MACE and counts on a
tensor-core rate these cards do not have.

That argument holds for the cards the ledger covers and not beyond them. On GeForce cards FP16 products with FP32
accumulation run at half rate, so the 5060 Ti's tensor cores give about 3x its FP32 rate; on an H100 the ratio is
about 15x (vendor figures, not measured here), and FP32-accurate emulation of some 150 TFLOP/s or more would put the
same 114.5 MFLOP at about 0.7 us per atom. The rows above that rest on issue rate, table traffic, force error and
drift (Emu2, the SO(2) backward, the shell split) do not depend on the card; the arithmetic count alone does.

### 2.3 What is kept

- Exact equivariance, forces as the exact gradient, smooth cutoffs, a ZBL core, FP64 state and accumulators. The
  engine has all of them.
- A fused per-atom kernel that runs a layer's forward and backward in one pass (its K2 / K3), used where a layer's
  weights fit shared memory (section 6.2).
- Distilling a small model from a large teacher (its Phase 3.3), which here becomes the main line.
- Judging by MD observables, phonons and NVE behaviour, with falsifiers before any training at scale.
- Curvature supervision, as a fallback (section 7), and a mesh long-range term, as an optional stage (section 10).

### 2.4 Not checked

The 2026 preprints and benchmarks StrataEQ-MD cites (DPA4 / EMFA-SO(2), FlashSO(2), e3j, the 12,255-atom H100 table)
were not verified for this review, and LoupMD has no H100 or B200 measurement. Its hardware projections for those
cards are neither confirmed nor refuted here; the review covers the cards the ledger covers.

## 3. What the measurements say a fast model looks like

**Where a MACE step goes** (OMAT-0 medium, 128x0e + 128x1o, 6 A, SrTiO3 5,000 atoms at 300 K, RTX 4000 Ada,
`meas-rest-vs-300k`): the layer-1 convolution pair 5.27 of 8.65 ms (61%), the projections on the two-half kernel 2.21,
the products 0.65, the two layer-0 edge kernels 0.70.

**What an edge costs, by kernel class** (per active edge; SrTiO3 at 6 A has about 370,000):

| Kernel class | Time | Per edge | Source |
| --- | --- | --- | --- |
| Layer 0 forward (species-only input, 66 radial columns) | 218-237 us | 0.6 ns | `meas-rest-vs-300k` |
| Layer 0 backward with ZBL and force assembly | 375-458 us | 1.0-1.2 ns | STATUS S142, `meas-rest-vs-300k` |
| Layer 1, scalar senders, 128 channels: forward / backward | 591 / 1,160 us | 1.6 / 3.1 ns | `s24-l0-conv` |
| Layer 1, scalars and vectors, 128 channels: forward / backward | 2,050 / 3,221 us | 5.5 / 8.7 ns | `meas-rest-vs-300k` |

**The levers, by measured size:**

1. The last layer's l >= 1 outputs are 58-60% of the step in the five 128-channel two-layer models timed
   (`meas-tp-path-pruning`). Masked out of a trained oxide model they cost 94-204 meV/A, so a model has to be trained
   without them.
2. Vector senders cost about three times scalar ones at the same width (table above; two different builds).
3. Edge work grows with the channels on the edge, and the GEMMs with the square of the node width.
4. Edges grow as the cube of the cutoff: 5 A has 0.58 of 6 A's.
5. The radial head's width sets layer 0's columns: 66 with a 64-wide head, 10 with an 8-wide one (`generic.py`,
   `J = H + 2`).
6. Two dense maps with nothing between them are one map. Folding them at load gained 1.5-7.5% per family and 18% on
   OMOL-0 (`gen-chain-fold`, `gen-chain-fold-ext`), and 8-11% in POLAR-1's SCF (`pol-dot-fold`).
7. The number format is the smallest lever: 1.03-1.3x of a call (row 1 of section 2.2), and only where GEMMs dominate.

**What a student needs to hold its teacher** (SrTiO3 4x4x4, MH-1 `omat_pbe`, 300 K test frames; lists 15, 16):

| Model | Force RMSE against MH-1 | Call, 320 atoms |
| --- | --- | --- |
| `s64l1`: 64x0e + 64x1o, two layers, 5 A | 8.0 meV/A (1.41%) | 249.6 us (0.141) |
| `s64l0`: 64x0e, two layers, 5 A | 9.7 meV/A (1.72%) | 165.3 us (0.094) |
| `s64r8`: `s64l0` with an 8-wide radial head | 9.2 meV/A (1.63%) | not timed |
| `s64r4`: 4-wide radial head | 10.9 meV/A (1.93%) | not timed |
| Linear ACE, two- and three-body, 2,106 parameters | 48.7 meV/A | not built |
| OMAT-0 medium (a sibling, not a student) | 25.2 meV/A | 0.32 of MH-1's call |

Sibling foundation models differ by 32.6-53.2 meV/A on the gate's oxide frames (`meas-tp-path-pruning`). A student
within 10 meV/A is therefore three to five times closer to its teacher than the teacher is to its peers. Each row is
one seed on one system.

## 4. Contract

**Product.** A student is trained for one system from any MACE teacher LoupMD runs and exported as an ordinary bundle.
The engine's 3x float32-against-float64 gate applies to the student's own float64 reference, unchanged; LAMMPS styles,
integrators and tiled mode come with the bundle. A student is a different potential from its teacher, so it carries its
own accuracy statement beside the 3x rule, as the ledger's reduced-model row requires.

**Fidelity bar.** Measured on frames taken from the student's own MD at the target conditions and relabelled by the
teacher, not on frames from the teacher's trajectory:

- force RMSE at most 2% of the frames' RMS force (about 10 meV/A on SrTiO3 at 300 K), overall and for each species;
- in molecular systems also the molecules' centre-of-mass forces and torques, each at 2% of its own RMS: in water the
  bonded O-H forces dominate the total, and a bar on the total alone would let the intermolecular forces carry an
  error of tens of per cent;
- the 99.9th percentile of the per-atom force error at most 5x the force bar, and the largest single error reported
  (the factor is a default to set in Stage 0);
- energy RMSE at most 1 meV per atom; the stress error reported, bounded in effect by the lattice-parameter tolerance;
- statistical observables within the teacher's own spread over two MD seeds, with error bars: radial distribution
  functions, each species' RMS displacement, lattice parameter or density under NPT at temperature, and diffusion
  where the system has any;
- deterministic observables within fixed tolerances, since two MD seeds give them no spread: the relaxed lattice
  parameter within 0.2%, finite-displacement phonon frequencies within 3% or 0.1 THz, whichever is larger (starting
  values, to be set in Stage 0);
- NVE energy RMS and drift no worse than the teacher's at the production step, and also at a step where the teacher
  itself conserves energy well (0.5 fs for flexible water), where the integrator's error does not hide the student's.

**Speed bar.** A block or kernel is adopted on the ledger's rule: it wins `A B B A` in both orders, on the GPUs the
owner names. The targets, both estimates until Stage 0:

- at most 0.15 us per atom-step from 5,000 atoms on the 5060 Ti, about 10x OMAT-0 medium and 30x MH-1;
- at least 2x `s64r8` on cells up to 500 atoms.

**Stop rule.** If the trained variants of Stage 1 predict less than 1.5x over `s64r8` at equal fidelity, the project
ships lean MACE students on today's kernels and builds no new ones. The baseline is `s64r8`, not `s64l0`: its 8-wide
radial head cuts `C0` from K 195 to K 27 on today's kernels (section 6.5), about 1.2x over `s64l0` (estimate) that
needs no new kernel, so it must not count toward the gain that justifies one.

## 5. Architecture

### 5.1 Blocks

Two layers in MACE's algebra. `N(i)` is the neighbours of atom `i` within the cutoff, `Y_lm` the real spherical
harmonics of the edge direction (l <= 3), `P3` MACE's order-3 symmetric contraction per channel with per-species
weights.

```
Layer 0
  S0_i[s, n, lm] = sum over j in N(i) with z_j = s of  phi_n(r_ij; z_i, s) Y_lm          n = 1..R
  m0_i[c, lm]    = sum over (s, n) of  C0_l[z_i][(s, n), c] S0_i[s, n, lm]               c = 1..C
  h_i[c]         = P3(m0_i[c, .]; z_i)                                                   scalars
  E0_i           = w0 . h_i

Layer 1
  x_j[k]         = sum over c of  U[c, k] h_j[c]                                         k = 1..C_msg
  A_i[k, lm]     = sum over j in N(i) of  w_kl(r_ij; z_i, z_j) Y_lm x_j[k]               w_kl = W[(k, l), .] . phi(r)
  m1_i[c, lm]    = sum over k of  V_l[k, c] A_i[k, lm]
  y_i[c]         = P3(m1_i[c, .]; z_i) + sum over c' of  K[z_i][c', c] h_i[c']
  E1_i           = MLP(y_i)                                                              C -> 16 -> 1

E = sum over i of (e0[z_i] + E0_i + E1_i) + ZBL pair term;  F = -dE/dR and the stress by the exact reverse pass
```

`phi` is the radial MLP's last hidden layer of width `R` on a Bessel basis with the polynomial cutoff. Layer 0 is the
form the engine already runs for every model's first layer. Layer 1 is MACE's interaction with scalar senders and
with `linear_up` narrowing to `C_msg`.

So Loup-S is a member of the MACE family, not a new equivariance mechanism. With `C_msg` = `C` it is a stock MACE
configuration (`64x0e`, `radial_MLP [64, 64, 8]`, 5 A: the student `s64r8`), and `U` is what the engine's fold already
makes of MACE's product map and `linear_up`. The narrow `linear_up` is the one change to the model itself. The rest
of the gain is the choice of widths and the runtime of section 6.

**Species in the radial functions.** The arguments `z_i, s` and `z_i, z_j` above are the engine's general form, not a
choice of Loup-S. A stock MACE radial MLP reads the Bessel basis only. The foundation models' blocks read the species
through a distance scaled by covalent radii, a symmetric term, and the gated (`nonlinear`) kind through separate
sender and receiver embeddings. The exporter writes one table per symmetric species pair, or per ordered pair for the
gated kind, in every case (`RadialChain`, `pair_sym`), so the choice changes the model and not the number of tables,
which grows as `S (S + 1) / 2` with `S` species. Loup-S takes the form `s64l0` was trained with, recorded in Stage 0;
the other is an ablation (section 5.2). With species-independent radial functions the 16 message channels carry the
sender's species as well as its environment, which three species may allow and six may not: that is what the
four-species system of Stage 0 tests.

**Stage 0 verification.** Both list 15 `s64l0` and list 16 `s64r8` used the plain distance-only Bessel input, not a species-scaled distance: both training logs say `Distance transform for radial basis functions: None`, and both checkpoints contain `BesselBasis` and `PolynomialCutoff`. The new recipe makes that choice explicit; the MLP widths are `[64,64,64]` and `[64,64,8]` respectively. ZBL is a separate pair contribution in the new stress-trained recipe, not a species embedding in its radial MLP.

### 5.2 Defaults and their ablations

| Choice | Default | Reason (section 3) | Ablation |
| --- | --- | --- | --- |
| Radial functions `R` | 8 | lever 5; `s64r8` loses nothing | 4, 16 |
| Node channels `C` | 64 scalars | lever 3; `s64l0` holds 9.7 meV/A | 32, 96 |
| Message channels `C_msg` | 16 scalars | levers 2, 3 | 8, 32, 64; 16x0e + 16x1o |
| Layer 1 outputs | l <= 3 (4 paths), `P3` to scalars | the form `s64l0` uses | scalar only (1 path, lever 1); order 2 in either layer |
| Dense maps | none back to back: `U` is the product's output map and the next input map in one | lever 6 | no `V_l` (each node channel reads one message channel) |
| Cutoff | 5.0 A, ZBL core | lever 4 | 4.5, 6.0 A |
| Depth | 2 layers | linear one-layer ACE: 48.7 meV/A | 1 layer with `P3` and the MLP readout at 5 A (the cost floor); 1 layer with correlation order 4 (`P4`), a 6 A cutoff and a `C -> 64 -> 1` readout (`num_interactions=1, correlation=4` in mace-torch). One layer runs as two launches with no `x`, `A`, `V_l` or `K`, and as one sweep in tiled mode in place of three. Against it: 6 A has 1.7x the edges of 5 A, the product's cost grows fast with its order, and no model the engine runs has correlation 4, so the product emitter has not compiled one. Nonlinear one-layer ACE models are reported close to two-layer ones on bulk materials (GRACE-1L); not checked here |
| Dual cutoff | one cutoff | lever 4 | the many-body layers at 4.5 or 4.2 A (27% or 41% fewer edges in both layers than at 5 A) with the learned pair term out to 6 A. The skin list must then reach 6 A plus the skin, about 1.7x today's list entries at 5 A. The compaction pass, which walks the list anyway, evaluates the pair term's energy, forces and virial on the outer shell and stores none of its edges, so the active-edge arrays keep the short cutoff's size and neither layer sees the shell. The forces go without atomics: each pair visited from both ends with only the receiver's half kept, or summed per receiver with the sender's half added atomically; the bench decides. Unlike masking a trained model (section 12), this is trained in |
| Harmonics | l <= 3 | all the engine takes today | l <= 2: 9 components for 16 in every sum, map and product. A stock mace-torch option; the kernels would have to be generated for it. Also l <= 3 in layer 0 and l <= 2 in layer 1 only, which needs the student's own blocks, since mace-torch's `max_ell` is global |
| Radial input | as `s64l0` was trained | section 5.1 | species-dependent (covalent-radius scaling) against plain Bessel, on the four-species system |
| Messages' radial form | a table column per channel and degree, as MACE | the kernels exist | factorised: a channel is a pair (radial function, sender scalar), 4 x 4 or 8 x 4, with one radial weight for all degrees and layer 0's table (section 12) |
| First hop | many-body: layer 0 with `P3` | the form `s64l0` uses | radial only: sender scalars from pair sums formed in the compaction pass (section 12) |
| Pair term | ZBL only | stock MACE | a learned table per species pair, evaluated in the assembly pass at no extra edge visit. A pair term took 72-74% of the Hessian norm of OMAT-0 medium's layer 1 (list 12). It carries the polynomial cutoff envelope (smooth to second order, for phonons and NVE) and is held non-negative, or at most a fixed share of ZBL, below 1.5 A, so it cannot make an attractive well inside the core at high temperature |

Each ablation is one training run on the same frames and seeds. A default changes only if the variant holds the
fidelity bar at a lower measured cost. The last five rows and the second one-layer and dual-cutoff variants change
the model itself and came from the second to fourth reviews. All of them can be trained and their accuracy measured
in torch before any kernel is written; the factorised messages and the radial first hop also run on today's
kernels, as constrained weights.

### 5.3 Left out

SO(2) frames, attention, softened coordinates and a separate short-range stratum are out, each for a row of
section 2.2. Degree-dependent cutoffs and lower precision are not in the default but are candidates when trained in
(section 12). Other fast families (NEP, Allegro, DPA) are untested here; the one-layer ablation stands in for the
strictly local class, and anything else would need its own kernels before it could be compared in this engine.

## 6. Runtime in LoupMD

### 6.1 First on today's generic path

Trained variants are exported and timed before any kernel is written.

- The bundle already carries separate node, convolution and message widths per layer (`C_in`, `C_x`, `C_msg`,
  `C_out`); MH-1 runs 512-wide nodes with 128-wide convolutions.
- Layer 0's column count follows the radial head (`J = H + 2`), so a rank-8 student gets the narrow layer 0 at once.
- Convolution widths without a built-in instance are compiled at load (`gen-jit-conv`), and a product signature the
  library lacks is compiled from the bundle (`gen-jit-products`).

To confirm in Stage 1: the exporter maps interaction blocks by class name, so the student's blocks need names it
knows or a new kind; narrow `C_x` is exercised only by the gated (`nonlinear`) kind today; the pruned
scalar-only kernels exist for vector senders (`gconv_*_L1s`) and not yet for scalar ones; the engine takes harmonics
up to l = 3 only. The convolution kernels run one thread per channel, so a 16-channel layer will under-fill its
blocks on this path: the time measured there understates what a kernel built for the width would reach.

### 6.2 Then fused kernels, where they win

Fusing the projections into the convolution failed for OMAT-0 medium because each block would read 655 KB of weights
(STATUS 4.1, priority 1). At `C` 64 and `C_msg` 16 a layer's weights are tens of KB: `V_l` 16 KB, `U` 4 KB, and per
species a few KB of product weights and 16 KB for `K`. They fit shared memory or L1, and a force call can be three
launches:

1. layer 0 forward: edge sums, `C0`, product, `U`; writes `S0`, `h` and `x`;
2. layer 1 forward, readout and backward in one pass per receiver: `A` and its gradient stay in shared memory, `m1`
   and its gradient in registers; writes the energy, the skip and readout gradients `dh_skip` of its receiver, the
   sender gradients `dx` (atomics) and the per-edge force vectors. It needs `V_l` and `K` but not `U`: `x` comes
   from launch 1, and `U^T` is applied only once `dx` is complete;
3. layer 0 backward with ZBL and force assembly. Before its edge pass, each atom needs
   `dh = dh_skip + U^T dx`, the product's backward and `dS0 = dm0 C0^T`, all local to the atom, and `dx` is complete
   only when launch 2 has ended. Today's `first_bwd_edge` is edge-parallel and cannot take that chain, so either a
   small node kernel runs it first (four launches) or launch 3 is receiver-owned and runs it as a prologue for its
   own receiver, which needs only that atom's `dx` and `dh_skip`. The bench decides; the estimates of section 6.5 do
   not separate the two.

Notes on that form:

- The launches are the tiled mode's three sweeps (layer 0 forward; the last layer forward and backward together;
  layer 0 backward, with its atom-local prologue), so the same kernels serve tiles, and what crosses between them
  is what the tiles already hand on.
- Launch 2 walks a receiver's edges twice. The first walk stages in shared memory what the second needs, so the
  second makes no global loads. What to stage depends on the radial form. The default's radial weights are a linear
  map of the 8 functions `phi` (`w_kl = W[(k, l), .] . phi`, section 5.1), so each edge can stage its unit vector and
  length, `phi` and `dphi/dr`, and the sender's 16 scalars: 36 floats, about 9 KB for 64 edges. The second walk then
  recomputes `Y_lm` and its gradient (layer 0's angular arithmetic is 1% of its kernel) and forms `w_kl` and its
  derivative in registers, 32 multiply-adds per channel thread and edge. `W` is one matrix of 2 KB per layer
  whatever the species count: the species enter only through the tables of `phi` (the exporter's `RadialChain`
  keeps one last-layer weight per layer and a spline per species pair). Staging the 64 table columns that today's
  kernels read instead takes 256 B per edge, twice that with their derivatives: 24-40 KB for 64 edges. Which form
  is faster is open: free table reads are worth 26-40% of a forward conv (`meas-conv-ablation-bounds`), and forming
  `W . phi` costs arithmetic in their place. A third form stages only the unit vector, the length and the sender's
  scalars (20 floats, 5 KB for 64 edges) and evaluates `phi` and `dphi/dr` again on the second walk from their
  tables in L2: less shared memory for about 32 spline coefficients read per edge. A receiver with more than 64
  edges (water at 5 A, any 6 A cutoff) stages the first 64 and reads the rest from L2 on the second walk.
- The block shape sets the occupancy, and Stage 2's bench decides it first. `V_l` (16 KB) is shared by the receivers
  of a block; `K` at 16 KB per species stays in L1 unless the receivers are sorted by species. With a thread per
  node channel, `m1[c, .]` and its gradient (16 values each) stay in that thread's registers, so a receiver adds only
  its staging and `A` with its gradient (2 KB) to shared memory. Four receivers per block (estimates, of the 100 KB
  of shared memory per SM on these cards):

  | Staging | Shared memory per block | Blocks per SM |
  | --- | --- | --- |
  | `phi` and `dphi/dr` staged | 4 x 11 + 16 = 60 KB | 1 (8 warps) |
  | `phi` and `dphi/dr` read again from L2 | 4 x 7 + 16 = 44 KB | 2 (16 warps) |

  At 16 warps the registers bind as well: 65,536 for 512 threads is 128 each. The bench reports the occupancy that
  shared memory and registers allow together; occupancy, not arithmetic, may bound launch 2.
- The compaction pass already visits every list edge with a warp per receiver. Layer 0's forward sums could be formed
  there, one traversal for two; in-place compaction was sized but not built (scan 2.5 us and scatter 27 us per call,
  the ledger's open table). A kernel variant for Stage 2.

That removes the GEMM launches (about 35 of a small call's kernels, estimate) and the node arrays between them. It is
a hypothesis until measured: receiver-owned forms have lost to edge-parallel ones here before (`i0-bwd-receiver-owned`
+1.1-1.8%; an edge-split spline 1.4-1.6x slower at 5,000 atoms), and a persistent kernel was bounded at 4.8-7.5% on a
GEMM-bound small cell (`meas-small-cell-idle`). Each fused kernel is kept only if it wins its `A B B A` against the
generic path, with the 3x gate on every forced variant.

### 6.3 Precision

FP32 kernels with FP64 accumulators, as now. No tensor cores: at these widths the GEMMs are a few per cent of the
step.

### 6.4 Memory

The device memory that grows with the system is the node buffers of `GenericModel::alloc_system` plus the edge
buffers: 104 B per edge slot (`src/engine.cu`, `GenericModel::ensure_edges`), allocated with a quarter of headroom,
so about 130 B per list edge. Counted that way, the two whole-system footprints of STATUS 6.35 come out within 1%:
OMAT-0 medium 98.5 kB per atom measured (the slope from 5,000 to 98,415 atoms) against 85.3 + 12.9 + 0.3 counted, and
OFF23 medium 90.3 against 81.1 + 9.1 + 0.3. Those measurements are of 18 November 2026. The count is of the code at
`9848a31`, and the two-half work since then added weight-sized panels only, so the figure should still hold; it was
not measured again for this document.

The same count for the students, in kB per atom (SrTiO3, 5 A cutoff, 0.5 A skin: 58.5 list neighbours per atom).
The student rows are estimates:

| Model | Node buffers | Edge buffers | Total | Atoms whole on the 5060 Ti |
| --- | --- | --- | --- | --- |
| OMAT-0 medium (measured) | 85.3 | 12.9 | 98.5 | 121,945 |
| FastMACE, OMAT-0 medium (measured, STATUS 6.34) | – | – | 84-87 (75.7 reserved) | 163,840 |
| `s64l0`, generic path | 40.2 | 7.6 | 48 | about 260,000 |
| Loup-S, generic path | 25.5 | 7.6 | 33 | about 380,000 |
| Loup-S, fused | 2.7 | 7.6 | 10.6 | about 1.2 million |

- On the generic path the footprint is the l <= 3 message arrays at 64 channels, 4.1 kB each: `msgf` and `dmsgf` in
  layer 0, `lin`, `dlin` and `dmsgf` in layer 1. The rank-8 head shrinks `S0` from 13.3 to 2.0 kB.
- Fused, only what crosses between the three launches is kept: `S0` with its gradient in place, `h`, `x` and their
  gradients. The edge buffers are then three quarters of the footprint.
- The atom counts take the 11.8 GiB the plan may use at fraction 0.9 on this card (STATUS 6.35). Water has 70 list
  neighbours per atom at the same cutoff, about 1.5 kB more. Tiled mode stays available beyond these counts.
- The model itself is small: roughly 0.2 MB of weights and 1.2 MB of radial tables for three species (estimate),
  against 45 MB for OMAT-0 medium's layer-1 table alone (`meas-mh1-conv-l2`).

**Lower still.** The fused student, same cell; each row adds to the one above. All estimates:

| Step | What changes | kB per atom | Atoms whole | Cost |
| --- | --- | --- | --- | --- |
| As designed | today's edge buffers | 10.6 | 1.2 million | – |
| Lean edge buffers | headroom 1.05 for 1.25; the tiled mode's 8 B list entries (`tile-device-compact`); launch 2 adds its own forces and the virial (per-block sums of `r f`, which stress and NPT need), so no per-edge force arrays; no compaction scratch; `dh` in `h`'s place | 5.4 | 2.3 million | force atomics, at most 30 us per 370,000 edges and kernel (`meas-s142-bounds`); one-pass compaction was a tie in MD (`edge-compaction-1pass`) |
| Receiver-owned edge data | no receiver index; the spline interval located in each kernel | 4.4 | 2.9 million | 1-3% of the step (prelocation was adopted as a gain, `gen-gconv-cfix-preloc`) |
| `S0` recomputed | the backward launch repeats layer 0's edge sums | 2.4 | 5.2 million | about 18% of the step |
| No active-edge arrays | kernels walk the skin list and form each edge's geometry | 1.2 | 10 million | 10-20% more: a quarter of the list lies outside the cutoff |

The first step belongs in Stage 2: it halves the footprint for a few per cent of the step. The last two trade time
for memory and matter only for cells of several million atoms, where the tiled mode already holds about 1 kB per
atom on the device (STATUS 6.35).

**In tiled mode** none of these steps matters. There the footprint is the rows handed between sweeps plus the lists,
not the layer buffers. The rows are layer 1's `x`, the readout's hidden layer and their gradients: 1,056 floats per
atom for OMAT-0 medium (4.2 kB) and 64 for Loup-S (0.26 kB, estimate). The lists take 8 B per list edge in the
compact form, about 0.5 kB per atom here, and go to pinned host memory at scale. A tiled student should therefore be
limited by host memory for its lists, where OMAT-0 medium is limited by the device again in the wavefront
(`docs/validation/capacity.md`). What will weigh more for a fast student is the fixed cost per tile, at most about
1 ms in the screen of 16 November (STATUS 6.23): 1% of an OMAT-0 medium tile of 65,536 atoms, about 10% of a
student's (estimate).

The generic path has the same lead for every model. STATUS 6.34 names sharing the per-layer buffers by lifetime and
has not measured it. A rough count for OMAT-0 medium, with each product's gradient written in place of its input and
layer 0's gradient buffers placed in layer 1's dead ones, gives about 58 kB of node buffers for today's 85, or about
71 kB per atom in all. The data flow of the density kinds needs checking before that count is relied on.

### 6.5 Measured baselines and estimated targets

SrTiO3, 5,000 atoms, 5 Å; ms per MD step. **Measured, RTX 4000 Ada only:** unprofiled sustained steps for each seed-1 bundle; separate 16-step Nsight Systems census at 300 K. The census rows sum kernel durations and include simultaneous streams; they do **not** add to the wall time. The trace identifies dedicated kernels but not callers of the shared dense kernels. Pool the latter rather than assigning them to `C0`, lifts or readouts from launch shape alone.

| Measured piece (Ada only) | `s64l0` | `s64r8` |
| --- | ---: | ---: |
| Layer 0 edge kernels (`first_fwd` / `first_bwd_edge`) | 0.487 | 0.319 |
| Layer 1 convolution pair (`gconv`) | 0.777 | 0.776 |
| All dense GEMMs, including `C0` and its transpose | 1.146 | 0.473 |
| Dedicated product / fused readout kernels (`gprod`) | 0.241 | 0.243 |
| Remaining kernels (compaction, rebuild, integrator and support) | 0.095 | 0.094 |
| Kernel union (overlap counted once) | 2.196 | 1.668 |
| **Unprofiled step** | **2.298 (0.460 µs/atom)** | **1.770 (0.354 µs/atom)** |

**320 atoms, Ada only — §9 falsifier 6.** In an unprofiled `A B B A` comparison, grouping 20 existing layer-0 projection launches into two saved 6.20% and 6.53% of the `s64l0` step in the two orders; `s64r8` saved 4.44% and 4.61%. Separate Nsight traces verified 18 fewer launches and unchanged projection block counts. Keep the small-cell fusion target: only the wide model cleared the 5% criterion in both orders. The criterion is in `tools/dev/student_launch_pairs.py`, SHA256 `0ef195979407513e5a0b51425a482a7d19dbbd743163c5944f64a9d065cee6b6` in the pre-run `launch-discriminator-v1/experiment.sha256`. This does not establish that most of either step is launch overhead, and is not a cross-GPU adoption.

**Estimates only, 5060 Ti design model:** §3 rates from different builds, cards and temperatures; FP32 GEMMs at 12 TFLOP/s. The design comparison below uses its original **estimated** `s64r8` baseline of 1.5 ms, **not** the measured Ada time. No local timing or cross-card speedup is implied.

| Piece | Loup-S, generic path | Loup-S fused (target) |
| --- | ---: | ---: |
| Layer 0 edge kernels | 0.37 | 0.25–0.30 |
| `C0` and its transpose | 0.05 (K 27) | in kernel |
| Layer 1 convolution pair | 0.13 (16 channels) | 0.10 |
| Lift / message maps | 0.03 | in kernel |
| Products and readouts | 0.33 | 0.10–0.15 |
| Compaction, rebuild, integrator | 0.08 | 0.08 |
| **Estimated step** | **1.0** | **0.55–0.65** |

The fused column's in-kernel maps are counted in its kernels' rows. The generic path's 16-channel figure scales the
128-channel rate by the width, which the under-filled blocks of section 6.1 will not reach on today's kernels.

Against the stop rule's baseline: Loup-S on the generic path is about 1.5x `s64r8`, and fused 2.3-2.7x. More than
half of the fused column lies outside layer 1: layer 0's edge kernels and the compaction pass are 0.33-0.38 ms of
0.55-0.65. If layer 0 stays at
today's 0.37 ms, fusing the rest gives 0.65-0.70 ms, 1.4-1.5x over the generic path and about 2.1-2.2x over `s64r8`.
The layer-0 launches (1 and 3) are therefore benched first in Stage 2, at 10 columns, before launch 2.

## 7. Distillation

1. **Teacher frames.** The teacher runs in LoupMD: Langevin MD at the target temperature and at 1.5-2x that, plus
   strained and rattled cells. List 15 used 300 frames of 320 atoms, the last 100 at 600 K, and 60 test frames.
   Labels are nearly free: one such frame is 25 MH-1 calls of 1.76 ms. The frame count is therefore a training-time
   choice, and Stage 0 measures the learning curve from 300 to 10,000 frames. If the error keeps falling, a smaller
   student reaches the same bar, which is a speed lever that costs no kernel work. The larger sets add independent
   configurations, not more steps of the same run: more temperatures, isotropic and shear strains, and a stride
   between frames longer than the slowest decorrelation time that matters for the observables.
2. **Train.** Stage 0 uses `mace-torch==0.3.16` to train the student, initially in float64, on energies, forces and, for periodic systems, stress:
   the bar holds lattice parameters and densities under NPT, and a model fitted to forces alone can carry a pressure
   offset. The teacher's stress comes with its forces. About 2-3 h per student on an L4 (list 15: 1 h 46 min and
   3 h 22 min, without stress). Float32 is usually twice as fast by mace-torch's documentation and replaces float64
   if Stage 0's comparison holds through the whole schedule, including the second phase with its raised energy
   weight, where float32 runs are most likely to go wrong. An auxiliary loss on the teacher's per-atom energies, N
   labels a frame for one, is an ablation. Those energies are a partition, not an observable: the teacher's come
   from a wider receptive field than the student's 10 A (two layers of 5 A), and models with the same totals can
   divide them differently by species. The ablation therefore removes each species' mean offset first and keeps the
   weight small, and is kept only if it lowers the force error. The energies come from torch MACE, since the
   engine returns totals.

   The force loss is uniform MSE by default, with two additions where section 4's bar asks for them. In molecular
   systems, terms on each molecule's centre-of-mass force and torque, formed from the predicted atomic forces: in
   water the O-H stretch dominates the atomic MSE, and these terms weight what the bar's molecular rows measure, at
   no teacher call or extra pass (tried in Stage 0's water student, and falsifier 2's first fallback). Where a
   student meets the overall bar but fails the tail cap or one species' bar: a term on the worst 5% of per-atom
   force errors, added after warm-up, or per-species weights against each species' count and RMS force. The
   teacher's labels carry no noise, so the tail term fits model error, not label noise. Neither weighting comes with
   mace-torch; both go in the student's own loss code.
3. **Export and gate.** The bundle passes the engine's 3x gate against the student's float64 reference.
4. **Close the loop.** Run the student's MD, relabel a sample of its frames with the teacher, measure the fidelity
   bar there, add the worst frames and retrain until the bar holds on frames the student visits. Each training run
   keeps the EMA checkpoint taken before mace-torch's second phase (the learning-rate decay and raised energy
   weight); a round resumes from it with the added frames and runs the second phase again, rather than starting
   from scratch or fine-tuning a fully decayed minimum. The learning-curve runs of Stage 0 start from scratch, so
   that no warm start bends the curve.
5. **Curvature.** Compare finite-displacement phonons with the teacher's. Only if they miss: add displaced frame
   pairs `x +- eps v` and supervise the difference of their forces, not the two frames on their own:

   ```
   L_Hv = | (F_S(x + eps v) - F_S(x - eps v)) / 2 eps  -  (F_T(x + eps v) - F_T(x - eps v)) / 2 eps |^2
   ```

   Fed to the force loss as two ordinary frames, the pair would carry the student's force error at `x` in both, and
   that error outweighs the curvature term `eps dH v` it is meant to correct by a factor of order `1 / eps` (its
   square in the loss), so the optimiser would hardly see it. The difference cancels the error at `x` and leaves the
   Hessian-vector product (central difference, error of order `eps^2`). It costs two teacher calls per probe and two
   student force passes in one batch, no more than force training: no third derivative is needed. The probes `v` are
   dense: a random sign (or a Gaussian) on every coordinate, so each pair of calls gives every atom a full-size
   signal. A single-atom displacement would spend the same two calls on one column of the Hessian, whose weight
   lies near the diagonal (a pair term held 72-74% of the norm of OMAT-0 medium's layer-1 Hessian, list 12).
   Single-atom probes stay as a diagnostic, since they compare directly with finite-displacement force constants.
   `eps` about 0.01 A per coordinate: with float32 labels from the engine (force error about 1e-5 eV/A) the label noise in `L_Hv` is then
   about 5e-4 eV/A^2, against force constants of order 10 eV/A^2. No anharmonicity check on `eps` is needed:
   student and teacher are differenced at the same points, so what a finite step adds is the same quantity on both
   sides, and a mismatch is a real one. Its size is small anyway: about `(a delta)^2` relative for a Morse O-H
   (`a` about 2.2 per A), 5e-4 at a bond change `delta` of 0.01 A and 6e-3 at the 0.035 A that dense probes reach at
   worst. For a RESPA fast force the probes are the stiffest
   normal modes instead (section 12). PFT (arXiv:2601.07742) fits the same quantity with
   autograd Hessian-vector products against DFT force constants; the finite-difference form keeps the teacher a black
   box.

The scripts of lists 15 and 16 lived in `.tmp_o/colab15/` and are not in the tree; Stage 0 rebuilds them as a kept
tool.

## 8. Operating modes

- **Student only.** The fast mode, covered by the fidelity bar.
- **Audited.** The teacher evaluates the current frame every `K` steps as a drift monitor and frame harvester. With
  a teacher call at about 35 student steps (estimate for MH-1), `K` = 500 costs about 7%. The audit threshold is a
  single frame's force RMSE above twice the fidelity bar (4% of the RMS force), a default to tune in Stage 0. The
  fraction does not grow with the cell: the teacher runs tiled, and both models cost about the same per atom at any
  size.

  **Stage 0, SrTiO3 only:** an exploratory rule (round the largest target-validation frame-relative force RMSE up to a percentage point) gave 3% in every leave-one-trajectory-out fit. In offline comparison, 0/60 nominal and 40/40 hard frames flagged; the original 4% flagged 0/60 and 33/40. The rounding rule was proposed after analysis. Keep 4% as the general default and qualify 3% prospectively on independent oxide trajectories. Neither the cost nor the effectiveness of a cadence `K` or a pre-call trigger was measured.
- **A per-atom trigger (Stage 3).** So that the teacher is called when the student leaves its data and not only on a
  clock, the kernels flag atoms by a distance in feature space, with the shortest interatomic distance as a second
  check. Ranges of the layer-0 scalar sums would not do: an atom can sit inside every radial range with a wrong
  angle. The candidates, each with a threshold set on held-out frames:
  - the baseline the others must beat, at almost no cost: each atom's energy (`E_i`, or `E1_i` alone) against the
    range its species took in training, since `P3` is a cubic and grows fast outside its data;
  - a per-species Mahalanobis distance of `h_i` from the training frames' mean and covariance: three- and four-body
    geometry through `P3`, the first 5 A only, about 4,000 multiply-adds per atom against the model's 300,000;
  - the same to the nearest of several clusters per species, with each cluster's threshold set from its own spread,
    as CALM does (arXiv:2609.40060). One Gaussian per species spans the empty space between distinct sites (an
    interface, two cation sites, a transition state) and calls an intermediate familiar;
  - a last-layer leverage `f^T (F^T F + s^2 I)^-1 f` (LLPR) on the features the energy is linear in: the readout's
    16 hidden units, which see the full 10 A, and `h_i`, since `E0_i = w0 . h_i` is linear too. On the hidden units
    alone it is about 256 multiply-adds per atom;
  - any of these averaged over each atom's neighbours with the cutoff weights, since a force depends on the
    neighbours' environments as well;
  - the two Mahalanobis forms on `h_i / (|h_i|^b + e)` with `b` 0.5 or 1, with a per-species two-sided range on
    `|h_i|`, a few operations per atom more. Normalising features helps image classifiers whose feature norms vary
    for reasons other than novelty; whether `h_i` behaves so is what the scoring shows.

  Stage 0 scores them by how well they rank the teacher's per-atom force error on its student-visited frames
  (precision against recall); the trigger is chosen there, not now. A two-head readout is not a candidate: its heads
  share every feature and tend to agree off the data.
- **Large cells.** A frame of a million atoms cannot go into mace-torch training. When the trigger fires there, the
  harvest is local: spheres around the flagged atoms, labelled by the teacher, with the force loss on an inner core
  only. A force reaches twice as far as an energy: atom `i`'s energy depends on the atoms within `L r_c` of it (`L`
  layers of cutoff `r_c`), and atom `k`'s force sums the derivatives of every such energy with `i` within `L r_c` of
  `k`. A core atom's force is therefore that of the cell only when the sphere's radius is the core's plus
  `2 L r_c`: 27 A for a 3 A core and a two-layer 6 A teacher, about 6,900 atoms of SrTiO3. The sphere is labelled
  at that radius in LoupMD and cropped for training to the core's radius plus the student's `2 L r_c`, 23 A for a
  two-layer 5 A student, about 40% fewer atoms. The force constants fall off fast (a pair term held 72-74% of the
  norm of OMAT-0 medium's layer-1 Hessian, list 12), so a smaller sphere may give labels well within the bar.
  Stage 3 cuts spheres from periodic frames, compares the teacher's core forces with its periodic ones, and takes
  the smallest radius whose label error stays below a tenth of the force bar. Energies and stresses of such
  clusters are not used. The student is fine-tuned on them at a low learning rate from the checkpoint of section 7
  item 4, with a share of the original periodic frames replayed with their energy and stress labels, since the
  clusters carry neither. Then the weights `(1 - alpha) old + alpha new` are swept over `alpha` from 0.2 to 1 and
  the value is chosen by the fidelity bar on Stage 0's evaluation set and on the harvested cores: forward passes
  for the forces, minutes of finite displacements for the phonons. Interpolation works because the fine-tune
  starts from the old weights. The million-atom distillation study of arXiv:2610.23371 harvests this way (13 A
  spheres with a 6 A core for MACE-MPA-0, by the review that cited it, which leaves the surface some reach into the
  core's labels); its figures were not checked for this document.
- **Exact RESPA (optional, the owner's call).** The student as the fast force and teacher minus student as the
  impulse gives the teacher's Hamiltonian exactly. Measured with `s64l0` on SrTiO3: 2.3-2.9x against Verlet at 2 fs
  at equal energy RMS (list 15 item 3). The outer step is the limit, 8-10 fs there and about 2 fs with hydrogen, so a
  cheaper student moves the gain toward 4x against Verlet at 2 fs and no further. It needs the device-side loop of
  list 14 item 5.3. A student trained for this role is the untested lever on the outer step (section 12). A
  published study of multiple time steps with a distilled fast model (DMTS, arXiv:2510.06562) reports gains of the
  same order on other systems and baselines.
- **Sampling at long steps.** For configurational averages only, and independent of the model. BAOAB with MH-1
  samples SrTiO3 at 10 fs within the noise of 1 fs (list 18 item 6): five to ten times fewer steps than a run at
  1-2 fs. Water needs SIN(R) or rigid molecules to pass 1 fs (lists 17, 18). To be checked again for each student
  and system.

## 9. Falsifiers

Each is checked before the stage that depends on it; a failure takes the stated fallback.

| # | Falsifier | Fallback |
| --- | --- | --- |
| 1 | The `s64l0` error on student-visited frames exceeds 1.5x its error on teacher frames after two rounds of the loop | more rounds and hotter frames; stop if a third round does not close it |
| 2 | A water student (hydrogen, a teacher such as OFF24 medium) misses 2% | first the molecular loss terms of section 7 (no cost in speed); then vectors (`s64l1`'s form) or 96 channels |
| 3 | `C_msg` 16 loses more than 10% in force RMSE against `s64l0` | 32, then 64 (the baseline) |
| 4 | The scalar-only last layer misses the bar | keep l <= 3 outputs |
| 5 | Student MD at 1.5x the target temperature blows up or leaves the audit threshold within 100 ps | more hot frames, a stiffer core; no release for that system |
| 6 | Stage 0's census shows the small student's step is not bound by launches | drop the small-cell target; fuse for large cells only |
| 7 | A fused kernel loses its `A B B A` or a forced-variant gate | keep the generic-path kernels for that layer |
| 8 | Stage 1's best variant predicts under 1.5x over `s64r8` | the stop rule of section 4 |
| 9 | The four-species student misses the bar at `C_msg` 16 | 32 message channels, or species-dependent radial functions (section 5.1), for systems of that many species |
| 10 | The layer-0 launches at 10 columns stay near today's 0.37 ms at 5,000 atoms | the fused target drops to 0.65-0.70 ms (section 6.5); launch 2 is built only if it still clears the stop rule |

## 10. Roadmap

Effort figures are rough.

**Stage 0: baseline and workflow (about a week).** This stage is the first implementation plan.
- A kept distillation tool: teacher frames from the engine, training configuration, export, gate, fidelity report.
- `s64l0` and `s64r8` retrained, two seeds each (the ledger has one), with stress in the loss, and measured on
  student-visited frames; the radial input they were trained with recorded (section 5.1).
- The fidelity bar's open defaults set against the teacher: its spread over two MD seeds for the statistical
  observables, the percentile factor, the deterministic tolerances, and the NVE step at which the teacher conserves
  energy well.
- One `s64l0` run in float32 on the same frames and seed, with the energies' constant offsets removed in float64
  first. Float32 is taken for the later runs if its test error lies within the two float64 seeds' spread.
- Their census (`nsys`, 300 K) and step times at 320, 5,000 and 40,000 atoms on the 5060 Ti and the Ada, and their
  device memory (NVML) against section 6.4's count.
- A water student, a student for a system of four or more species (the owner's choice, decision 6), and the
  variants today's kernels already run: cutoffs 4.5 and 6 A, 32 and 96 channels.
- On each of the three systems, a fixed evaluation set of student-visited and hard frames, relabelled by the teacher,
  which Stage 1a and falsifiers 3, 4 and 9 use without running MD again; and the per-atom trigger's candidates
  scored on it (section 8).
- For the diffusive or disordered system, a training set with half the frames rattled with Gaussian noise and
  relabelled, against one from trajectories alone, measured by its transport observables. A study distilling
  foundation models for electrolyte transport (arXiv:2610.39513) reports that such a mix keeps transport accurate;
  not checked here.
- The learning curve of `s64l0` at 300, 1,000, 3,000 and 10,000 teacher frames, with epochs cut as the frames grow
  and the larger sets drawn as section 7 item 1 describes.
- Exit: section 6.5's first two columns measured; falsifiers 1, 2 and 6 answered; rows in the ledger and the status
  file.

**Between Stages 0 and 1: training backend.** We expect to build our own training backend before the Stage 1 ablations. This is not a Stage 0 deliverable. The [training-engine design](plans/loup-s-training-engine.md) starts with a whole-epoch mace-torch profile on frozen Stage 0 data, then replaces one measured bottleneck. It specifies the reference force/stress parameter gradients, optimizer update, checkpoint branches and time/CU-to-fidelity decision. Stage 1 can start on mace-torch if the candidate is not ready.

**Stage 1: the blocks in torch (about two to three weeks, mostly training time).**
- 1a, accuracy before export: train the student's interaction blocks using the selected backend, and run the ablations of section 5.2 with
  the per-atom-energy loss, about 31 runs on one seed on SrTiO3, each measured on Stage 0's evaluation set. No
  exporter work yet. Then, since which paths and channels a model can drop depends on the site symmetry (the owner's
  paper, arXiv:2611.21683; on the ideal lattice Sr and Ti keep scalars only, `meas-tp-path-pruning`), the ablations
  that cubic SrTiO3 may flatter (`C_msg`, l <= 2, the radial first hop, the radial input) and the leading variants
  run again on water and the four-species system: about 16 more runs.
- 1a's compute, about 100-140 GPU-hours at full length, is cut three ways (estimates). Successive halving: every
  variant to 30% of its epochs, then only those within 1.2x of the front of fidelity against estimated cost to the
  end, with three or four of the pruned ones run to the end as well, to check that the early order holds through
  the second phase with its raised energy weight. Warm starts: the variants that keep the trunk (the pair term, the
  per-atom energy loss, the curvature loss) start from the converged default and are compared with the default
  fine-tuned for as many epochs; the dual cutoff does not, since its Bessel basis changes with the cutoff. Float32,
  if Stage 0 takes it. Together about 30-45 GPU-hours.
- 1b, for the three or four variants on the front of fidelity against estimated cost: a second seed, the closed loop
  of section 7, and the exporter and loader changes of section 6.1 for the forms they need. Each variant the engine
  can run is exported and timed on the generic path; the others (l <= 2, the pair table, the dual cutoff) get a step
  estimate from section 6.5's rates. A correlation-4 variant on the front first goes through the product emitter
  (`gen_product.py`) and its gate.
- Exit: a table of fidelity against measured step time and against the fused estimate; the defaults fixed; the stop
  rule evaluated against `s64r8`. The table also decides the kernels of Stage 2: two launches if the radial first
  hop holds, layer 0's template for both layers if the factorised messages do.

**Stage 2: fused kernels (two to four weeks).**
- First the layer-0 launches (1 and 3) at 10 columns, benched against today's layer-0 kernels (falsifier 10); then
  launch 2, its block shape and staging form decided by a bench (section 6.2).
- Generator templates for the launches, the float64 reference executor for the new blocks, forced-variant gates,
  `A B B A` on the owner's GPUs. The lean edge buffers of section 6.4, with the virial in launch 2, belong here.
- Exit: the speed bar, or the generic path kept with its measured time.

**Stage 3: modes.** Audited MD in the package with the per-atom trigger chosen in Stage 0 and the local harvest
for large cells (section 8); the exact-RESPA device loop if the owner wants it.

**Stage 4 (optional).**
- Long range, only if a short-range student misses the bar on a POLAR-1 teacher: an additive latent-charge term on
  the D3 mesh (`src/dispersion/d3_mesh.cuh`), evaluated on an outer step as an exact additive split.
- A foundation-scale study: per-pair tables grow with the square of the species count and `C0` with the count, so the
  element handling has to change; decided after Stage 1's table.

## 11. Risks and decisions for the owner

**Risks.**
- The evidence for the bar is one system, one teacher and one seed per student, on teacher-trajectory frames.
- A student knows only the frames it was given. The audit and the hot frames bound that; they do not remove it.
- A student inherits its teacher's errors.
- Narrow messages are the least tested choice; nothing in the ledger trains them, and three species may hide what
  six would show.
- Fused kernels may tie, as several receiver-owned forms did, and launch 2 may be bound by occupancy (section 6.2).
- Half of the fused estimate rests on layer-0 kernels that are bound by latency today.

**Decisions.**
1. The GPUs an adoption must win on (5060 Ti only, or with the Ada and the A4000).
2. Whether the exact-RESPA device loop is built (Stage 3).
3. Rigid water or mass repartitioning for hydrogen systems: both change the model.
4. Where training runs (Colab L4 so far).
5. Whether Stage 4's foundation study is wanted at all.
6. The third test system: four or more species, ideally disordered or diffusive, with a teacher LoupMD runs.

## 12. Beyond Loup-S

Candidates for a more efficient model once Loup-S is measured. None is tested here, and the sizes named are
judgments, not estimates from measurements.

**Where the headroom is.** Loup-S's own arithmetic is about 0.3 million multiply-adds per atom-step (a count from the
widths of section 5.2), some 70% of it in the node maps and products. At the rate the engine's GEMMs reach, 65-70%
of the FMA pipe, that is about 0.05 us per atom: a factor of two to three below section 6.5's fused estimate. So the
same model has engineering headroom, and going well below 0.05 us needs less arithmetic: fewer edges, fewer channels
or a smaller basis.

| Candidate | Why it could pay | What the ledger says | First test |
| --- | --- | --- | --- |
| Degree-dependent cutoffs, trained in | l >= 2 terms and the messages on a short cutoff, the long shell for l <= 1 only (NEP separates its radial and angular cutoffs this way) | Masking a trained model fails by 270-4,250x (list 17 item 10): trained models put weight on every path out to the cutoff. A model trained that way is untested | Train `s64r8` with a second envelope on the l >= 2 columns. Today's tables can hold it; the kernels would not yet skip the edges |
| Models shaped per system by cost-aware sparsity | Gates on units a kernel can skip (channel prefixes per species and degree, whole paths), trained with the engine's cost per unit as the penalty: the ranking of arXiv:2611.21683 as part of training | Ranked columns are twice as good a unit as paths; 30% of them off costs 15-25 meV/A without retraining; on the ideal lattice Sr and Ti keep scalars only (`meas-tp-path-pruning`) | Distil with the gates and compare with hand-set widths at equal step time |
| One strong local layer | Half the edge passes of two layers | A linear two- and three-body ACE reaches 48.7 meV/A (list 15); higher order with the MLP readout is untested | The one-layer ablation of section 5.2 |
| Layer 0's edge kernels at 8 columns | They are bound by latency, not arithmetic, and were built for 66 columns; they are half of the fused estimate (section 6.5) | `meas-conv-ablation-bounds` | The first bench of Stage 2 |
| Displacement expansions for crystals | Force constants to fourth order cost far less than any neighbour-sum model | Nothing; valid only while atoms keep their sites | As the fast force of exact RESPA on SrTiO3, where the outer step caps the gain near 4x |
| Two fidelities in one cell | The full model near defects or interfaces, a cheap one in the bulk, blended smoothly | ONIOM is a workflow, not built (list 17 item 8) | Only for cells that are mostly bulk |
| Weights on the binary16 grid | A model trained with FP16-exact weights, or defined so at export, is still exactly smooth and conservative, and `hgemm2`'s `H(A) L(B)` product vanishes: a quarter less tensor work on GeForce cards, a third on professional ones (by the accounting in `src/hgemm2.cu`) | Rounding the weights alone moves an FP32-trained model by 19x the yardstick, 0.3-0.6 meV/A (list 17 item 2). Only GEMM-bound models gain, so not Loup-S | A kernel instance without the `L(B)` panel on MH-1's shapes. For a foundation model it is the owner's decision, since the reference changes |
| A wide student trained for single-pass FP16 | Capacity in wide node maps on tensor cores, the opposite of Loup-S's narrow FP32 kernels | On an FP32-trained model single-pass FP16 leaves a largest force error of 1-4 meV/A (82-149x the yardstick, list 17 items 1, 2), the activations being the sensitive operand. FP16 runs MACE-shaped GEMMs at 27-44 TFLOP/s against 11-14 for FP32 on the 5060 Ti (`meas-tc-overlap-ozaki-locality`) | Fake-quantised training in torch; force noise and NVE drift against the FP32 student at equal step time |
| A student trained as the RESPA fast force | The resonance between a third and a half of the fastest period grows with the slow force's curvature on stiff modes. A Hessian-vector loss on those directions shrinks it, and constrains nothing in the total, which stays the teacher | `s64l1` holds 10-12 fs outer steps better than `s64l0` (RMS 0.015 / 0.023 against 0.019 / 0.050, list 15). Splits of one model not trained for it fail (lists 12, 17, 18). The cap follows the system's fastest mode: about 2 fs with hydrogen | The curvature share of teacher minus student on the stiffest modes, before and after; NVE at 2 x 6 and 2 x 8 on SrTiO3 |
| Factorised messages (an ablation of section 5.2) | Layer 1 in layer 0's form: `A[(n, c), lm] = sum over j of phi_n Y_lm x_j[c]`. One 8-column radial table serves both layers, a channel thread reads one table value for four, and every sender scalar reaches every node channel with a radial shape of its own | Layer 0 runs this way already. Rank-8 radial heads cost nothing and rank-4 12% (list 16 item 5); the form was turned down for foundation models, which need rank 20-44 at 128 channels (list 16 item 4). Free table reads are worth 26-40% of a forward conv (`meas-conv-ablation-bounds`) | Train with the radial weight tied across degrees. It runs on today's kernels as a constrained table, so the accuracy is known before a kernel is written |
| A radial first hop (an ablation of section 5.2) | Sender scalars from pair sums `rho_j[s, n] = sum over k of phi_n(r_jk)`, formed in the compaction pass that already visits every list edge. One angular pass each way then replaces two | `first_fwd`'s angular update is 1% of its time, so a radial pass of its own would cost a full one: it has to ride on `scatter_active_edges` (27 us a call). The paths that read hidden vectors are the cheapest to lose (`meas-tp-path-pruning`); senders with radial information only are untested | Train with layer 0's l >= 1 projections held at zero; if the bar holds, Stage 2 builds two launches |
| Replicas of a small cell in one launch | Small cells are bound by launches: `s64l0` takes 0.52 us per atom at 320 atoms against an estimated 0.35 at 5,000. Independent copies in one set of launches would recover the large-cell rate for ensembles, beads and images | Multi-cell batches are a feature not built (list 10 item 7B); two engines on two threads gain 1.12-1.13x on NaCl216 (list 18 item 10) | A batched system in the engine, after Stage 2 |

On the size of what is left. That no force model is ten times cheaper than a well-built small local one at the same
fidelity is a judgment, not a result. It rests on one system's accuracy-cost points (a linear ACE at 48.7 meV/A,
students at 8-10) and on a floor: any model must visit its neighbours, and a pair potential, which does little
else, is itself only some ten to twenty times cheaper than section 6.5's fused estimate (from published GPU
benchmarks, not measured here). Stage 1's table of fidelity against step time is the test: a flat curve would
refute the judgment. Larger factors exist outside the force model, each at a price: long steps for sampling
(section 8), accelerated dynamics for rare events, learned long-stride propagators without conservation, and
displacement models for crystals near equilibrium.

On time stepping. The ledger's failed splits are of models that were not trained to be split, so they do not show
that a trained split fails. Two things training does not change. The resonances sit at fractions of the fastest
period of the whole system (39 fs for SrTiO3 here, about 9 fs with O-H), and training only weakens them: the outer
step stays near a quarter to a third of that period. And the fast part must carry all the stiff curvature, including
its dependence on the far environment: OMAT-0 medium's layer 1 kept 15% of it after the best pair term was moved out
(list 12). The gain also needs a slow part far dearer than the fast one. StrataEQ-MD's medium-range stratum was 45
times its short-range one. Inside Loup-S a 3.5 A one-layer fast part would cost about 0.35-0.4 of the fused student
(estimate: the same fixed costs, a third of the edges), so four inner steps per outer one give at most 1.6x where
plain Verlet needs 0.5 fs and about 1.1x on a crystal that already runs at 4 fs. That is why the split is kept only
between student and teacher.

On precision. The ledger's failures are of FP32-trained models held to the 3x rule, which is two to three orders
tighter than a student's fidelity bar. Training can absorb anything that is a fixed property of the weights. It
cannot remove the rounding noise of activations and gradients at run time, and a force is a gradient that MD
integrates over millions of steps: that noise is a non-conservative force. At FP16 it is small (an estimated heating
of the order of 1 meV per atom per ns in NVE, nothing under a thermostat); it rules out finite-difference phonons.

Not candidates, each for a measurement in section 2.2 or the ledger: edge-frame convolutions at l <= 3, a Cartesian
basis in place of the spherical one (layer 0's angular arithmetic is 1% of its kernel), four-bit or block-scaled
formats on the signal path (a 14-bit block-scaled screen did worse than 11-bit floating point: 588-1,498x against
82-149x), and non-conservative or long-stride propagators.

## 13. Evidence index

Ledger entries and status sections this document rests on.

- **Speed and capacity:** STATUS 5.3 (family table, S20, S40, S144, S145), 6.23, 6.34, 6.35, 6.62, 6.63;
  `docs/validation/capacity.md`; `meas-rest-vs-300k`.
- **Kernel bounds:** `meas-conv-ablation-bounds`, `meas-conv-bwd-algebra-bound`, `meas-s142-bounds`,
  `meas-tc-overlap-ozaki-locality`, `meas-small-cell-idle`, `meas-mh1-conv-l2`, `s24-l0-conv`, `int1-l2-chunks`,
  `i0-bwd-receiver-owned`, `edge-compaction-1pass`, `gen-gconv-cfix-preloc`, `tile-device-compact`.
- **Precision:** `gen-two-half-jam`, `meas-radial-precision-screens`, `meas-degree-channel-shell-screens`,
  `meas-ada-power-sustained`, `ideas-20261121-ws`, `ideas-20261123-list32`, `ideas-20261123-list33`;
  `src/hgemm2.cu`'s header.
- **Folds, pruning and run-time compilation:** `gen-chain-fold`, `gen-chain-fold-ext`, `pol-dot-fold`,
  `gen-conv-prune`, `gen-prod-scalar`, `gen-jit-conv`, `gen-jit-products`, `meas-tp-path-pruning`.
- **Students and time stepping:** `ideas-20261121-list10` (multi-cell batches), `list12` (layer split, the pair
  term), `list13` (dual-model RESPA), `list14`, `list15` (students, linear ACE), `list16` (radial rank, basis-space
  messages), `list17` (shell split, rigid water, undirected conv, tiles, k-space, FP16 screens),
  `ideas-20261122-list18` (angular split, PCA, low rank, SIN(R), two replicas).
- **Long range:** STATUS 6.31 (D3 mesh); the "POLAR k-space on a mesh" row of the ledger's open table.
- **Code read for this document:** `GenericModel::alloc_system` and `ensure_edges` (`src/generic.cu`), the edge
  buffers in `src/engine.cu`, `loupmd/export/generic.py` (layer widths, `J = H + 2`).
