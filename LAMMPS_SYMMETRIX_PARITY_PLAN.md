# LoupMD in LAMMPS: the options of Symmetrix's pair styles

Status: plan only (19 November 2026). Nothing in it is built. Effort figures are *estimates*. Measured figures are
from the 5060 Ti under WSL 2 and are quoted from `LAMMPS_KOKKOS_PLAN.md`.

Goal (owner, 19 November): the LoupMD package for LAMMPS offers the same options as Symmetrix's, ideally without
losing speed.

Reference: Symmetrix `main` at `b907f04` (10 November 2026, v0.3.0), `pair_symmetrix/`, as checked out in the
benchmark environment (`~/loup_benchmark_env/src/symmetrix`). Its option list below is read from the `settings()`,
`coeff()` and `init_style()` of each style and from `pair_symmetrix/README.md`.

## Summary

- Symmetrix's MACE option surface is: three run modes, three precisions, per-atom energies, any boundary, several
  processes and GPUs, CPU execution, and seven keywords of its MACE-POLAR style.
- Four of these cost LoupMD nothing in speed and little work: the mode keywords, per-atom energies, open boundaries,
  and the `float32` / `float32-mixed` names.
- Two are large pieces of engine work: several processes (two modes) and the POLAR mesh.
- Two cannot be had "at no loss": double precision and CPU execution. They need a decision before any work.
- The release-branch engine already contains several building blocks. So the first step is to bring the LAMMPS work
  onto that engine, not to rebuild those blocks on `develop`.

## The speed rule

The default stays what it is today: one process, a periodic cell, the engine's own neighbour list. In Symmetrix's
terms that is `no_domain_decomposition`.

1. Every new option is a mode chosen once in `init_style`, with its own captured graphs. No option may add a
   kernel, a conditional node or a host exchange to a step of the default mode.
2. After every stage the three default configurations (`run_style loupmd`, `loupmd`, `loupmd/kk`) are timed
   `a b b a` against the stage's parent on OFF23-S water216, SrTiO3 with 320 atoms and SrTiO3 with 5,000 atoms. A
   stage is accepted only with b/a inside the run-to-run spread.
3. The Symmetrix side-by-side job (`in.sym_md` protocol) stays as the reference number:

   | ms/step, OMAT-0 medium SrTiO3 | 5,000 atoms | 320 atoms |
   |---|---|---|
   | `run_style loupmd` | 7.81 - 7.86 | 0.61 - 0.64 |
   | `pair_style loupmd` | 8.20 | 0.86 - 0.90 |
   | `pair_style loupmd/kk` | 8.38 - 8.45 | 1.29 - 1.38 |
   | Symmetrix `symmetrix/mace/foundation/float32/kk` | 13.47 - 13.54 | 1.55 - 1.59 |

4. A new mode gets its own target where a like-for-like Symmetrix number exists (same mode, same process count,
   same GPU).

## Symmetrix's options, one by one

| Symmetrix option | what it does there | LoupMD today (`develop`, S145) | plan |
|---|---|---|---|
| `pair_style … no_domain_decomposition` (default on one process) | one process; LAMMPS's full list, ghosts mapped back to owners | this is LoupMD's only mode, with the engine's own list | P1: accept the keyword |
| `pair_style … mpi_message_passing` (default on several processes) | hidden features exchanged between layers through LAMMPS's forward / reverse communication; ghost cutoff = model cutoff | refused (one process only) | P4b |
| `pair_style … no_mpi_message_passing` (general `symmetrix/mace` styles only) | each process evaluates owned + ghost atoms; communication cutoff raised to 2 × cutoff + skin | refused | P4a |
| several processes and GPUs (`mpirun -np 4 lmp -k on g 4 -sf kk`) | through the two modes above | one process, one GPU | P4 |
| precision: plain name = float64, `/float32`, `/float32-mixed` (GPU foundation styles) | three arithmetic variants | one arithmetic: float32 kernels, tensor-core GEMMs, float64 accumulation (`LOUP_NO_HYBRID=1`: float32 GEMMs) | P5 |
| CPU execution (styles without `/kk`, OpenMP; `/kk/host`) | runs without a GPU | needs a CUDA GPU; the non-Kokkos style still computes on the GPU | P8 (decision) |
| per-atom energies (`compute pe/atom`) | yes | no | P2 |
| per-atom virials (`compute stress/atom`) | no (refused) | no (refused) | same already |
| boundaries | any (`p`, `f`, `s`, `m`), through LAMMPS's lists and ghosts; POLAR: periodic only | three periodic dimensions only | P3 |
| `newton on` required | yes | either | same or better already |
| `atom_modify map yes` required | yes | yes | same already |
| `pair_coeff * * model.json El1 El2 …` | one element per type | `pair_coeff * * model.loup El1 El2 …` | same already |
| model file from `symmetrix_extract_mace -m MODEL -s H O` / `-Z 1 8`, `--head`, `--r-min`, `--num-spline-points`, `-o` | JSON with splines | `python/export.py --model … --elements 1,8 --output …`, `--head`, `--charge`, `--spin-multiplicity` | P1: one command with the same flags |
| `symmetrix/mace/foundation*` (fixed shapes of the foundation models) | faster kernels for listed architectures | every supported family runs LoupMD's one path | nothing to do: no separate style needed |
| `symmetrix/mace/polar`: `charge Q`, `spin S` | total charge and spin multiplicity of the cell | POLAR-1 neutral singlet only | P6a |
| … `mesh h`, `order P` | smooth particle-mesh Ewald grid for the electrostatics | direct sums over the model's k-vectors | P6b |
| … `direct` | direct k-sums (one process) | what LoupMD does | P1: accept the keyword |
| … `grid lammps\|local` (`/cuda`) | the mesh whole on one GPU (cuFFT), or LAMMPS's distributed grid | – | P6b (`local`), P6c (`lammps`, with P4) |
| … `float32-mixed` (`/cuda`) | tensor-core GEMMs for POLAR | tensor-core GEMMs are off for long-range models: they failed POLAR accuracy gates (ledger `gemm-tc-rounded-polar`) | P6d (only if it passes the gate) |
| `symmetrix/nequip*`, `direct_exchange [fp16_wire]` | NequIP models; direct GPU-to-GPU feature exchange exists only for these | not a MACE option | P9 (decision) |

LoupMD options with no Symmetrix counterpart stay: `device N`, `skin S`, `ghost yes|no`, and `run_style loupmd`
(Symmetrix has no integrator of its own).

## What the release engine already has

Read from `include/loupmd.h` on `release/v1` (`ae347d4`); none of it is on `develop`:

- **Per-atom energies:** `loup_evaluate2` with `LOUP_EVAL_PER_ATOM` (one more energy-only pass; refused for
  long-range models).
- **Open domains with ghost atoms, evaluated layer by layer:** `loup_debug_set_domain` /
  `loup_debug_evaluate_domain`. The first `n_local` atoms receive, the ghosts only send, and the ghosts' share of the
  forces is folded back. Its tiled runs evaluate a large system as such domains, one sweep per layer. That looks like
  the scheme of `mpi_message_passing` with tiles in place of processes; to be confirmed in the code before P4.
  Marked "testing only" there.
- **Eight integrators and a minimiser:** `loup_run` (NVE, Langevin, CSVR, Nosé-Hoover chain, Berendsen NVT / NPT,
  BAOAB NPT, MTK NPT), `loup_minimize`, `loup_set_fixed_atoms`, `loup_set_cell`, and a variable-cell step that forms
  the Cartesian shifts in every step.
- **A B-spline mesh with FFT** for the reciprocal part of its D3 term (`LOUP_D3_PME_ORDER`).
- **Errors as text** (`loup_last_error`), `loup_get_info`, an ABI version.

`develop` has what the LAMMPS styles need and release lacks: `loup_compute_device`, `loup_evaluate` as one graph
launch, run-graph reuse, and the optimisations 0.0.2 is waiting for.

## Stages

### P0. One engine (prerequisite)

Bring the S145 engine work and `lammps/` onto the release engine, with or after the merge of the `develop`
optimisations that 0.0.2 is waiting for.

- Port, not copy: release's `engine.cu` has its own step graphs (variable cell, tiled runs), so `loup_compute_device`
  and the one-graph `loup_evaluate` have to be fitted to them.
- Gain on the way: `fix npt` no longer rebuilds the list every step (the variable-cell step), and `run_style loupmd`
  can offer NVT and NPT at the engine's step time (P7).
- Done when: the 101 tests of S145 pass on the merged library, and the speed rule's three configurations are within
  spread of `develop`'s.

### P1. Same words (no engine work)

- `pair_style loupmd [no_domain_decomposition | mpi_message_passing | no_mpi_message_passing] [device N] [skin S]
  [ghost yes|no]`, with Symmetrix's defaults by process count. Until P4 exists, the two multi-process modes stop with
  a message naming this plan.
- POLAR keywords on the same style when the bundle is a POLAR model: `direct` accepted now; `charge`, `spin`, `mesh`,
  `order`, `grid`, `float32-mixed` parsed and refused with the stage that will add them.
- One bundle command with Symmetrix's flags: `loupmd lammps bundle -m MODEL (-s H O | -Z 1 8) [--head H] -o x.loup`.
  `--r-min` and `--num-spline-points` are accepted and reported as not needed: LoupMD builds and certifies its own
  radial tables (`kPackedTableMaxRelErr`).
- Speed: none at stake.

### P2. Per-atom energies

- Engine: release's `LOUP_EVAL_PER_ATOM` through `loup_evaluate`'s LAMMPS path and through `loup_compute_device`
  (the reserved `atom_energy` field). Add it for long-range models, which release refuses.
- Pair styles: `eflag_atom` fills `eatom`. `compute pe/atom` sums to `pe` (to the 6e-7 relative that release
  documents).
- Speed: only the steps that ask for per-atom energies pay the extra energy pass. The default step's graph is
  unchanged.

### P3. Any boundary

- Open axes (`f`, `s`, `m`) as a padded periodic cell:
  - Along each non-periodic axis the pair style gives the engine a cell that spans the atoms plus more than
    cutoff + skin of vacuum on both sides. This is what release's open domain does ("a box around them all with room
    for cutoff plus skin on every side").
  - The padded length grows in fixed steps (for example 2 Å) and only when atoms come within the margin. So the cell
    does not change from step to step and the list is not rebuilt for it.
- No engine change is needed for correctness. A per-axis periodicity flag in the list builder would save the empty
  bins and can follow if the padding shows up in timings.
- POLAR stays periodic-only, as in Symmetrix.
- Speed: periodic runs are untouched.
- Tests: a molecule in `boundary f f f` against the same molecule in a large periodic box; a slab with `p p f`.

### P4. Several processes and GPUs

Both modes build on release's open domain. Both need it as a supported interface instead of a debug one, and both need
an engine system whose atom count changes at every reneighbouring without a full `loup_set_system`:

- capacity buckets for the species-grouped GEMM shapes, so the captured graphs are re-recorded rarely;
- the device-side row map that `loup_compute_device` already has.

**P4a. `no_mpi_message_passing` (ghost shell).**
- LAMMPS supplies owned atoms and ghosts within `n_layers × cutoff + skin`.
- The engine evaluates one open domain: owned atoms and the ghosts within one cutoff of them receive, the outer
  ghosts only send.
- The energy is the owned atoms' sum (P2's per-atom energies, with the backward pass seeded by the owned mask).
- Forces come back on every atom and LAMMPS's reverse communication adds the ghosts' share.
- This is the simpler mode and the slower one: the shell holds several times the owned atoms on small domains.

**P4b. `mpi_message_passing` (features exchanged between layers).**
- The ghost cutoff is the model's cutoff.
- The force pass splits at every interaction layer:
  - forward: the layer's sweep on the owned atoms, then LAMMPS's forward communication of the hidden features into
    the ghost rows (`pack_forward_comm_kokkos` on device buffers);
  - backward: the mirrored sweeps with reverse communication of the feature gradients.
- `comm_forward` = `comm_reverse` = the feature row width, as in Symmetrix.
- With `loupmd/kk` the buffers stay on the GPU (GPU-aware MPI, or LAMMPS's host staging). The plain style passes them
  through host memory.

**Speed.**
- One process is untouched: the modes are separate graphs.
- For the modes themselves the yardstick is Symmetrix in the same mode with the same process count.
- This machine has one GPU, so only "two processes on one GPU" can be measured here. Scaling needs the workstations
  or A100 nodes.

**Order.** P4a first (it exercises the variable-count system and the domain interface), then P4b.

### P5. Precision names

| Symmetrix | LoupMD equivalent | work |
|---|---|---|
| `/float32-mixed` (tensor-core GEMMs) | today's default arithmetic for models without long range | a style name |
| `/float32` | float32 GEMMs only: today `LOUP_NO_HYBRID=1` for the whole process | make it a per-handle setting, add the style name |
| plain name (float64) | does not exist | see below |

- Double precision would be a second kernel set: the generators emitting `double`, DGEMMs, and its own gates. It
  would run at a fraction of the float32 speed on consumer GPUs. It does not make the default slower, but it cannot
  be "the same option at the same speed".
- LoupMD's accuracy statement today is the gate: within 3× of MACE's own float32-against-float64 error.
- Open question for the owner: what the unsuffixed `pair_style loupmd` should mean (below).

### P6. MACE-POLAR options

- **P6a. `charge Q`, `spin S`.** The engine evaluates POLAR-1 for neutral singlets only. Work: the total charge and
  spin inputs of PolarMACE through the export and the SCF, checked against PolarMACE float64 references for charged
  cells. First step: read how PolarMACE applies them, before estimating.
- **P6b. `mesh h`, `order P`, `grid local`.**
  - A smooth particle-mesh Ewald path for the structure factors and potentials on one GPU (cuFFT), reusing the mesh
    code release has for D3.
  - This is also a speed item: the direct sums grow with atoms × k-vectors, the mesh with atoms plus grid points.
  - To lose no speed, the default stays whichever is faster for the system, timed at the first rebuild as the conv
    variants are. `direct` and `mesh` force one.
  - Accuracy: the mesh result against `direct` within the gate's limits, with Symmetrix's defaults (0.7 Å, order 8)
    as the starting point.
- **P6c. `grid lammps`.** LAMMPS's distributed grid and FFTs for several processes (KSPACE package). With P4.
- **P6d. `float32-mixed`.** Tensor-core GEMMs for POLAR failed LoupMD's gates on some structures. The keyword is
  offered only for the model / GPU combinations that pass; otherwise it stops with the reason. "Accuracy decides"
  (owner, 15 November) stays the rule.

### P7. Engine-speed thermostats and barostats (not a Symmetrix option)

`run_style loupmd` with release's integrators: `nvt` (CSVR, Nosé-Hoover chain), `npt` (MTK, BAOAB), fixed atoms. This
is what keeps "no speed lost" true for users who need more than NVE and Langevin: Symmetrix users get those from
LAMMPS fixes at LAMMPS's per-step cost, and LoupMD users would get them at the engine's step time. After P0.

### P8. CPU execution (decision)

Symmetrix's styles without `/kk` run on CPUs. LoupMD is a CUDA engine. Its non-Kokkos style runs in a CPU build of
LAMMPS but still needs the GPU. A CPU engine would be a separate project (other kernels, other performance work), not
an option to add. Recommendation: not planned; say so in the user guide.

### P9. NequIP styles (decision)

`symmetrix/nequip*` and `direct_exchange [fp16_wire]` serve NequIP models. LoupMD evaluates MACE models.
Recommendation: not planned. `direct_exchange` exists in Symmetrix only for NequIP, so MACE option parity does not
need it. A GPU-to-GPU exchange for P4b can be weighed once P4b is measured.

## Order and effort (*estimates*)

| stage | work | depends on |
|---|---|---|
| P0 one engine | 1 - 2 weeks | the 0.0.2 merge decision |
| P1 same words | 2 - 3 days | – |
| P2 per-atom energies | 3 - 4 days | P0 |
| P3 any boundary | 3 - 5 days | – |
| P5 `float32` / `float32-mixed` names | 2 days | – |
| P7 run-style thermostats and barostats | 1 week | P0 |
| P4a ghost-shell mode | 2 - 3 weeks | P0, P2 |
| P4b feature-exchange mode | 3 - 5 weeks | P4a |
| P6a POLAR charge and spin | 1 - 2 weeks after a first look | – |
| P6b POLAR mesh, one GPU | 2 - 3 weeks | P0 |
| P6c POLAR distributed grid | 1 - 2 weeks | P4b, P6b |
| P5 double precision | 4 - 8 weeks | owner's decision |

Suggested order: P1, P3 and P5's names now on `develop` (no engine dependency), P0, then P2 and P7, then P4a, P4b,
P6.

## Open points for the owner

1. **Base.** Build the engine-dependent stages on the release engine (P0), as recommended, or on `develop`?
2. **The unsuffixed name.** In Symmetrix the plain style is float64. Should `pair_style loupmd` stay LoupMD's fast
   default (recommended, printing its arithmetic at start-up), with `/float32` and `/float32-mixed` added, or should
   the plain name mean float64 as in Symmetrix, which needs the double-precision engine?
3. **Double precision** at all (P5)?
4. **CPU styles and NequIP** (P8, P9): confirm they are out of scope, or say which matter.
5. **Hardware for P4.** Two GPUs are needed to measure scaling: the workstations or A100 nodes, when available.
6. **POLAR `float32-mixed`** (P6d): offer it only where it passes the gate, as proposed?
