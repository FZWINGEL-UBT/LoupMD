# LoupMD public release: master implementation plan

> **How to use this plan:** it fixes the decisions, the target architecture, the interfaces and every task with
> its acceptance criteria. Before a stage starts, write its step-level plan as `plans/stage-<n>-<name>.md`
> (test first: failing test, run, implement, run, commit). Task ids such as `S2-T04` are the tracking ids.

**Goal:** release LoupMD as a pip-installable package that has FastMACE's feature set and front-end shape, needs no
PyTorch at run time, benchmarks itself with one command, and includes a tiled and wavefront capacity mode for very
large cells in its first public release.

**Architecture:** three layers. A C++/CUDA core engine behind a versioned C ABI; a torch-free Python package
(`loupmd`) that carries FastMACE's API names, CLI verbs and output files; an optional export extra that converts
MACE checkpoints into LoupMD bundles. One core abstraction, a force domain with ghost atoms evaluated in three
sweeps, serves the whole-system call, the tiled call and the wavefront schedule.

**Tech stack:** C++17 and CUDA 13 (`nvcc`, cuBLAS, cuBLASLt, CUB); Python 3.10+ with `numpy` and `ase` at run time;
`torch` (any CPU build), `mace-torch==0.3.16` and `e3nn` for export only; `make` for the library, `python -m build`
for the wheel, `pytest` for tests, MkDocs Material for the docs.

Written 14 November 2026 on branch `release/v1` (worktree `D:\Margraf\LoupMD-release`), branched from
`round6-hardening` at `0336751`. Nothing in this plan has been built or measured; every number about LoupMD's future
behaviour is an estimate and says so.

---

## 1. Decisions

### 1.1 Made by the owner

| Id | Decision | When |
| --- | --- | --- |
| O1 | LoupMD is released as an easy-to-use package like FastMACE, with all of FastMACE's features and an easier self-benchmark | 14 Nov 2026 |
| O2 | The tiled and wavefront modes are in the first public release | 14 Nov 2026 |
| O3 | Dispersion covers the newer implementations, with the variant chosen by the user; research on what to implement for the best user experience comes before any dispersion code | 14 Nov 2026 |
| O4 | Dispersion scope after the research: the default is as fast as possible at the same accuracy (not a fixed cutoff); the three-body term is offered, off by default, with a 20 bohr cutoff in its preset; the mesh sum is in 1.0; D4 is left out | 14 Nov 2026 |
| O5 | An automatic mode may pick the dispersion method by speed, but every method stays selectable by name and an explicit choice is never replaced, because a method can be inaccurate for a specific system | 14 Nov 2026 |
| O6 | Licences: MIT for LoupMD's own code, LGPL-2.1-or-later for files ported from ASE, third-party notices as FastMACE's (settles A6) | 15 Nov 2026 |
| O7 | Total charge and spin multiplicity stay per bundle, folded at export; no run-time setter in ABI v1 | 15 Nov 2026 |
| O8 | An FP16 tensor-core instruction is allowed on the force path where the result is accurate (the hybrid 3xTF32 GEMM); the 3x gate decides | 15 Nov 2026 |
| O9 | A model is not refused for tiled execution because of its layer count; the owner asked why OMOL-0 extra large would be. The sweeps are per layer (§4.4) | 15 Nov 2026 |

### 1.2 Defaults this plan assumes (open; each can be changed before the stage that depends on it)

| Id | Question | Default taken | Changing it costs | Decide before |
| --- | --- | --- | --- | --- |
| A1 | Package name: `loupmd`, or `loup` (free on PyPI on 28 Oct, reserved in FastMACE's rename plan) | `loupmd` for distribution, import and CLI | a directory and `pyproject.toml` rename, if done before the first upload | Stage 7 |
| A2 | Keep the specialised OMAT-0 medium kernels (about 8% faster on `SrTiO3`: 8.28 against about 8.9 ms) | keep for whole-system runs; tiled runs use the generic path for every model | retiring it removes `interaction0.cu`, `generated_kernels.cuh`, `gen_kernels.py`, `export_omat0.py` | S6-T10 |
| A3 | FastMACE features left out | no float64 force mode, no fallback to the MACE calculator, no bitwise-reproducible mode; each is refused with a message that names the alternative | float64 and bitwise would each be a stage of their own | Stage 3 |
| A4 | CUDA 12 wheels | CUDA 13 only at 1.0 (as FastMACE) | a second build and test matrix | Stage 2 |
| A5 | Platforms | Linux x86-64 and WSL2; GPUs of compute capability 8.0 and newer. The T4 (7.5) is not supported: the own SGEMM uses `cp.async` and the hybrid GEMMs use TF32 tensor cores | T4 support needs fallbacks for both | Stage 2 |
| A6 | Licence | settled by O6 (15 Nov 2026): MIT for LoupMD's own code; LGPL-2.1-or-later for files ported from ASE; third-party notices as FastMACE's | | |
| A7 | Pre-releases | stages end in internal tags (`v1.0.0a1` ...); nothing is uploaded before 1.0.0 | none | Stage 7 |

### 1.3 Open points found while building (owner's instruction of 15 November 2026: document, recommend, keep working)

Each row is the owner's to decide. Work continues on the recommendation, which is the reversible choice in every
row; "Undo" says what a different answer costs.

| Id | Question | Recommendation, in use | Why | Undo |
| --- | --- | --- | --- | --- |
| D1 | Licence (A6) | **decided, O6:** MIT, and LGPL-2.1-or-later for files ported from ASE | | |
| D2 | When to move `python/` into `loupmd/` and `codegen/` (S0-T03) | not moved. New Python goes to `loupmd/`; `loupmd/_legacy.py` loads the bundle cache from `python/loupmd.py` | the owner named this as waiting for his go | none; the move itself is `plans/stage-0-ground-rules.md` |
| D3 | `loup_set_charge_spin` of §4.3 against the code, which folds charge and spin into the bundle at export | **decided, O7:** the call is dropped from ABI v1; charge and spin are arguments of `lm.load` / `ensure_bundle` and select a bundle | | |
| D4 | OMOL-0 extra large has three layers; §4.4's sweeps assumed two | **changed after O9:** not refused. The sweeps are defined per layer: a model with L layers takes 2L - 1 sweeps and keeps L - 1 layer states, each sweep with a halo of one cutoff. S1-T08, S1-T09 and Stage 5 are written for any L; OMOL-0 extra large joins their acceptance lists | the refusal came only from the plan's two-layer wording | extra effort, an estimate: about one week across S1-T08 and Stage 5. If the five-sweep wavefront proves too costly, the fallback is plain tiled for L = 3 and the wavefront for L = 2, put to the owner with numbers |
| D5 | §2 forbade FP16 "anywhere on the force path"; the default hybrid GEMM cubins use one FP16 tensor-core instruction for the cross terms of the 3xTF32 split | **decided, O8:** allowed where accurate. §2 now says so; the 3x gate is the test | | |
| D6 | `cubins/tf32x3_hyb_sm120.cubin` has no build record in either repository | **partly answered by the merge of `e21faa2`:** the developing session added its own `cubins/gemm_tc_*.cubin` with source, recipe and build record (`cubins/src/`), `gemm_tc_hyb_sm120.cubin` being the recorded rebuild of the default; the default `tf32x3_hyb_sm120.cubin` itself still has none. Recommended: make the recorded cubin the default once it is gated (the developing session's work); `NOTICE.md` lists both sets | | |
| D7 | The hybrid kernels overflow for matrix elements of 32752 and above (FastMACE `hybrid_f16.h`) | **done upstream (S65, merged with `e21faa2`):** a non-finite energy, force or stress with the hybrid loaded drops it for that engine and evaluates again without it; `loup_run` reports a non-finite energy after a block. A loader check on the weights is no longer planned | a silent `inf` would be worse than a slower GEMM | none |
| D8 | Version string while the stages run | `LOUP_VERSION "1.0.0a1"`, so `loup_version()` gives `1.0.0a1+<hash>`; `1.0.0+<hash>` at the release | a library built today should not call itself 1.0.0 | one line in the header and one in `loupmd/__init__.py` |
| D9 | `python/bench_md.py`, `bench_family.py`, `test_gate.py` still bind the library through `python/loupmd.py`, which cannot read `loup_last_error` | leave them until S0-T03; `LOUP_DEBUG_ERRORS=1` prints the reason | editing `python/` is what D2 holds back | a ten-line edit of `python/loupmd.py` |
| D10 | Baseline commit of S0-T01 | the merged code (`bac354c`), not `0336751` | the owner's order put the merge first; Stage 1 is compared with what it starts from | none |
| D11 | S1-T03's scope | besides keeping the graphs, a run call no longer makes a force pass before its first step when the forces on the device are current | the 2% criterion cannot be met otherwise: that pass is one force evaluation per call | one `if` in `run_md_loop` |
| D13 | S1-T11's criterion says water at 300 K pipelines over 5,000 steps. The ledger's S46 measurement says it cannot at skin 0.5 A: a hydrogen moves 0.08 A per 1 fs step (up to 0.11 A), and the pipelined loop needs four times that below skin / 2 | the decay is built as review item S2 asks; the criterion is tested where it can hold: water pipelines at skin 1.0 A and stays unpipelined at 0.5 A, one fast step is forgotten after two windows. For the front end, recommend `skin="auto"` in `lm.md` (S3-T07): measure the per-step maximum over the first steps and set the skin to 8 times it plus a margin, at least 0.5 A; the engine's default and the benchmark configuration stay at 0.5 A (ledger item md-skin stays the owner's) | S46: no decay changes water at skin 0.5 A; at skin 1.5 A water ran 2.90 against 3.13 ms/step | none: the skin is an argument |
| D16 | S1-T09's criterion asks the open domain to give the periodic energy to 1e-7 eV/atom. Measured on the RTX 5060 Ti (15 November, second run; a periodic cell with its images as ghosts), energy / forces / virial stress against the periodic call: OMAT-0 small 3.1e-7 eV/atom, 2.0e-6 eV/A, 2.9e-8 eV/A^3; OMAT-0 medium on the generic path 1.7e-7, 2.1e-6, 3.2e-8; MH-1 1.5e-8, 1.4e-6, 4.2e-9; OFF23 medium 1.0e-8, 7.4e-7, 1.1e-9; OMOL-0 extra large (three layers) 7.9e-10, 7.8e-7, 4.3e-10. For scale, the periodic call on other GEMM kernels (`LOUP_NO_HYBRID=1`) changes by 3.9e-10 to 6.3e-8 eV/atom and 5.5e-7 to 2.5e-6 eV/A | the test bounds the energy at 1e-6 eV/atom and the stress at 1e-7 eV/A^3 (a tenth of the accuracy gate's floors) and keeps the plan's 3e-6 eV/A for the forces. Recommend changing the criterion to that, and letting the tiled gate (S5-T04, the 3x rule against MACE float64) be the judge | forces differ by what another GEMM kernel changes in the periodic call: rounding. The energy of the two OMAT-0 models differs by 5 to 17 times that; the cause is not isolated (candidates: the `linear_up` GEMM over 843 instead of 135 rows; the order of equal-key edges in a row). It is about a seventh of LoupMD's own float32 energy error against MACE float64 on this cell (2.2e-6 eV/atom in the gate) | if the owner wants 1e-7: find the term (double-precision readout sums are the first thing to try) |
| D14 | S1-T07's per-atom energies need variants of the readout kernels, which `python/gen_kernels.py` and `python/gen_generic.py` generate; `python/` is held by D2 | energy-only is built for both paths (`loup_evaluate2`); `LOUP_EVAL_PER_ATOM` is refused by name. Recommend doing the per-atom variants right after the layout move, in `codegen/`, together with S6-T08 which is their only user | editing the generators now would collide with the developing session's open work in the same files | none |
| D15 | The plan says 19 model labels; FastMACE's table has 20 since 8 November (`omat0-small` was added) | `loupmd/models.py` carries all 20; "19 labels" in S2-T04, S2-T06 and S7 reads as "every label of the table" | the engine supports and gates `omat0-small` | none |
| D12 | S6-T06 and S6-T07 are done on `round6-hardening`; A2 concerns five labels | **accepted by the owner (15 Nov 2026):** the two tasks shrink to the gate under the release layout and the front-end arguments; S6-T10 measures all five labels | | |
| D17 | S2-T01 pins the export extra | exact versions the exporters were tested with: `torch==2.14.1`, `mace-torch==0.3.16`, `cuequivariance==0.12.0`, `cuequivariance-torch==0.12.0`; the runtime pins `cuda-toolkit[cublas,cudart]==13.0.3`, as torch 2.14.1+cu130 does | an export with another mace or torch version was never gated | loosen the pins in `pyproject.toml` once a second version passes the gates |
| D18 | The benchmark and gate structures are not in the package: `NbCl5_R2SCAN.cif` (standard suite, default gate set), the 216-molecule water box, the three COD cells of the `organic` set, the three R2SCAN cells of the `crystals` set | they are read from the folders of `$LOUPMD_BENCH_STRUCTURES`; `loupmd bench --quick` and the `families` suite for the Materials Project, MH-1 and POLAR models need no file (`SrTiO3` and rocksalt NaCl are generated) | redistribution is the owner's decision (§8) | copy the allowed files into `loupmd/bench/structures/`; nothing else changes |
| D19 | `loupmd bench`: `--compare fastmace` is not built, and `--ab` cannot take a library of another C interface version (the `round6-hardening` tip has version 0 and is refused, as S1-T02 requires) | compare against FastMACE by hand until its step timing has a public entry point; `python/bench_md.py` stays the A/B tool against builds without the version call | the comparison is not part of S2-T08's acceptance; refusing a foreign library is the ABI rule | build `--compare fastmace` on `fastmace.bench.step_time` when wanted |
| D20 | S2-T08's acceptance says `loupmd bench` reproduces `bench_md.py` within 1% on the same build. Measured: `SrTiO3` -0.5% and +0.1%; `NbCl5` -0.7% and +1.3% in two runs, while `bench_md.py`'s own `NbCl5` runs of this build spread from 1.783 to 1.839 ms (3%) on the same day | read the criterion as "within the run-to-run spread of `bench_md.py`" (1% for `SrTiO3`, 3% for `NbCl5`); the protocol is the same code path, call for call | 1% is below what the card repeats for the 1,944-atom cell | none |
| D21 | After the merge of `round6-hardening` `e21faa2`, seven `bench_md.py` A/B runs against that tip give head / tip 1.0030 for `NbCl5` (0.9946 to 1.0080) and 0.9997 for `SrTiO3` (0.9965 to 1.0044); before the merge Stage 1 tied its baseline at 0.9994 / 0.9986 | counted as a tie: the `NbCl5` mean is 1.4 standard errors from 1, the MD loop differs from the tip only by Stage 1's host-side counters, and single `NbCl5` runs of one build spread by 3% | §2 forbids a speed regression on the standard benchmarks, so the owner should see the number | a longer A/B on an idle machine, or `nsys` kernel lists of both builds, if 0.3% is to be chased |
| D22 | §4.4 said every family has two interaction layers except OMOL-0. The released files say: POLAR-1-S one, POLAR-1-L three, OMOL-0 extra large three, all others two (`loupmd/models.py`, read with `loupmd/_probe.py`) | §4.4 corrected; no consequence for the sweeps, which take any layer count, and POLAR is refused for tiled execution | the plan was wrong | none |
| D23 | A built wheel cannot run yet: `loupmd/__init__.py` loads the bundle cache from `python/loupmd.py` and the exporters live in `python/`, neither of which is in the wheel; `_lib/` is empty until S2-T02 | S2-T01 is checked with an editable install; S2-T03, S2-T11, S2-T12 and the tag `v1.0.0a2` wait for the layout move (D2) | a wheel needs the move | none |
| D24 | `IntegratorConfig` (S3-T01) had no `fixcm` | **settled by S4-T01 (`42cd77d`):** `IntegratorConfig.fixcm`, and the records' `fixcm="auto"` is on (off with fixed atoms), as in FastMACE | | |
| D25 | S3-T06 said the engine sees a schedule's temperature at chunk boundaries | built more exactly: the thermostat's target is a device scalar that the step advances, so every step of a ramp runs at its end-point target, as in FastMACE, and a changed temperature captures no graph. `LoupIntegrator` gained `temp_rate_k_per_step` (the layout of ABI v1 changed before its first release) | a target constant over 20 steps would differ from FastMACE's runs and each change would capture the step again | none |
| D26 | When a restart is written | `Simulation.run` writes one every `Output.restart_every` steps and when `save_restart()` is called; `lm.md` and `loupmd md` save one at the end of the run. What FastMACE's `fm.md` does at its end was not compared | a finished run should be resumable without the user thinking of it | drop the call at the end of `md` |
| D27 | The run file (`loupmd.run/1`, `loupmd init md`, `loupmd run`) and the `--schedule ramp:0:400:3,hold:400:1` syntax are LoupMD's own | keys are `lm.md`'s argument names plus `[output]` and `[[schedule]]`; they were not compared with FastMACE's `init` / `run` files | S3-T08 fixes behaviour, not the file's spelling | rename keys before the documentation stage if FastMACE's files should run unchanged |
| D28 | The gate's MACE float64 reference may not fit the GPU for a large structure | with `reference_device="auto"` an out-of-memory error moves the reference to the CPU without a time estimate (`cpu_float64_time_limit_s` only switches the CPU off at 0); with an explicit `"cuda"` the gate is unavailable. FastMACE's reduced-cell reference is not built | a gate that silently takes an hour is better than none, and `check="allow_unavailable"` exists; untested on a real large cell | build the time estimate and the reduced cell when a large-cell gate is needed |
| D29 | Output on a Windows-mounted directory under WSL | measured: rows and frames every 20 steps cost 1.3% on the Linux file system (5,000 atoms) and 5.5% on `/mnt/d` (1,944 atoms) | the documentation should say to write runs to the Linux file system | none |
| D30 | S4-T10 asks for an optimiser overhead under 2% of the force calls at 5,000 atoms, with the optimisers on the host | FIRE meets it (0.077 ms a step, 0.9%). LBFGS does not: its products with the history of up to 100 pairs cost 1.4 ms a step after 40 steps (17%), and grow with the history; they are in matrix form already (six BLAS calls a step) | the criterion assumed vector algebra that is cheap beside an 8 ms force call; for LBFGS at this size it is not, on the host | move LBFGS's history products to the device (as FastMACE's) when relaxations of cells this large matter; until then FIRE is the cheap optimiser for large cells |
| D31 | S4-T01's `fix_com` | **built (`42cd77d`)** as designed: Langevin's noise is drawn in a pass of its own and loses its mass-weighted mean (FastMACE's `mass_weighted` projection); NVE's total momentum is removed at the start of every run call (FastMACE removes it after every half-kick; the forces of a periodic cell sum to zero to rounding, so the two agree to rounding); the temperature counts 3N - 3. `fixcm="auto"` is on, so `lm.md` and `lm.Simulation` now keep the centre of mass at rest by default; measured cost at 5,000 atoms 1.0009 and 0.9995 of the step without it. `loup_run_langevin` and `bench_md.py` are unchanged (no `fix_com`). Not built from S4-T01: the general integrator slots, which the next integrators bring; ASE's own projection (`fixcm_projection="ase"`) | parity with FastMACE at no measurable cost | owner: say so if `auto` should stay off |
| D32 | S4-T13's drop-ins | built for what the engine has: `loupmd.ase.md.Langevin`, `VelocityVerlet`, `loupmd.ase.optimize.LBFGS`, `FIRE`. ASE's default `fixcm=True` is taken since `42cd77d` (with the mass-weighted projection, which differs from ASE's own for unequal masses); ASE's `trajectory`, `logfile` and `restart` arguments are not taken (`output=` instead); a cell relaxation takes `cell_filter=lm.optimize.FrechetCell()` instead of a filter object. The other eight classes are refused by name | nothing is replaced silently | accept ASE's `trajectory=` and filter objects if FastMACE's example needs them |
| D33 | Quench settings name FastMACE's optimisers (`gpu-LBFGS`, `ase-LBFGS`, `gpu-FIRE`) | all run LoupMD's host optimisers; `ase-BFGS` is refused | profiles and FastMACE's settings files keep working; the name no longer says where the arithmetic runs | rename in LoupMD's own profiles before the documentation stage |

---

## 2. Global constraints

Every task inherits these. They come from `AGENTS.md`, the status file and the owner's instructions.

- **Precision:** no reduced-precision arithmetic on the force path: no BF16, FP8 or 1xTF32, no BF16x9 emulation,
  and no FP16 except where the result is as accurate as float32's (owner, O8). The one such use is the hybrid
  3xTF32 GEMM, whose FP16 tensor-core instruction multiplies split parts that float16 holds exactly; the 3x gate
  is the test for it and for anything like it. MACE-POLAR-1 uses float32 SIMT GEMMs only.
- **Accuracy gate:** against MACE float64, `dE <= max(3 dE32, 0.01 meV/atom)`, `dF <= max(3 dF32, 1e-5 eV/A)`,
  `dS <= max(3 dS32, 1e-7 eV/A^3)`. Every stage exit runs it for every supported model set.
- **No speed regression:** a stage may not slow the standard benchmarks. Check with the A/B harness in both build
  orders on the RTX 5060 Ti; on the RTX 4000 Ada when the owner allows the workstations.
- **Build and run in WSL.** Never overwrite a `libloupmd.so` that a process has loaded; build to a side name and
  select it with `LOUP_LIB`. Never mix object files of different `CUDA_ARCH`.
- **One GPU job at a time** on the local card. Another session works on `round6-hardening`; check `nvidia-smi`
  before a GPU job and do not benchmark while it does.
- **Do not touch `.internal/`** unless asked. **Do not touch the workstations** unless the owner says so.
- **Keep `STATUS_AND_PROFILE.md` and `OPTIMISATION_LEDGER.md` current** for anything that changes speed.
- **Checkpoints are pickles:** a path or URL is loaded only with a matching SHA-256 or an explicit `trusted=True`.
  This guard is never weakened.
- **No bitwise criteria on GPU results across runs:** force and adjoint sums use atomics, so two runs of one build
  differ at rounding level. Acceptance compares against the measured run-to-run spread or a stated tolerance.
- **Fully periodic, one structure, one GPU** at 1.0 (as FastMACE). The domain abstraction of Stage 1 is built so
  that non-periodic systems and several GPUs can follow without another redesign.
- **Git:** run git from Git Bash on Windows inside this worktree. WSL's git (2.43) cannot read the worktree's
  `D:/` gitdir, and `git worktree add --relative-paths` must not be used: it sets a repository extension that
  breaks WSL git in the main checkout.

---

## 3. Where we start

Read at `0336751`; every reference below was re-checked on 15 November 2026 at `da42ec3`, the merge of
`round6-hardening` `11778cd` (38 commits) into `release/v1`. Line numbers are those of `da42ec3`.

**Engine.** `src/engine.cu` (handle, GEMM autotune, plan cache, `record_force_pass` at line 1253, `run_md_loop` at
1978, the C entry points from 1589; unchanged by the merge), `src/generic.cu` (`GenericModel::record`: forward over
the layers, long range, backward), `src/interaction0.cu` and `src/generated_kernels.cuh` (specialised path),
`src/polar.cu`, `src/neighbors.cu`, `src/integrator.cu` (NVE and Langevin halves, kinetic energy), `src/sgemm.cu`,
`src/model_loader.cu`, `src/bundle.cu`. Nine C functions in `include/loupmd.h`. Families (status §5): every label
of FastMACE's table. On the specialised kernels: OMAT-0 medium, MP-0b3 medium, MPA-0 medium, MatPES PBE and
r2SCAN (one density architecture). On the generic path: MP-0 medium and large, OMAT-0 small, OFF23 small / medium /
large, OFF23b medium, OFF24 medium, MH-1 (any head, one per bundle), POLAR-1 S / M / L, and OMOL-0 extra large
(three layers, 1024 hidden channels; total charge and spin multiplicity are export arguments, one bundle per
setting).

**Python.** `python/loupmd.py` (ctypes wrapper, `ensure_loup_bundle`, `LoupEngine`), `export.py` and the three
exporters, `bundle.py`, the two kernel generators, `ref_generic.py`, `test_gate.py`, `bench_md.py`,
`bench_family.py`. 38 test functions in six files (36 at `0336751`); the status file records 71 passed cases
at S59.

**What blocks a release.**

| Gap | Evidence |
| --- | --- |
| No `README`, licence, `pyproject.toml`; one library per GPU architecture, found by relative path | `Makefile`, `python/loupmd.py:54` |
| Exporters import FastMACE from hard-coded checkouts; about 4,400 lines of it are used at export | `export_omat0.py:20-36`, `export_generic.py:27-31` |
| Models, structures and FastMACE paths of two machines in scripts and tests | `bench_family.py:28`, `test_gate.py:111-124,264`, `tests/conftest.py:21-26`, `tests/test_gate_wrapper.py:16-17` |
| Product kernels compiled only for models found in one folder; others refused at load | `gen_generic.py:1237`, `generic.cu:1173` |
| Errors go to stderr; Python raises "loup_evaluate failed" | `engine.cu:1520`, `loupmd.py:257` |
| NVE and Langevin only; no output, restart, calculator, CLI | `include/loupmd.h` |
| MD graphs re-captured on every run call; the displacement maximum never decays | status §4.2, items S4 and S2 |
| No capacity mode: about 80 KB per atom for OMAT-0 medium by buffer arithmetic, so roughly 150-200k atoms on 16 GB (estimate; largest recorded run 40,000 atoms) | `engine.cu:1789-1809`, status §1 |

**Found at the merge of 15 November; answered by the owner the same day (§1.1 O7 to O9, §1.3) and worked into
§2, §4.3, §4.4 and the task rows.**

- S6-T06 is ahead of the plan: all six labels are supported and gated on `round6-hardening` (`7c8ca4e`). Four of
  them (MP-0b3, MPA-0, both MatPES) run the specialised kernels, not the generic path the task names. What is left
  for the task is the gate under the release layout.
- S6-T07 is ahead of the plan and built differently: OMOL-0 extra large runs (`4af8b06`), with charge and spin
  folded into the bundle at export. `loup_set_charge_spin` of §4.3 assumes a run-time setting and has no
  counterpart in the code.
- §4.4 said every supported family has two interactions and refused any other count for tiled execution. OMOL-0
  extra large has three. §4.4 now defines the sweeps per layer and refuses no model for its layer count (O9).
- Decision A2 now concerns five labels on the specialised kernels, not OMAT-0 medium alone.
- The risk row "models not yet tested on the generic path" (§10) and the open point on the six untested labels
  (§11) are settled by the same commits.
- New test switches in `include/loupmd.h`: `LOUP_POLAR_FOLD`, `LOUP_CHAIN_FOLD`, `LOUP_SPECIES_TABLES`.

**What FastMACE offers** (two read-only inventories of `D:\Margraf\FastMACE` on 14 Nov; only the rename plan and
the tiled documents were re-read directly): 19 model labels with hash-checked download; NVE, Langevin, CSVR,
Nose-Hoover chain, Berendsen; three NPT integrators (flagged experimental); temperature schedules; `FixAtoms`;
LBFGS and FIRE with a Frechet cell filter; swap, transmute and displace Monte Carlo; GPU D3(BJ); thermo CSV, ASE
trajectory, complete restart, `metadata.json` provenance; an ASE calculator and drop-in ASE integrators; the CLI
`doctor models md init run relax quench`; `check="gate"`; a tiled and wavefront capacity path that is driven only
by `tools/bench/tiled_*.py`. It has no user benchmark command, needs an exact torch pin, a user-chosen ops wheel
and gcc, and its first start takes about 40-60 s.

---

## 4. Target architecture

### 4.1 Layers

```
loupmd (Python, torch-free)         load / md / Simulation / Output / calc / CLI / bench / validate
        |  ctypes, C ABI v1
libloupmd.so (C++/CUDA)             domains + three sweeps, integrators, tile driver, neighbour lists, GEMM autotune
        ^  .loup bundle (weights, certified tables, kernel signature)
loupmd.export (optional extra)      torch + mace-torch on CPU: checkpoint -> bundle
```

Run time needs only the wheel, `numpy`, `ase` and the NVIDIA CUDA 13 runtime wheels. Export is needed once per
(model, element set, head) and can run on another machine.

### 4.2 Repository layout after Stage 2

```
pyproject.toml  README.md  LICENSE  NOTICE.md  CHANGELOG.md  PLAN.md  Makefile
include/loupmd.h                      C ABI v1
src/                                  engine; new: domain.hpp, sweeps in generic.cu / engine.cu, tiled.cu,
                                      integrators/*.cu, thermo.cu, dispersion/*.cu
codegen/gen_kernels.py gen_generic.py build-time kernel generators (moved from python/)
loupmd/
  __init__.py  _core.py  errors.py  config.py  cache.py  models.py
  forcefield.py  simulation.py  integrators.py  protocols.py  output.py  restart.py
  provenance.py  report.py  quick.py  capacity.py  mc.py  dispersion.py
  ase/ calculator.py md.py optimize.py
  optimize/ lbfgs.py fire.py cell.py
  bench/ protocol.py suites.py compare.py ab.py reference/*.json structures/
  validate/ gate.py references.py
  export/ __init__.py omat0.py generic.py polar.py bundle.py ref_generic.py _vendor/
  cli/ __init__.py doctor.py models.py md.py init.py run.py relax.py quench.py bench.py validate.py export.py cache.py
  _lib/ libloupmd.so cubins/          filled by the wheel build, not tracked
tests/ cpu/ gpu/ acceptance/ _support/
tools/ release/ ci/
docs/  examples/  plans/
```

The move of `python/` into `loupmd/` and `codegen/` conflicts with every open edit on `round6-hardening`. It is
one commit, made at a moment the optimisation session agrees to (S0-T03).

### 4.3 C ABI v1 (`include/loupmd.h`)

All calls return 0 or -1 as today. New and changed entry points:

```c
#define LOUP_ABI_VERSION 1
int         loup_abi_version(void);
const char* loup_version(void);                       /* "1.0.0+<library hash>" */
const char* loup_last_error(const LoupHandle* h);     /* h == NULL: the calling thread's last create error */

typedef struct {
    int abi, compute_capability, n_layers, has_long_range, specialised, n_species;
    double r_max;
    char gpu_name[64];
    char kernel_signature[160];
} LoupInfo;
int loup_get_info(const LoupHandle* h, LoupInfo* out);

typedef struct {                  /* size-versioned: first field is sizeof(LoupIntegrator) */
    unsigned size;
    int kind;                     /* LOUP_NVE, LOUP_LANGEVIN, LOUP_CSVR, LOUP_NHC, LOUP_BERENDSEN_NVT,
                                     LOUP_BERENDSEN_NPT, LOUP_BAOAB_NPT, LOUP_MTK_NPT */
    double dt_fs, temp_k;
    double temp_rate_k_per_step;  /* Langevin: step i of the call runs at temp_k + i * this; the target is a
                                     device scalar, so a schedule's calls reuse the captured step (S3-T06) */
    double friction_per_fs, tau_t_fs;
    double pressure_gpa, tau_p_fs, compressibility_per_gpa;
    int chain_length, cell_mode;  /* LOUP_CELL_HYDROSTATIC, LOUP_CELL_FULL; per-axis mask in cell_mask[3] */
    int cell_mask[3];
    int fix_com, thermo_every;    /* fix_com: the centre of mass at rest (S4-T01), three degrees of freedom fewer */
    unsigned long long seed;
} LoupIntegrator;
int loup_run(LoupHandle* h, const LoupIntegrator* it, int n_steps, LoupMDStats* out_stats);

typedef struct { long long step; double e_pot, e_kin, temp_k, pressure_gpa, volume_a3, conserved; } LoupThermoRow;
int loup_read_thermo(LoupHandle* h, LoupThermoRow* rows, int max_rows, int* n_rows);   /* drains the device buffer */

#define LOUP_EVAL_STRESS 1
#define LOUP_EVAL_ENERGY_ONLY 2
#define LOUP_EVAL_PER_ATOM 4
int loup_evaluate2(LoupHandle* h, const double* positions, const double* cell, int flags, double* out_energy_ev,
                   double* out_forces, double* out_stress, double* out_atom_energies);

int loup_export_state(LoupHandle* h, void* buffer, size_t capacity, size_t* written);  /* versioned blob */
int loup_import_state(LoupHandle* h, const void* buffer, size_t size);
int loup_set_cell(LoupHandle* h, const double* cell, int scale_positions);
int loup_set_fixed(LoupHandle* h, const unsigned char* fixed);            /* FixAtoms mask, n_atoms entries */
int loup_set_masses(LoupHandle* h, const double* masses_amu);
int loup_set_species(LoupHandle* h, const int* atomic_numbers);           /* same element set as the bundle */
int loup_swap_atoms(LoupHandle* h, int i, int j);                         /* Monte Carlo swap, no re-capture */
/* no loup_set_charge_spin: total charge and spin multiplicity are export arguments, one bundle per setting (O7) */

typedef struct { unsigned size; int mode;           /* LOUP_EXEC_AUTO, _WHOLE, _TILED, _WAVEFRONT */
                 int receivers; int state;          /* LOUP_STATE_DEVICE, LOUP_STATE_HOST */
                 double memory_fraction; } LoupExecution;
int loup_set_execution(LoupHandle* h, const LoupExecution* ex);
typedef struct { size_t whole_bytes, tiled_global_bytes, tile_workspace_bytes, free_bytes; int receivers; int mode; }
        LoupMemoryPlan;
int loup_plan_memory(LoupHandle* h, int n_atoms, double edges_per_atom, LoupMemoryPlan* out);
```

`loup_run_nve`, `loup_run_langevin`, `loup_evaluate` and the rest of today's header stay as wrappers.

### 4.4 The force domain and the per-layer sweeps (internal, `src/domain.hpp`)

```cpp
struct Domain {                     // one force-evaluation unit
    int n_local, n_ghost;           // locals are receivers and own their edges; ghosts only send
    bool periodic;                  // whole system: periodic with shifts, n_ghost == 0; tile: open, no shifts
    // device arrays over [n_local + n_ghost]: positions, species index; CSR edges over locals
};
struct LayerState {                 // one per layer boundary t = 0 .. L-2 of a model with L interaction layers;
                                    // rows over [n_local + n_ghost], width = input width of layer t + 1
    float* x;                       // output of layer t: written by its forward sweep for locals; gathered for
                                    // ghosts by the driver
    float* dx;                      // adjoint: added by the sweep of layer t + 1 for locals and ghosts; read by
                                    // layer t's backward sweep for locals
};
// 2L - 1 sweeps for L layers. With L = 2 they are the three sweeps of FastMACE's tiled path.
void sweep_forward(int t, Domain&, LayerState* states, bool keep_activations);         // t = 0 .. L-2
void sweep_last(Domain&, LayerState* states, ForceAccumulators&);                       // layer L-1: forward,
                                                                                        // readouts, backward
void sweep_backward(int t, Domain&, LayerState* states, ForceAccumulators&, bool replay_forward);  // t = L-2 .. 0
```

- The whole-system call is one periodic domain: the forward sweeps keep their activations, the backward sweeps do
  not replay, and all run back to back inside one captured graph, as `record` does today.
- A tile is an open domain. A backward sweep replays its layer's forward because the activations were not kept.
  Each sweep needs a halo of one cutoff: what reaches further travels through the stored layer states.
- Most families have two interaction layers. MACE-OMOL-0 extra large and MACE-POLAR-1-L have three (five sweeps,
  two layer states) and MACE-POLAR-1-S has one (one sweep), read from the released files (§1.3 D22). No model is
  refused for its layer count (owner, O9).
- MACE-POLAR-1 is refused for tiled execution: its k-space part is global (as in FastMACE).

### 4.5 Python API (same names and arguments as FastMACE wherever the feature exists)

```python
import loupmd as lm
ff  = lm.load("omat0-medium", download=True)                    # label, path or URL; sha256= or trusted=True for files
run = lm.md("NaCl.cif", model="mh1", head="matpes_r2scan", ensemble="NVT", temperature_K=300,
            steps=20000, repeat=(3, 3, 3), output="run/", seed=1)   # -> RunResult(summary, plan, gate, simulation, metadata, output)
sim = lm.Simulation(atoms, ff, integrator=lm.integrators.Langevin(temperature_K=300, timestep_fs=1.0),
                    output=lm.Output("run/", trajectory_every=20, thermo_every=20, restart_every=10_000), seed=1)
sim.run(steps=1000); sim.save_restart(); sim2 = lm.Simulation.resume("run/")
atoms.calc = lm.calc("omat0-medium")                             # ASE calculator: energy, forces, stress
```

- `lm.md` takes FastMACE's keyword names. Arguments that select FastMACE internals (`mode`, `allow_reference`,
  `fallback`, `runner`, `engine`, `neighbours`, `calibrate`) are accepted at their defaults and refused by name
  otherwise. New: `capacity="auto" | "whole" | "tiled" | "wavefront"`. `dispersion=` takes a shorthand such as `"d3bj"` or
  an `lm.Dispersion` record (Stage 6D).
- Units as FastMACE: A, eV, eV/A, K, amu, fs at the API, GPa where a name ends in `_GPa`; the ASE shims keep ASE's.
- Output files have FastMACE's names and formats: `thermo.<n>.csv`, `trajectory.<n>.traj`, `restart.npz`,
  `simulation.json`, `metadata.json` (schema `loupmd.metadata/1`).
- Errors derive from `LoupError(what, why, next_step)` and print as FastMACE's do; the CLI shows no traceback
  without `--debug`.

### 4.6 CLI

`loupmd [--version] [--debug] COMMAND`, exit codes 0 success, 1 LoupMD error, 2 usage error.

| Command | Purpose | Stage |
| --- | --- | --- |
| `doctor` | checks GPU, driver, CUDA runtime libraries, library load, architecture support, cache, model hashes; changes nothing | 2 |
| `models` | label table with heads, files, SHA-256, licences; `--download LABEL` | 2 |
| `export` | checkpoint to bundle for an element set | 2 |
| `cache` | `list`, `size`, `prune` for bundles, plans, references, models | 2 |
| `bench` | standard benchmark, comparison with stored references and optional baselines, JSON record | 2 |
| `validate` | the 3x accuracy rule on the user's model and structure | 2 |
| `md`, `init`, `run` | one-command MD, a commented TOML of every `md` argument, run a TOML or a shipped protocol, `--resume` | 3 |
| `relax`, `quench` | relax a structure; quench frames of a finished run | 4 |
| `dispersion` | `list` the built dispersion variants and their parameter sets | 6D |

### 4.7 Caches and environment

One root, `$LOUPMD_CACHE` (default `~/.cache/loupmd`; `LOUP_CACHE_DIR` stays as an alias), with `models/`,
`bundles/`, `plans/`, `gate_refs/`, `lib/`. It must lie outside `sys.prefix`. `$LOUPMD_MODELS` names a model
folder. `LOUP_LIB` selects another library build. The variant overrides of `AGENTS.md` stay as test switches.

---

## 5. Stages and tasks

Effort is in working weeks of one focused lane and is a rough estimate.

### Stage 0: ground rules (0.5 week)

| Task | What | Files | Acceptance |
| --- | --- | --- | --- |
| S0-T01 | Baseline record: build at `0336751`, full `pytest`, default gate, `bench_md.py` once, when the GPU is free | `plans/baseline-20261114.md` | the file holds the test count, the gate verdict and both ms/step numbers with the library hash |
| S0-T02 | Licence and notices (A6): `LICENSE` (MIT), `NOTICE.md` listing the hybrid cubins' origin and every file generated from FastMACE code | `LICENSE`, `NOTICE.md` | every tracked binary and generated file is named in `NOTICE.md` |
| S0-T03 | Agree the layout move with the session on `round6-hardening`: merge its tip into `release/v1`, then one commit that moves `python/*.py` to `loupmd/`, `loupmd/export/` and `codegen/` with `git mv` and fixes imports | as §4.2 | `pytest` passes from the new layout; `git log --follow` reaches each file's history |
| S0-T04 | Merge rule: `release/v1` merges `round6-hardening` at least at every stage exit; speed work stays on that branch | `PLAN.md` §9 | recorded in the stage exit notes |

**Progress (15 November 2026).** Step-level plan: `plans/stage-0-ground-rules.md` (`d78b8de`).
- S0-T01: **done**, `c0f70bf`: `plans/baseline-20261115.md`. Measured on the merged code (`bac354c`, §1.3 D10):
  71 tests passed, default gate PASS (worst force ratio 1.56, stress 1.71), `NbCl5` 1.820 and `SrTiO3` 8.137
  ms/step, library sha256 `df58a9bd2623a502...`.
- S0-T02: **done**, `bcb529f` (licences confirmed by the owner, O6): `LICENSE`, `cubins/LICENSE`, `NOTICE.md`,
  `tests/test_notice.py` (3 passed). The `sm_120` cubin's build record is with the developing session (D6).
- S0-T03: waits for the owner's moment. New Python code goes to `loupmd/`; `python/` is untouched.
- S0-T04: **done** as a rule; merges so far: `da42ec3` merged `round6-hardening` `11778cd` on 15 November 2026,
  before Stage 0; `11edd3c` merged `e21faa2` (S64 to S67) on 15 November 2026, during Stage 1.

### Stage 1: core API v1 (3 weeks)

The base of everything else. Each task keeps the standard benchmarks within noise and the gate passing.

| Task | What | Files | Interfaces produced | Acceptance |
| --- | --- | --- | --- | --- |
| S1-T01 | Error channel: `guarded()` stores the message in the handle; a thread-local holds create errors | `src/engine.cu`, `include/loupmd.h`, `loupmd/_core.py`, `loupmd/errors.py` | `loup_last_error`; `LoupError` raised with the engine's text | each of today's refusal tests asserts the message text, not only the failure; nothing is printed to stderr on a handled error |
| S1-T02 | ABI version, library version, `loup_get_info` | same | `loup_abi_version`, `loup_version`, `loup_get_info` | Python refuses a library whose ABI differs, naming both versions |
| S1-T03 | Keep captured MD graphs across run calls (review item S4): cache by integrator kind, stress flag and edge bucket | `src/engine.cu` (`run_md_loop`, `ensure_force_graph`) | none | 100 calls of 10 steps cost within 2% of one call of 1,000 steps on `SrTiO3` 5,000 atoms; trajectories agree with the old code within the spread of two runs of the old code |
| S1-T04 | `loup_run` with `LoupIntegrator`; NVE and Langevin move under it; the old entry points wrap it | `src/engine.cu`, `src/integrator.cu`, header | `loup_run`, `LoupIntegrator` | NVE and Langevin trajectories from the same seed differ from the old entry points by no more than two runs of the old entry point differ from each other; the kernel list of a step is unchanged (`nsys`) |
| S1-T05 | Thermo buffer on the device, written every `thermo_every` steps inside the graph | `src/thermo.cu`, engine, header | `loup_read_thermo`, `LoupThermoRow` | rows match `get_state` energies at the same steps to 1e-12 relative; no measurable step cost at `thermo_every = 20` |
| S1-T06 | State blob: positions, velocities, cell, step counter (the Philox offset), integrator state, displacement tracker | engine, header | `loup_export_state`, `loup_import_state` | export, destroy, create, import: every restored array equals the exported one bitwise; 100 further Langevin steps end within the spread of two uninterrupted runs (at most 1e-4 A); a blob of another atom count or version is refused |
| S1-T07 | `loup_evaluate2` with flags: energy only (no backward), per-atom energies (separate graph variant, off by default: the ledger's `frc-per-atom-energy` cost applies only when requested) | engine, `src/generic.cu`, header | `loup_evaluate2`, the three flags | energy-only equals the full call's energy to 1e-12 relative; per-atom energies sum to the total to 1e-9 relative; the default graph is unchanged (same kernel list in `nsys`) |
| S1-T08 | Split the generic pass into the per-layer sweeps of §4.4 (three for two layers, five for OMOL-0's three); the whole-system call composes them | `src/domain.hpp`, `src/generic.cu`, `src/generic.hpp` | `sweep_forward`, `sweep_last`, `sweep_backward`, `LayerState` | energy, forces and stress differ from the unsplit pass by no more than two calls of the unsplit pass differ from each other, for every generic family; the gate passes; A/B within noise |
| S1-T09 | Open domains with ghost atoms: neighbour build without periodic shifts where only locals receive; the sweeps accept `n_ghost > 0` | `src/neighbors.cu`, `src/generic.cu`, `src/domain.hpp` | `Domain` | a 135-atom periodic cell evaluated as an open domain whose ghosts are its explicit periodic images gives the periodic call's energy to 1e-7 eV/atom and forces to 3e-6 eV/A, for OMAT-0 small, OFF23 medium, MH-1 and (on a water cell) OMOL-0 extra large |
| S1-T10 | Memory plan from buffer sizes and `cudaMemGetInfo` | engine, header | `loup_plan_memory`, `LoupMemoryPlan` | the predicted whole-system bytes are within 10% of the measured peak for 5,000, 20,480 and 40,000 `SrTiO3` atoms |
| S1-T11 | Decaying displacement maximum (review item S2), so one fast step no longer keeps a run out of the pipelined loop | `src/engine.cu`, `src/integrator.cu`, `src/neighbors.cu` | none | a run with one fast first step pipelines again after two windows; water at 300 K reports `pipelined_steps > 0` at skin 1.0 A and 0 at skin 0.5 A, where its hydrogens are too fast for any decay (§1.3 D13), with `dangerous_builds == 0`; the existing pipeline tests pass |

**Exit:** all of the above, the default gate and `pytest`; tag `v1.0.0a1`.

**Progress (15 November 2026, 02:30).** Step-level plan: `plans/stage-1-core-api.md`. Everything below was
measured on the RTX 5060 Ti only; numbers and commands are in `STATUS_AND_PROFILE.md` §6 and the ledger
(`md-graph-reuse`, `md-disp-decay`, `md-thermo-buffer`, `rel-stage1-api`).

Stage 1 as a whole against the baseline (`plans/baseline-20261115.md`): `pytest` 205 passed at `7bafa23`
(baseline 71); default gate PASS at `a68cc04` (worst force ratio 1.55, stress 1.71); `bench_md.py` A/B 0.9994 /
1.0006 baseline first and 0.9986 / 0.9975 head first, a tie.

| Task | Commits | State |
| --- | --- | --- |
| S1-T01 | `22b5f57` | **done**: every refusal of `engine.cu` has its own text, tested; nothing on stderr |
| S1-T02 | `d1eba02` | **done**. `loup_version` gives `1.0.0a1+<hash>` (D8) |
| S1-T03 | `dd5268d` | **done**: 100 calls of 10 steps cost 1.011 and 1.006 of one call of 1,000 (baseline 1.113); criterion 2% |
| S1-T04 | `5f40a54` | **done**, except the `nsys` kernel list, which was not run (`run_md_loop` is unchanged; the old entry points share their graphs with `loup_run`, tested) |
| S1-T05 | `d0b1fc9` | **done**: rows equal the engine's energies; `thermo_every = 20` costs 1.0003 to 1.0041 of the step, inside the block spread |
| S1-T06 | `2cfec6d` | **done**: bitwise round trip, resumed Langevin run, seven refusals |
| S1-T07 | `903dabe` | **partly**: `loup_evaluate2` with the energy-only pass on both paths (equal to the full call's energy within 1e-11 relative, five model paths); per-atom energies wait for the layout move (D14) |
| S1-T08 | `11c5205`, `7bafa23`, `a68cc04` | **built**: the pass per layer, the replay schedule and the sweep selector, any layer count; equal to the ordinary pass for six model paths. Open: the family gates and an A/B on the generic path after the next merge |
| S1-T09 | `7bafa23`, `a68cc04`, `3477c38` | **built as a test driver**: a cell with its images as ghosts gives the periodic forces to 0.7e-6 to 2.1e-6 eV/A (criterion 3e-6) for five models, OMOL-0's three layers included; the energy is within 3.1e-7 eV/atom, above the criterion's 1e-7 for the two OMAT-0 models (D16). The tile workspace is Stage 5 |
| S1-T10 | `e9b55c5`, `2f56b11` | **done**: -5.1%, +1.4%, -0.1% against measured memory at 5,000, 20,480 and 40,000 atoms (criterion 10%). It scales the current system's buffers, so a small system must be set first |
| S1-T11 | `d4d5433`, `2f56b11` | **done** against the changed criterion (D13): one fast step is forgotten after two windows; water pipelines at skin 1.0 A, not at 0.5 A |

Also done ahead of its stage, because it needs no GPU: S2-T04 (`3bdbd39`), the model registry and `loupmd models`.

**After the merge of `round6-hardening` `e21faa2` (`11edd3c`, 15 November 2026, 04:30 to 05:10; RTX 5060 Ti
only).** Conflicts in `src/engine.cu` and `src/generic.cu` were resolved by keeping both sides: upstream's
hybrid drop-and-retry (S65) around this branch's `prepare_evaluate` / `evaluate2`, with its messages on the error
channel. On the merged code: `pytest` 212 passed; default gate PASS, 30 rows and both reorder checks, worst force
ratio 1.52, stress 1.68; `bench_md.py` against the tip's own build in seven A/B runs, head / tip 1.0030 for
`NbCl5` (0.9946 to 1.0080) and 0.9997 for `SrTiO3` (0.9965 to 1.0044), counted as a tie (§1.3 D21); the generic
path (S1-T08's open A/B) with `bench_family.py`: OMAT-0 small `SrTiO3` 0.9977, OMAT-0 small `NbCl5` 1.0066 (six runs
a side), MH-1 `SrTiO3` 1.0009, OFF23 medium `water216` 0.9944 over 18 blocks a side (1.0127 over six runs before
that; its blocks spread by 13 to 15% in both builds); 100 calls of 10 steps cost 1.0106 of one call of 1,000. Numbers
and commands: `STATUS_AND_PROFILE.md` §6.1, ledger `rel-merge-r6tip`.

Not done for the exit of Stage 1: the rest of S1-T07 (D14), the family gates of S1-T08 on the merged code (the
default gate ran; `--set organic` and `--set crystals` for the generic families did not), and the owner's word on
D13, D14, D16 and D21. The tag `v1.0.0a1` is not set. Two observations are open. The replay schedule of
S1-T08 on OFF23 medium failed once inside a full run of 450 tests (15 November, 06:05) and passed in 12 isolated
repeats, with and without the per-system conv variant timing; its message was not kept, so the size is unknown.
And:
`test_polar_energy_follows_a_small_cell_change_on_the_same_engine` failed once inside a full run (stress difference
3.97e-8 against its 3e-8 eV/A^3 bound) and passed in 12 isolated repeats and in the next full run.

### Stage 2: installable runtime, benchmark and validation (2.5 weeks)

| Task | What | Files | Acceptance |
| --- | --- | --- | --- |
| S2-T01 | `pyproject.toml` (setuptools, dynamic version from `loupmd/__init__.py`, entry point `loupmd = "loupmd.cli:main"`, extras `export`, `dev`, `docs`), library loaded from `loupmd/_lib/`, CUDA 13 runtime libraries found from the NVIDIA wheels. Pin the exact distribution names from the dependency list of `torch==2.14.1+cu130` | `pyproject.toml`, `loupmd/_core.py` | `pip install -e .` in a venv without torch, then `python -c "import loupmd; print(loupmd.runtime_info())"` prints GPU and library version |
| S2-T02 | Multi-architecture library: `make wheel-lib` compiles with `-gencode` for 8.0, 8.6, 8.9, 9.0, 10.0 and 12.0 into one `libloupmd.so`; `tools/release/build_wheel.sh` runs it in a manylinux_2_28 CUDA 13 image and tags the wheel `py3-none-manylinux_2_28_x86_64` | `Makefile`, `tools/release/build_wheel.sh` | one wheel loads and passes the fast GPU tier on the RTX 5060 Ti (12.0); on the RTX 4000 Ada (8.9) when allowed; `cuobjdump` lists all six targets. Untested targets are named as such in the README |
| S2-T03 | Vendor the FastMACE modules the exporters import (`kernels/cuda/sc_factored.py`, `sc_generated.py`, `rewrites/radial/`, `rewrites/generic/modules.py`, `rewrites/linear/modules.py`, `kernels/triton/compose_skip`) under `loupmd/export/_vendor/`, trimmed to what export needs, with the FastMACE commit in `_vendor/VERSION`; remove every `sys.path` insertion | `loupmd/export/` | for OMAT-0 medium (`SrTiO3`), OMAT-0 small, OFF23 medium, MH-1 and POLAR-1-S, the bundle exported with the vendored code is byte-identical to the one exported before; no tracked file names a path under `/home`, `/mnt/d` or `/mnt/homes` |
| S2-T04 | Model registry: the 19 labels with file, URL, SHA-256, heads and licence, ported from `fastmace/model/labels.py`; lookup order `checkpoint=`, `$LOUPMD_MODELS`, cache, then download only when asked; hash check before a file is kept; trust rule for paths and URLs | `loupmd/models.py`, `loupmd/cli/models.py` | CPU tests: a wrong hash is refused and the file removed; a path without hash or `trusted=True` raises `CheckpointTrustError`; `loupmd models` lists 19 labels |
| S2-T05 | Cache layout of §4.7, `ensure_bundle` on it with a file lock, `loupmd cache list / size / prune --older-than / --max-size` | `loupmd/cache.py`, `loupmd/cli/cache.py` | two processes asking for the same bundle export it once; `prune --max-size 1G` leaves at most 1 GiB and never removes a file in use by the caller |
| S2-T06 | Kernel coverage: the release build generates product kernels from all 19 labels' checkpoints and writes the signature list to `src/generated_generic_manifest.json`; an unknown signature raises `UnsupportedModel` naming the signature and `loupmd build-kernels` | `codegen/gen_generic.py`, `loupmd/export/`, `src/generic.cu`, `loupmd/cli/export.py` | CPU test: every label whose export path exists maps to a signature in the manifest; a synthetic model with another signature gets the documented message. `loupmd build-kernels MODEL` (needs `nvcc`) builds a library with that model's kernels into `$LOUPMD_CACHE/lib/` and the package prefers it |
| S2-T07 | `loupmd doctor`: GPU and driver, CUDA runtime libraries, library load and ABI, compute capability against the built targets, hybrid cubin for the card, cache root, model hashes, export extra present or not | `loupmd/cli/doctor.py` | each line is `[ ok ]`, `[warn]` or `[FAIL]` with a fix; exit 1 on any `[FAIL]`; CPU tests fake each failure |
| S2-T08 | `loupmd bench` (see §8): shipped structures, standard and quick suites, reference comparison, `--compare mace,fastmace`, `--ab`, JSON record | `loupmd/bench/`, `loupmd/cli/bench.py` | `loupmd bench --model omat0-medium` reproduces `bench_md.py`'s two numbers within 1% on the same build; the JSON validates against `loupmd.bench/1`; a busy GPU is reported and refused unless `--force` |
| S2-T09 | `loupmd validate`: `test_gate.py` as a library and command, references cached under `gate_refs/`, sets `oxide`, `organic`, `crystals` and any user structure | `loupmd/validate/`, `loupmd/cli/validate.py` | same verdicts and ratios as `test_gate.py` for the default set; `--structure my.cif` gates two rattled copies; exit 1 on failure; without the export extra it raises `GateUnavailable` naming the extra |
| S2-T10 | ASE calculator `lm.calc`: energy, free energy, forces, Voigt stress; reuses the engine while the species and atom count stay; rebuilds on a change | `loupmd/ase/calculator.py` | ASE `VelocityVerlet` with the calculator conserves energy within 0.05 meV/atom/ps on `SrTiO3` 3x3x3; energies and forces equal `evaluate` |
| S2-T11 | Tests re-homed: `tests/cpu` (no GPU; a fake core with a harmonic potential implements the `_core` interface), `tests/gpu` in tiers fast / full / acceptance by marker; models found only through `LOUPMD_MODELS`; structures from the package | `tests/` | `pytest tests/cpu` passes in a venv without torch and without a GPU; the fast tier runs under 10 minutes |
| S2-T12 | Clean-install proof: fresh venv without torch, install the wheel, `loupmd doctor`, `loupmd bench --quick` on a pre-exported bundle, with `nvcc`, `g++` and torch shadowed; a second venv with the export extra exports and runs | `tools/release/clean_install.sh` | the script exits 0 and prints both runs' ms/step |

**Exit:** a user with a CUDA 13 driver can install, run `doctor`, `bench` and `validate`, and use the ASE calculator.
Tag `v1.0.0a2`.

**Progress (15 November 2026, 05:15).** Tasks that need neither the layout move nor a container were done ahead of
Stage 1's exit; GPU checks on the RTX 5060 Ti only, on the merged engine build (`11edd3c`).

| Task | Commits | State |
| --- | --- | --- |
| S2-T01 | `1cf9cfc` | **done except the network install**: `pyproject.toml`, `loupmd.runtime_info()`. Checked offline: the wheel's metadata; an editable install in a venv without torch prints library version and GPU. Not checked: installing the dependencies from PyPI; loading the CUDA runtime from the NVIDIA wheels on a machine without a toolkit (D17, D23) |
| S2-T02 | | not started (needs a manylinux CUDA 13 image) |
| S2-T03 | | waits for the layout move (D2) |
| S2-T04 | `3bdbd39`, `4358fcc` | **done**: 20 labels (D15); since `4358fcc` each release's elements, cutoff and layer count |
| S2-T05 | `403f65a` | **done**: `loupmd cache list / size / prune`, the bundle cache with a file lock (CPU tests: two processes export once; prune keeps files in use) |
| S2-T06 | | not started (generators are in `python/`, D2) |
| S2-T07 | `dd8eb2f` | **done**: `loupmd doctor`, every failure faked in CPU tests, run once on the GPU |
| S2-T08 | `93f296e` | **done except `--compare fastmace` and the shipped structures** (D18, D19): suites `standard`, `quick`, `families`; `capacity` is refused until Stage 5. On the GPU: 1.795 / 7.997 and 1.830 / 8.040 ms against `bench_md.py`'s 1.807 / 8.034 on the same build (D20); records valid; `--compare mace` 395.8 ms/step against 1.767 on `SrTiO3` 6x6x6; `--ab` a b b a runs. The reference table holds these RTX 5060 Ti numbers |
| S2-T09 | `9521e71` | **done for the default set**: `loupmd validate` PASSED with worst ratios 0.66 / 1.53 / 1.67 (energy / force / stress) against `python/test_gate.py`'s PASS with 1.52 / 1.68 on the same build, reading the same cached references; a user structure (`SrTiO3` 3x3x3, two rattled copies) PASSED at 1.98 / 1.43. Not run: `--set organic`, `--set crystals`, a model on the CPU reference path |
| S2-T10 | `f99f1d1`, `4358fcc` | **done**: `lm.calc`. ASE `VelocityVerlet` on `SrTiO3` 3x3x3 drifts -0.0006 meV/atom/ps over 1 ps (criterion 0.05); energy, forces and stress equal `LoupEngine.evaluate`; a FIRE relaxation converges |
| S2-T11, S2-T12 | | wait for the layout move (D2, D23) |

### Stage 3: FastMACE-shaped front end (3.5 weeks)

| Task | What | Files | Acceptance |
| --- | --- | --- | --- |
| S3-T01 | Errors and config records: frozen dataclasses for integrator, output, gate, capacity; `LoupError` family (`InputError`, `UnsupportedModel`, `ModelElementError`, `CheckpointTrustError`, `PlanRefused`, `GateFailed`, `GateUnavailable`, `RestartMismatch`, `NonFiniteError`, `CellDivergedError`, `CapacityError`, `BackendError`, `CacheLocationError`); unknown keys refused by name | `loupmd/errors.py`, `loupmd/config.py` | CPU tests for every class's printed form and for refused keys |
| S3-T02 | `lm.load` and `ForceField`: label or file to bundle for the structure's element set; `evaluate(atoms, stress=False)`; head rule (no default head for multi-head models, as FastMACE) | `loupmd/forcefield.py` | a missing element raises `ModelElementError` before any export; a multi-head model without `head=` is refused naming its heads |
| S3-T03 | `Output`: chunked thermo CSV and ASE trajectory, asynchronous writer, readers `read_frames`, `read_thermo` | `loupmd/output.py` | a 1,000-step run writes the same row and frame counts as FastMACE's defaults (every 20 steps); frames equal `get_state` at those steps |
| S3-T04 | Restart: `restart.npz` (the state blob plus the Python-side schedule position and counters) and `simulation.json`; `Simulation.resume` | `loupmd/restart.py` | resume after 500 of 1,000 NVE steps on 135 atoms ends within the spread of two uninterrupted runs (at most 1e-3 A), and the restored state equals the saved one bitwise; a restart of another model hash or atom count raises `RestartMismatch` |
| S3-T05 | `Simulation.run`: chunks the engine run at output cadence, drains the thermo buffer, feeds sinks, checks for non-finite values | `loupmd/simulation.py` | with output every 20 steps the run is within 3% of the same run without output on 5,000 atoms |
| S3-T06 | Temperature schedules `protocols.ramp / hold` and the profiles `slow`, `fast`, `ultrafast`, `anneal`; pressure schedules refused | `loupmd/protocols.py` | the temperature seen by the engine follows the schedule at chunk boundaries; a ramp's mean kinetic temperature tracks the target within 5% on 5,000 atoms |
| S3-T07 | `lm.md`, `RunResult`, the one-screen report (header, physics, plan, gate, setup, speed, licence, warnings) with ms/step, ns/day and atom-steps/s | `loupmd/quick.py`, `loupmd/report.py` | every argument error is raised before the model is loaded (CPU test with a model path that does not exist) |
| S3-T08 | CLI `md`, `init`, `run` (TOML and the shipped protocols), `--resume` | `loupmd/cli/` | `loupmd init md ... > run.toml && loupmd run run.toml` equals the same `loupmd md` call; unknown TOML keys are refused by name |
| S3-T09 | Provenance `metadata.json`: code version and git commit, library hash, bundle key and model SHA-256, label and licence, plan (execution mode, kernel variants, GEMM plan file), precision, software versions, GPU name, compute capability and the NVIDIA driver number from NVML, seed, gate record, counters, timing | `loupmd/provenance.py` | schema test; a run's metadata reproduces the run when fed back through `loupmd run` |
| S3-T10 | `check="gate"` inside a run: two rattled copies (0.05 A), plus two strained copies (2%) for NPT, through `loupmd.validate`; statuses `passed`, `failed`, `unavailable`, `skipped` | `loupmd/simulation.py`, `loupmd/validate/` | a deliberately corrupted bundle table fails the gate and the run does not start |

**Exit:** `lm.md`, `Simulation`, output, resume and the CLI run NVE and Langevin end to end. Tag `v1.0.0a3`.

**Progress (15 November 2026, 06:10).** Every task of the stage is built for what the engine has today (NVE and
Langevin). GPU checks on the RTX 5060 Ti only; numbers and commands in `STATUS_AND_PROFILE.md` §6.2. Full suite on
the last engine build: 450 tests, 449 passed and 1 failed (`tests/test_sweeps.py`, the replay schedule on OFF23 medium; it then passed in 12 isolated repeats and had passed in the four full runs since the merge; its message was not kept).

| Task | Commits | State |
| --- | --- | --- |
| S3-T01 | `43aff6c`, `459e07a` | **done**: the error family and the records; `IntegratorConfig` without `fixcm` (D24) |
| S3-T02 | `4358fcc` | **done**: `lm.load`, `ForceField.evaluate`; a missing element and a multi-head model without `head=` are refused before any export (CPU tests) |
| S3-T03 | `824b6cd`, `0147e7f` | **done**: 1,000 NVE steps on 135 atoms write 51 rows and 51 frames (step 0 and every 20); the last frame equals the engine's state bitwise; a row's and its frame's energy agree to 1e-12 relative |
| S3-T04 | `0147e7f` | **done**: resumed after 500 of 1,000 NVE steps, the run ends 7.0e-7 A from the uninterrupted one (two uninterrupted runs differ by 7.5e-7 A; criterion 1e-3 A); the restored state equals the saved one bitwise; another model's force field and a state of another atom count are refused |
| S3-T05 | `0147e7f` | **done**: rows and frames every 20 steps cost 1.0132 of the run without output on 5,000 atoms (criterion 1.03), on the Linux file system (D29) |
| S3-T06 | `a436947`, `e11370e` | **done**, per step instead of per chunk (D25): 200 to 400 K at 100 K/ps on 5,000 atoms, mean kinetic temperature 296.6 K for a mean target of 301.0 K (0.9855; criterion 5%); the ramp's steps cost 7.985 ms against 7.839 on the hold before it. Profiles `slow`, `fast`, `ultrafast`, `anneal`; pressure schedules and NPT are refused |
| S3-T07 | `cb8b020` | **done**: `lm.md`, `RunResult`, the report. 21 argument errors are raised before the model is looked for (CPU); a real run of 320 atoms with its report (GPU) |
| S3-T08 | `cb8b020` | **done**: `loupmd md`, `loupmd init md`, `loupmd run`, `--resume`. `init` then `run` equals `md` and unknown keys are refused by name (CPU, fake engine); both commands ran once on the GPU. The file's spelling is LoupMD's own (D27) |
| S3-T09 | `cb8b020` | **done**: `metadata.json` validates against `loupmd.metadata/1`, and `loupmd run metadata.json` repeats the run (CPU). Deviations from the task's list: the driver number comes from nvidia-smi, not NVML, and the GEMM plan file is not recorded (the engine does not report which file it read) |
| S3-T10 | `e11370e`, `cb8b020`, `1b06826` | **done for NVE and NVT**: `check="gate"` passes on `SrTiO3` 3x3x3 (worst ratios 0.91 / 1.80 / 1.44); a bundle with one table scaled by 2% fails with ratios 6,172 / 6,637 / 19,809 and no step runs. Strained copies wait for NPT; the CPU fallback of the reference is untested on a real large cell (D28) |

Not done for the exit of Stage 3: the tag `v1.0.0a3` (not set; Stage 1's and 2's exits are open, see above),
and a comparison of the front end's spelling with FastMACE's on a real FastMACE run (D26, D27).

### Stage 4: dynamics, relaxation and ASE drop-ins (4.5 weeks)

FastMACE's integrators are ports of ASE 3.27 classes and are the reference implementations and test oracles
(`fastmace/integrators/*.py`, `fastmace/reference/`). LoupMD's versions run on the device inside the captured step.

| Task | What | Files | Acceptance |
| --- | --- | --- | --- |
| S4-T01 | Integrator slots in the engine: pre-force and post-force kernels per kind, a device struct for scalar state, degrees of freedom with `fix_com` and fixed atoms | `src/integrators/common.cuh`, `src/engine.cu` | NVE and Langevin unchanged: same kernel list per step, trajectories within the run-to-run spread |
| S4-T02 | `FixAtoms`: fixed mask applied in every integrator and optimiser; other ASE constraints refused | `src/integrators/`, `loupmd/simulation.py` | fixed atoms do not move over 1,000 Langevin steps; the temperature uses the reduced degrees of freedom |
| S4-T03 | CSVR (Bussi) | `src/integrators/csvr.cu`, `loupmd/integrators.py` | the kinetic-energy distribution over 20,000 steps on 135 atoms matches the canonical one (Kolmogorov-Smirnov p > 0.01); the conserved quantity drifts under 0.05 meV/atom/ps |
| S4-T04 | Nose-Hoover chain | `src/integrators/nose_hoover.cu` | positions agree with ASE's `NoseHooverChainNVT` driven by `lm.calc` to 1e-4 A after 200 steps from the same state |
| S4-T05 | Berendsen NVT | `src/integrators/berendsen.cu` | as S4-T04 against ASE's `NVTBerendsen` |
| S4-T06 | Variable cell inside the graph (`loup_set_cell`): integer shifts and the cell on the device, Cartesian shifts computed where edges are compacted, volume and stress available every step; cell limits raise `CellDivergedError` outside [V0/20, 20 V0] | `src/neighbors.cu`, `src/engine.cu` | a fixed-cell run through the variable-cell graph gives forces within 1e-6 eV/A of the fixed graph; its extra step cost is recorded in the ledger |
| S4-T07 | Berendsen NPT with the per-axis mask | `src/integrators/berendsen.cu` | volume agrees with ASE's `NPTBerendsen` to 1e-6 relative after 200 steps |
| S4-T08 | Langevin BAOAB NPT | `src/integrators/baoab.cu` | mean pressure within its standard error of the target over 20,000 steps on `SrTiO3` 5x5x5; agreement with FastMACE's independent BAOAB reference for a noise-free step |
| S4-T09 | MTK NPT, hydrostatic and full cell | `src/integrators/mtk.cu` | the conserved quantity drifts under 0.05 meV/atom/ps; volume agrees with ASE's `MTKNPT` to 1e-5 relative after 200 steps. All NPT runs print "experimental", as FastMACE's do |
| S4-T10 | LBFGS and FIRE on the host (NumPy), ported from `fastmace/optimize/` | `loupmd/optimize/lbfgs.py`, `fire.py` | same step count as ASE's optimisers within 2 steps on rattled `SrTiO3` 3x3x3 to `fmax = 0.05`; optimiser overhead under 2% of the force calls at 5,000 atoms |
| S4-T11 | Frechet cell filter with scalar pressure, hydrostatic, constant-volume and mask options; refused for POLAR-1 and with fixed atoms (as FastMACE) | `loupmd/optimize/cell.py` | relaxed lattice constant of strained `SrTiO3` agrees with ASE's `FrechetCellFilter` to 1e-4 A |
| S4-T12 | `Simulation.relax`, CLI `relax` and `quench`; relax writes extxyz | `loupmd/simulation.py`, `loupmd/cli/` | `loupmd relax` output equals the Python call's |
| S4-T13 | `loupmd.ase` drop-ins: `Langevin`, `LangevinBAOAB`, `VelocityVerlet`, `Bussi`, `NoseHooverChainNVT`, `IsotropicMTKNPT`, `MTKNPT`, `NVTBerendsen`, `NPTBerendsen`, `Inhomogeneous_NPTBerendsen`, `LBFGS`, `FIRE`, with ASE's arguments and units plus `model=`, `head=`, `seed=`, `output=`, `check=`; `dyn.attach` supported with a host sync per call | `loupmd/ase/md.py`, `optimize.py` | FastMACE's ASE-port example runs after changing the import line |

**Exit:** every ensemble and optimiser of FastMACE runs natively. Tag `v1.0.0a4`.

**Progress (15 November 2026, 06:50).** The host side of the stage and fixed atoms are built; the integrators that
need new engine kernels are not. RTX 5060 Ti only; numbers in `STATUS_AND_PROFILE.md` §6.3.

| Task | Commits | State |
| --- | --- | --- |
| S4-T01 | `42cd77d` | **partly**: `fix_com` for Langevin and NVE with 3N - 3 degrees of freedom (D31): the total momentum stays at 1.6e-15 of its thermal value over 500 Langevin steps (1.4 without). The thermostat's target and its noise coefficients are a device struct the step advances. Not built: slots for further integrator kinds |
| S4-T02 | `06230cf` | **done**: `loup_set_fixed_atoms`, `FixAtoms` in `Simulation` and the optimisers, other constraints refused by name. 27 fixed atoms keep their positions bit for bit over 1,000 Langevin steps; the temperature counts 3 (N - n_fixed) |
| S4-T03 to S4-T09 | | **not started**: CSVR, Nose-Hoover chain, Berendsen, the variable cell in the graph, the three NPT integrators |

Next in line, S4-T03 (CSVR), as it will be built (design read off ASE 3.27's `Bussi` and the engine as it stands):
each step scales the velocities by alpha and then does the velocity-Verlet step. In the captured step that is a
kinetic-energy reduction, one thread that draws R1 and a Gamma((Nf - 1) / 2) variate (Marsaglia-Tsang on Philox,
offset by the device step counter), forms alpha from equation A7 with the target read from the device struct of
the Langevin target (so schedules work), and adds (alpha^2 - 1) K to the transferred energy; and a first half-kick
that multiplies the velocities by alpha. Three small nodes more than NVE, in CSVR runs only. The thermo row's
conserved quantity subtracts the transferred energy. The state block does not hold the transferred energy, so the
conserved quantity starts anew after a resume (the block's format would have to change to keep it). A structure
at rest is refused, as ASE's class does. Acceptance as in the table: a Kolmogorov-Smirnov test of the kinetic
energy over 20,000 steps on 135 atoms and the conserved quantity's drift.
| S4-T10 | `4f20442`, `8c709b9` | **done**, with one criterion missed: LBFGS and FIRE take ASE's step counts on rattled `SrTiO3` 3x3x3 (15 = 15 and 44 = 44, positions within 3e-7 A of ASE's with `lm.calc`). FIRE costs 0.9% of the force calls at 5,000 atoms, LBFGS 17% (D30) |
| S4-T11 | `4f20442` | **done**: the Frechet cell filter with pressure, hydrostatic, constant-volume and mask options, refused with fixed atoms and for POLAR-1. Strained `SrTiO3` 3x3x3: 157 steps as ASE's `FrechetCellFilter`, lattice constants within 1.0e-5 A (criterion 1e-4) |
| S4-T12 | `06230cf`, `4f20442` | **done**: `Simulation.relax`, `loupmd relax` (writes extended XYZ equal to the Python call's), `loupmd quench` with FastMACE's selection and files (CPU tests on the fake engine). Both commands ran once on the GPU: `loupmd relax` on rattled `SrTiO3` 3x3x3 converged in 25 steps; a 1,000-step ramp 0 to 600 to 300 K and `loupmd quench` of five picked frames, all to -7.98953 eV/atom |
| S4-T13 | `4f20442` | **partly**: four of the twelve classes (D32) |

### Stage 5: tiled and wavefront capacity mode (6 weeks)

Design sources: `FastMACE/docs/design/tiled-backend.md` and `tiled-wavefront.md`; results in
`FastMACE/docs/validation/tiled/`. FastMACE's records with OMAT-0 medium: 149M `SrTiO3` atoms on a 20 GB RTX 4000
Ada, 397.5M on an A100-80; 6.42 us per atom tiled and 7.74 with the wavefront at 975,560 atoms on the RTX 5060 Ti.
LoupMD does not port `fastmace/model/tiled.py`; it builds the same schedule on the sweeps of Stage 1.

Estimate, not a measurement: the extra work is the layer-0 replay (about 8% of the step), the halo gathers and
per-tile launches, so about 1.15-1.2 times the whole-system cost per atom, near 2.1 us per atom on the RTX 5060 Ti
with the generic path.

| Task | What | Files | Acceptance |
| --- | --- | --- | --- |
| S5-T01 | Tile planner on the host: bin atoms by fractional coordinates on a grid chosen from the cell lengths and the receiver cap, split blocks above the receiver or edge cap, list each tile's locals and its ghost candidates within cutoff plus skin with their image shifts (rules of `plan_tiles`, `tiled.py:895`) | `src/tiled.cu`, `src/tiled.hpp` | CPU test on 40,000 atoms: every atom is local in exactly one tile; every neighbour within the cutoff of a local is local or ghost in that tile; planning 1M atoms takes under 0.5 s |
| S5-T02 | Per-tile neighbour list on the device through the open-domain build of S1-T09; unused ghost candidates dropped | `src/neighbors.cu`, `src/tiled.cu` | the union of all tiles' edges equals the whole-system list's edge set on 40,000 atoms |
| S5-T03 | Tile driver, plain per-layer sweeps (three for two layers, five for three), state on the device: each forward sweep writes its layer's `x` rows; the next layer's sweep gathers them for locals and ghosts and scatter-adds `dx`; each backward sweep replays and finishes its layer; forces and stress accumulate through index lists; plain stream launches, GEMM plans reused across tiles by size bucket | `src/tiled.cu`, engine | energy within 1e-7 eV/atom and forces within 1e-5 eV/A of the whole-system call, and the 3x gate passes, on 40,000 `SrTiO3` atoms for OMAT-0 medium (generic path), OMAT-0 small, OFF23 medium and MH-1; OMOL-0 extra large (five sweeps) at a size its 1024 channels allow; POLAR-1 is refused |
| S5-T04 | Gate in tiled mode for every non-POLAR model set (`loupmd validate --capacity tiled`) | `loupmd/validate/` | all sets pass |
| S5-T05 | Receiver cap from the memory plan: the first call measures the workspace per receiver, later calls take the largest cap of 4,096 to 65,536 that fits with 10% headroom (FastMACE's budget rule); `loup_set_execution`; `capacity="auto"` chooses tiled when the whole-system plan does not fit | `src/tiled.cu`, `loupmd/capacity.py` | at 320,000 atoms on 16 GB the call runs without a user setting and reports its mode and cap in the plan; the device peak stays under 90% of free memory |
| S5-T06 | MD in tiled mode: integrators on the whole-system arrays, tile lists kept across steps until an atom moves skin/2, then re-planned | `src/tiled.cu`, engine | NVE on 320,000 atoms for 1,000 steps drifts under 0.05 meV/atom/ps; time per step is reported next to the whole-system per-atom figure |
| S5-T07 | State on the host (pinned memory) with `dx` aliased onto `x` rows, per FastMACE's `state="host"` and `alias_state` | `src/tiled.cu` | same forces as the device state to rounding; device memory per atom outside the tile workspace under 150 B at 5M atoms |
| S5-T08 | Wavefront schedule: tiles ordered in layers along the slowest grid axis; sweep 1 one layer ahead of sweep 2, sweep 3 one layer behind; `x` and `dx` live in a row pool for about three layers (schedule of `wavefront_schedule`, `tiled.py:953`); on by `auto` from 12 layers and 2^20 atoms | `src/tiled.cu` | CPU test of the schedule's invariants on a periodic ring; forces equal the plain tiled call's to rounding and energy to 1e-9 relative; the row pool holds under 25% of the atoms at 20 layers |
| S5-T09 | Large boxes: tile-local coordinate frame (float64 global positions, float32 offsets inside a tile) and 64-bit index audit above 2^31 edges | `src/tiled.cu`, `src/neighbors.cu` | a 1,000 A box gives the same forces as the same crystal in a 100 A box to rounding; an edge count above 2^31 is handled or refused with a message, never wrapped |
| S5-T10 | Speed at one million atoms: screen the receiver cap, measure launch overhead per tile, overlap host copies with the next tile | `STATUS_AND_PROFILE.md`, ledger | recorded us per atom at about 1M `SrTiO3` atoms on the RTX 5060 Ti, tiled and wavefront, next to the whole-system figure and FastMACE's 6.42 / 7.74 |
| S5-T11 | Capacity records with the owner's go for each machine: largest `SrTiO3` cell on the RTX 5060 Ti; on the Ada and on an A100 if allowed | `docs/validation/capacity.md` | each record names atoms, edges per atom, energy per atom, largest force, device and host peaks, time per call |
| S5-T12 | Front end: `capacity=` in `lm.md` and `Simulation`, the plan line in the report, documentation page | `loupmd/quick.py`, `docs/guide/large-cells.md` | `loupmd md --capacity auto` on a cell that does not fit whole runs tiled and says so |

**Exit:** tiled and wavefront runs pass the gate and run MD without user tuning. Tag `v1.0.0b1`.

### Stage 6: remaining parity (3.5 weeks without dispersion)

| Task | What | Files | Acceptance |
| --- | --- | --- | --- |
| S6-T01 | Events: `set_positions`, `set_momenta`, `set_masses`, `set_species`, `set_charges`, `set_friction` as transactions on `Simulation` | `loupmd/simulation.py`, engine (`loup_set_masses`, `loup_set_species`) | a rejected event leaves the previous state usable; `set_species` within the bundle's element set needs no re-export |
| S6-T02 | Monte Carlo swap: energy-only evaluations, acceptance on the host, `loup_swap_atoms` exchanges two atoms' slots without a re-capture | `loupmd/mc.py`, engine | acceptance ratio of an ideal binary mixture matches the analytic value within its standard error; move energies agree with the full call's to 0.01 k_B T |
| S6-T03 | Transmute (semi-grand canonical) through `loup_set_species` | `loupmd/mc.py` | composition under a chemical-potential difference of zero is symmetric within its standard error; the re-setup cost per accepted move is recorded |
| S6-T04 | Displace moves and hybrid MD/MC through `run(mc=...)` | `loupmd/mc.py`, `loupmd/simulation.py` | detailed-balance test on a harmonic fake core (CPU); the MC counters appear in `metadata.json` and survive a resume |
| S6-T05 | Dispersion: replaced by Stage 6D below (research first, then the build) | | |
| S6-T06 | Model coverage: export and gate `mp0-medium`, `mp0-large`, `mp0b3-medium`, `mpa0-medium`, `matpes-pbe`, `matpes-r2scan` on the generic path | `loupmd/export/generic.py`, codegen | each label passes its gate set or is listed in the README as unsupported with the reason |
| S6-T07 | OMOL-0 XL: total charge and spin multiplicity as arguments of `lm.load` and `ensure_bundle`, folded at export (O7; the engine side is done on `round6-hardening`, `4af8b06`) | `loupmd/export/generic.py`, `loupmd/forcefield.py` | passes the `organic` gate set for neutral singlets and one charged case under the release layout |
| S6-T08 | Per-atom energies in the calculator for atom-resolved models without dispersion | `loupmd/ase/calculator.py` | `get_potential_energies().sum()` equals the total to 1e-9 relative |
| S6-T09 | The refusals of decision A3 in code and docs: `precision="float64"`, `allow_reference=True`, `Workload(reproducible=True)` | `loupmd/quick.py`, docs | each raises `PlanRefused` with the alternative named |
| S6-T10 | Checkpoint on A2: measure generic against specialised OMAT-0 medium on both GPUs and put the keep-or-retire question to the owner with the numbers | ledger | a recorded decision |

#### Stage 6D: dispersion (about 11.5 weeks; research done and approved on 14 November 2026)

Owner decision O3: LoupMD should offer the newer dispersion implementations and let the user choose the variant,
and research on what to build comes first. Starting points, read on 14 November 2026:

- **FastMACE's `fm.D3`** (`fastmace/longrange/d3.py`): Becke-Johnson damping only; the parameter set follows the
  model's training functional or is given as explicit numbers; 40 bohr pair and coordination-number cutoffs; no
  three-body term; no cutoff smoothing. The shorthand `"d3bj"` expands from the label table and is refused when the
  functional already contains dispersion or is unknown.
- **FourierD3** (`github.com/vicvaleeva/FourierD3`, MIT; arXiv:2607.15103): the pairwise D3 sum in reciprocal space
  on a particle mesh, O(N log N), with no real-space cutoff on the dispersion sum; zero and Becke-Johnson damping; a
  coordination number that vanishes at its cutoff; no three-body term. The paper was read in full for the note:
  the code covers Becke-Johnson damping only, and the mesh carries one channel per element and rank.
- **`CheukHinHoJerry/torch-dftd`** (a fork of `pfnet-research/torch-dftd`): adds an experimental skin neighbour
  list, which its README places under the Academic Software License, r2SCAN parameters and the start of a C++ D3;
  last commit April 2025 (same caveat).

| Task | What | Files | Acceptance |
| --- | --- | --- | --- |
| S6-D01 | Research note: done; approved by the owner on 14 November 2026 (decision O4) | `docs/design/dispersion.md`, `docs/design/dispersion-measurements/` | done |
| S6-D02 | Kernel bake-off on the RTX 5060 Ti (needs Stage 1 only). Time each candidate for the pair sum and measure its error: a plain real-space sum at 40 bohr (receiver-owned kernel with a cell walk, float32 pair arithmetic, float64 sums); a shorter cutoff with a tail correction; a split sum (damped short range in real space, smooth long range on a coarse mesh); the long-range part updated every few steps | prototype kernels under `src/dispersion/`, `docs/validation/dispersion-bakeoff.md` | for NbCl5 (1,944 atoms), SrTiO3 (5,000) and water (5,184): ms per step of each candidate and its energy, pressure and force error against the 150 bohr reference; the default is named as the fastest candidate within the accuracy contract, or the 40 bohr sum if none is; the result goes into the ledger |
| S6-D03 | Framework: the `lm.Dispersion` record (`method`, `damping`, `functional` or raw parameters, `three_body`, `three_body_cutoff_bohr`, `pair_cutoff_bohr`, `cn_cutoff_bohr`, `smoothing`, `long_range` = `"auto"`, `"none"`, `"tail"`, `"split"`, `"mesh"` with the mesh options, `update_every`, `auto_methods` to restrict what `"auto"` may pick), the presets `"d3bj"`, `"d3bj-mace"`, `"d3bj-atm"`, `"d3-reference"`, `"d3zero"`, `"d3bjm"`, `"d3zerom"`, `"auto"`, a registry in which a variant is one entry, parameter tables with their sources, the label-table rule and refusals, the engine interface for a dispersion term (energy, forces, stress, an energy-only path, inside the captured step), CLI `--dispersion`, a `[dispersion]` TOML table, `loupmd dispersion list` | `loupmd/dispersion.py`, `src/dispersion/term.hpp`, `loupmd/cli/` | CPU tests: every preset expands to a complete record; an unknown combination is refused by name; a model whose data contains dispersion (OFF, OMOL-0, POLAR-1, the `spice_wB97M` and `omol` heads) is refused with the reason; zero damping with r2SCAN is refused for lack of parameters; equal records give equal metadata entries; an explicit `long_range` is used as given and reported as chosen by the user, never replaced by the automatic choice; `loupmd dispersion list` shows each method's known limits |
| S6-D04 | Real-space two-body D3 with zero, Becke-Johnson and both modified dampings, coordination-number-dependent C6, forces, stress, both cutoffs and the smooth switch | `src/dispersion/d3_pair.cu` | energy, forces and stress agree with simple-dftd3 at the same settings to 1e-6 eV/atom, 1e-5 eV/A and 1e-7 eV/A^3 for each damping on `NbCl5` and one molecular crystal; `"d3bj-mace"` agrees with `mace_mp(dispersion=True)` at the same tolerances |
| S6-D05 | The long-range methods the bake-off keeps: tail correction beyond the pair cutoff, split sum | `src/dispersion/d3_tail.cu`, `src/dispersion/d3_split.cu` | each stays within the accuracy contract on the whole validation set |
| S6-D06 | Reciprocal-space D3 for Becke-Johnson damping after FourierD3's method, implemented from the paper, with any reused MIT or Apache code attributed in `NOTICE.md`; default mesh settings tighter than the public code's 1.2 A spacing | `src/dispersion/d3_mesh.cu` | within the accuracy contract on the validation set at its default settings; cost at 5,000, 100,000 and 1,000,000 atoms recorded; refused for non-BJ dampings and with the three-body term until those are derived |
| S6-D07 | Automatic choice for `long_range="auto"`, on the user's own structure: at setup evaluate each candidate once on the starting structure against the plain real-space sum, drop any candidate outside the accuracy contract, time the rest and keep the fastest, as the engine does for GEMM plans. The tail correction is a candidate only for cells whose density is uniform on the scale of the cutoff. `auto_methods` restricts the candidates. The check repeats when the cell or the composition changes by an event, and on request | `src/dispersion/`, `loupmd/dispersion.py`, `loupmd/report.py` | on each benchmark system the chosen method is the fastest of those timed in S6-D02 within 5%; on a slab with vacuum the tail correction is not chosen; a candidate made wrong on purpose (a mesh far too coarse) is dropped; the report and `metadata.json` list every candidate with its measured deviation and time, and name the method as chosen automatically or by the user |
| S6-D08 | Three-body Axilrod-Teller-Muto term, off by default, with its own smoothed cutoff (20 bohr in `"d3bj-atm"`, 40 bohr in `"d3-reference"`) | `src/dispersion/d3_atm.cu` | agrees with simple-dftd3 at the same cutoffs to the tolerances of S6-D04; its cost per step is recorded and shown in the report |
| S6-D09 | Integration: stress in NPT, energy-only calls in Monte Carlo, restart, the tiled driver (a real-space sum with a cell walk, three sweeps for coordination numbers, energies and forces, and a position halo of the pair cutoff), the full record in `metadata.json` | `loupmd/simulation.py`, `src/tiled.cu` | an NPT run, a swap-MC run and a tiled run each complete with dispersion on and report it in the plan line; the tiled result agrees with the whole-system result within the accuracy contract |
| S6-D10 | `loupmd validate --dispersion`: compares the chosen settings with simple-dftd3 on the user's structure; documentation page on choosing a variant, with the recipe for adding D4 through its own ASE calculator | `loupmd/validate/`, `docs/guide/dispersion.md` | the command passes for every preset and fails for a deliberately wrong parameter set |

**Approved scope (decision O4).** The default is the fastest evaluation of the pair sum that is as accurate as the
conventional setting; the three-body term is offered but off by default; the mesh sum is in 1.0; D4 is not.
By decision O5 every summation method can also be chosen by name, and the automatic mode checks its candidates on the
user's own structure before it picks one.

**Accuracy contract of the default.** On every structure of the dispersion validation set (the six cells of the
note plus one metal and one layered compound other than graphite), the default's error against the converged
pair sum (simple-dftd3 at 150 bohr) is no larger than the error of a plain 40 bohr real-space sum, in energy
per atom, in pressure and in the largest force; in MD it meets the engine's drift criterion of 0.05
meV/atom/ps. If no candidate beats the 40 bohr sum within the contract, that sum is the default.

**The six questions S6-D01 answered** (answers in the note).

1. **Inventory.** Which dispersion methods and implementations matter for machine-learned potentials today: D2;
   D3 with zero, Becke-Johnson and the modified dampings; the three-body term; D4; the reciprocal-space sum; the GPU
   implementations (torch-dftd and its forks, the Grimme group's PyTorch packages, LAMMPS's D3 pair style, vendor
   CUDA kernels, the LASP D3 that the FourierD3 paper uses as a baseline); and methods beyond pairwise sums (MBD,
   XDM, VV10), with a verdict on whether each is in scope. For each: terms, parameters, treatment of periodicity,
   scaling, forces and stress, licence, state of maintenance, who uses it.
2. **Models.** For each label: the training functional, whether it already contains dispersion, which parameter
   sets exist for it, and what other users of the model default to (for example `mace_mp(dispersion=True)`).
3. **Measurements on our structures** (`NbCl5`, a molecular crystal, water, a layered material): differences in
   energy, forces and stress between dampings, between 40 bohr, 95 bohr and no cutoff, with and without the
   three-body term, with and without smoothing; NVE drift for truncated, smoothed and reciprocal-space sums; cost
   per step against the model's force call at 2,000, 5,000, 100,000 and 1,000,000 atoms.
4. **Large cells.** How each variant runs in tiled mode, where the pair cutoff (about 21 A at 40 bohr) is far beyond
   a tile's 6 A halo: a stored list, a cell loop without a list, or the mesh sum.
5. **User experience.** The `dispersion=` surface: shorthands, record fields, the default per label, what `"auto"`
   means, what is refused, the report line, how a user checks the result.
6. **Recommendation.** What to build natively and in which order, what to offer only through an optional adapter to
   an outside package, what to leave out and why, with an effort figure for each.

Scope changes after 14 November 2026 need the owner's approval; cut variants are listed in §11.

**Exit:** every row of §6 is done or an acknowledged cut, and every dispersion variant approved in S6-D01 passes its checks. Tag `v1.0.0rc1`.

### Stage 7: release (2.5 weeks)

| Task | What | Acceptance |
| --- | --- | --- |
| S7-T01 | Documentation site with FastMACE's structure: install, quickstart (CLI, Python and ASE for the same run), guides (MD, ASE, models, large cells, restart, Monte Carlo, NPT, dispersion, relax, benchmark yourself, FAQ), reference (CLI, models, integrators, output, environment, errors), autogenerated API | `mkdocs build --strict` passes; every CLI flag appears in the reference |
| S7-T02 | "Coming from FastMACE" page: the import line, the arguments that are refused, the differences in files and caches | the FastMACE quickstart runs with the changes the page lists and no others |
| S7-T03 | Licence audit: every vendored and ported file with its origin and licence; the cubins' build recipe; the D3 parameter file | `NOTICE.md` complete; `pip-licenses` of the run-time dependencies attached |
| S7-T04 | Five examples as FastMACE's (NVT anneal, NPT with dispersion melt-quench, swap MC, relaxations, ASE port) plus a large-cell example | each runs with `--steps 200` in the acceptance tier |
| S7-T05 | Reference benchmarks for the shipped `reference/*.json`: RTX 5060 Ti and, with the owner's go, RTX 4000 Ada, A100, L4 | each record carries its provenance block |
| S7-T06 | Final verification: full CPU suite, all GPU tiers, the gate for every label in whole and tiled mode, clean-install proof on the built wheel, A/B against `v1.0.0a1` | all pass; results in `docs/validation/release-1.0.0.md` |
| S7-T07 | `tools/release/make_release.sh`: git archive, build, `twine check --strict`, SHA-256; it never uploads | artefacts and hashes listed |
| S7-T08 | **Owner go-ahead:** name (A1), repository visibility, upload to PyPI | recorded |

---

## 6. Parity matrix: every FastMACE feature and where it lands

| FastMACE feature | Task | Note |
| --- | --- | --- |
| pip install, no compiler | S2-T01, S2-T02, S2-T12 | also no torch at run time |
| `doctor` | S2-T07 | |
| 19 model labels, hash-checked download, trust rule | S2-T04, S6-T06, S6-T07 | |
| Checkpoint used directly, no manual export | S2-T05, S3-T02 | export runs on demand and needs the export extra |
| `fm.load`, `ForceField` | S3-T02 | |
| `fm.md`, `RunResult`, report | S3-T07 | |
| `Simulation` run / relax / resume / events | S3-T05, S4-T12, S3-T04, S6-T01 | |
| `Output`: thermo CSV, trajectory, restart, metadata, simulation record | S3-T03, S3-T04, S3-T09 | |
| NVE, Langevin | S1-T04 | exist today |
| CSVR, Nose-Hoover chain, Berendsen | S4-T03 to S4-T05 | |
| NPT: BAOAB, MTK, Berendsen | S4-T06 to S4-T09 | experimental, as in FastMACE |
| Temperature schedules and profiles | S3-T06 | |
| `FixAtoms` | S4-T02 | |
| LBFGS, FIRE, Frechet cell, `relax`, `quench` | S4-T10 to S4-T12 | |
| Monte Carlo: swap, transmute, displace | S6-T02 to S6-T04 | |
| D3(BJ) dispersion | Stage 6D | goes beyond FastMACE: the user chooses damping, three-body term and the kind of sum; scope approved on 14 November 2026 (O4) |
| MACE-POLAR-1 | exists | neutral singlet, zero field, as today |
| ASE calculator and drop-in integrators | S2-T10, S4-T13 | |
| Per-atom energies | S1-T07, S6-T08 | |
| `check="gate"`, MC move-energy check | S3-T10, S6-T02 | |
| CLI `models md init run relax quench` | S2-T04, S3-T08, S4-T12 | |
| TOML and YAML run files, shipped protocols | S3-T08 | YAML only if PyYAML is installed |
| Provenance | S3-T09 | adds the NVIDIA driver number |
| Tiled and wavefront capacity path | Stage 5 | public here; a bench tool there |
| A/B timing harness | S2-T08 | public here |
| float64 forces | cut (A3) | refused, S6-T09 |
| Fallback to the MACE calculator ("reference tier") | cut (A3) | refused, S6-T09 |
| Bitwise-reproducible mode | cut (A3) | refused, S6-T09 |
| Planner tiers, backends, rewrite switches, calibration | not applicable | LoupMD has one engine; its own choices (kernel variants, GEMM plans, execution mode) are reported in the plan |

---

## 7. Test and validation strategy

- **CPU tier (`tests/cpu`)**: everything in Python that does not need forces from a model, on a fake core with a
  harmonic potential: config, errors, registry, cache, CLI parsing, output, restart, schedules, Monte Carlo
  statistics, tile planner and wavefront schedule invariants. Runs without torch and without a GPU.
- **GPU tiers (`tests/gpu`)**: fast (under 10 minutes: one model per path, the engine guards, one run per
  integrator), full (all families, restart, NPT, tiled against whole), acceptance (long conservation runs, examples,
  the gate for all labels).
- **Oracles**: MACE float64 for the gate; ASE's classes driven by `lm.calc` for integrators and optimisers;
  FastMACE's independent BAOAB reference; `torch-dftd` for D3; the whole-system call for the tiled and wavefront
  calls.
- **Every bug fix gets a test that failed first** (the practice of review rounds 5 and 6).
- **Stage exits** run the full CPU tier, the full GPU tier, the default gate and the A/B against the previous tag.

## 8. Benchmarking yourself (`loupmd bench`)

- **Protocol** (the one of `bench_md.py`, so ledger numbers stay comparable): Langevin 300 K, 1 fs, skin 0.5 A,
  friction 0.01/fs, 100 warm-up steps, median of 6 x 256 steps. `--protocol fastmace` runs FastMACE's (300 warm-up,
  4 x 300) for comparisons with its published tables.
- **Suites**: `standard` (`NbCl5` 3x3x3 and `SrTiO3` 10x10x10 for the chosen model), `quick` (one system, 2 x 128
  steps, under a minute), `families` (one system per model family, as `bench_family.py`), `capacity` (the tiled call
  at a size that does not fit whole).
- **Structures** ship in the wheel: `SrTiO3` and rocksalt NaCl (a = 5.64 A) are generated; `NbCl5_R2SCAN.cif` and
  the 216-molecule water box are copied in once the owner confirms they may be redistributed (S2-T08 asks before
  copying). Until then they are read from the folders of `$LOUPMD_BENCH_STRUCTURES` (§1.3 D18).
- **Output**: a table with ms/step, us per atom-step, ns/day at the timestep, block spread, final temperature and
  dangerous builds; the stored reference for the same GPU model with the ratio; a JSON record `loupmd.bench/1` with
  GPU name, compute capability, NVIDIA driver, library hash, model SHA-256, bundle key, protocol, per-block times and
  other processes found on the GPU.
- **Comparisons**: `--compare mace` times `mace-torch`'s ASE calculator on the same structure; `--compare fastmace`
  times FastMACE if it is installed; both print a ratio measured on this machine.
- **Development**: `--ab ENV=VAL,...` runs sides in fresh processes in the order a b b a, as `bench_md.py` does.
- **Refusals**: another process on the GPU stops the benchmark unless `--force`; a non-finite temperature marks the
  row `UNSTABLE`.

## 9. Sequencing, lanes and effort

```
Stage 0 -> Stage 1 -+-> Stage 2 -> Stage 3 -+-> Stage 4 (from S4-T10 on needs Stage 3) -+-> Stage 6 -> Stage 7
                    |                       |                                             |
                    +-> Stage 4 (S4-T01 to S4-T09 need only Stage 1) ---------------------+
                    +-> Stage 5 (needs S1-T08, S1-T09, S1-T10; S5-T12 needs Stage 3) -----+
```

- Serial effort: about 37 weeks. With four lanes after Stage 1 (front end; dynamics; tiled; dispersion) about
  14-16 weeks. Both are rough estimates.
- The dispersion research is done. The kernel bake-off (S6-D02) needs only Stage 1 and decides the default
  early; the framework (S6-D03) needs Stage 3 for the front-end record; S6-D09 needs Stage 5 for the tiled driver.
- The local GPU is the shared resource: one job at a time, so GPU-heavy acceptance work is scheduled, not run in
  parallel lanes.
- `release/v1` merges `round6-hardening` at every stage exit (S0-T04). Speed work continues there; the release
  branch does not change kernels except where a task says so (S1-T08, S1-T09, S4-T06).

## 10. Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| The sweep split or the variable-cell graph slows the standard step | misses the no-regression rule | S1-T08 and S4-T06 are A/B tested in both orders before merging; the variable-cell graph is a separate variant used only by NPT |
| Ghost-domain forces differ from the periodic call beyond rounding | tiled mode fails the gate | S1-T09's equivalence test comes before any tile code |
| Tiled per-atom cost well above the 1.2x estimate | large cells slower than hoped | S5-T10 measures and screens the receiver cap; FastMACE's own figure is 1.34x |
| Export needs more of FastMACE than the six vendored modules | export extra pulls in FastMACE | S2-T03's byte-identity test shows it at once; fall back to a pinned `fastmace` dependency in the export extra |
| A manylinux CUDA 13 build image or the runtime wheel names differ from the assumption | wheel build blocked | S2-T01 pins the names from torch's own dependency list; S2-T02 can build in the vendor's devel image and repair the tag |
| One GPU for development, optimisation and acceptance | stages slip | acceptance runs are batched at stage exits; the workstations only with the owner's go |
| Models not yet tested on the generic path (MP-0, MPA-0, MatPES, OMOL-0) need engine changes | S6-T06 and S6-T07 grow | each is its own task with its own gate; an unsupported label is documented, not hidden |
| The layout move collides with open optimisation work | lost edits | S0-T03 is one agreed commit after a merge |
| Dispersion code elsewhere carries licences that do not fit (the torch-dftd fork's skin list is under the ASL) | it cannot be copied | S6-D01 records each project's licence; LoupMD implements from the papers and reuses only MIT or BSD code, with attribution |
| Dispersion cutoffs of about 21 A in tiled mode, against a 6 A halo | dispersion dominates the step or does not fit large cells | the real-space sum walks a cell grid without a stored list and needs only positions, species and coordination numbers in its halo (note §4); S6-D09 builds it |
| No candidate is both faster than the 40 bohr sum and within the accuracy contract | dispersion costs more than the model on small cells | S6-D02 measures this first; the report states the cost share and names the cheaper explicit settings |

## 11. Cuts and open points

- Cut on purpose (A3): float64 forces, the reference-tier fallback, bitwise reproducibility. Each is refused with a
  message, never silently replaced.
- Not at 1.0, same as FastMACE: non-periodic systems, several GPUs, batching of structures, ASE constraints other
  than `FixAtoms`, tiled MACE-POLAR-1. The domain abstraction is the route to the first two.
- Dispersion, cut by decision O4: D4 (no periodic GPU implementation to build on; its charge solve is a dense
  matrix problem in the reference code), D2, the rare dampings D3(op), D3(CSO) and D3(Z), and every method that
  needs the electron density (MBD, TS, XDM, dDsC, VV10). Reasons in `docs/design/dispersion.md` §6.3.
- Not at 1.0, differs from an "any model" promise: an architecture whose product signature is not in the shipped
  manifest needs `loupmd build-kernels` and `nvcc`.
- Open with the owner: A1, A2, A4 and A6 of §1.2; redistribution of the benchmark structures (S2-T08); which
  machines may run the capacity records (S5-T11) and the reference benchmarks (S7-T05).
- Not verified while writing this plan: the FastMACE inventory beyond the tiled documents and the rename plan;
  whether the generic exporter handles the six untested labels; export time per element set (not recorded
  anywhere); every estimate marked as such above.

## 12. Sources

- LoupMD at `0336751`: `include/loupmd.h`, `src/*.cu`, `python/*.py`, `tests/`, `Makefile`, `AGENTS.md`,
  `STATUS_AND_PROFILE.md`, `OPTIMISATION_LEDGER.md`.
- FastMACE (`D:\Margraf\FastMACE`): `docs/design/tiled-backend.md`, `docs/design/tiled-wavefront.md`,
  `docs/validation/tiled/README.md`, `docs/validation/tiled/tiled-wavefront.md`, `plans/billion-atoms-20261111.md`,
  `plans/future-rename-loup.md`, `fastmace/model/tiled.py`, `fastmace/bench/step_time.py`,
  `fastmace/planner/selfcheck.py`; and two read-only inventories of its install, API, CLI, documentation, features,
  benchmark and validation tooling made on 14 November 2026.
- Dispersion starting points, each from one page fetch on 14 November 2026: `https://github.com/vicvaleeva/FourierD3`
  with arXiv:2607.15103, and `https://github.com/CheukHinHoJerry/torch-dftd`; FastMACE's own term in
  `fastmace/longrange/d3.py`.
