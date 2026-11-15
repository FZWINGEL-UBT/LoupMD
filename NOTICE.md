# Third-party notices

LoupMD is distributed under the MIT licence (`LICENSE`); files ported from ASE will carry LGPL-2.1-or-later
(owner decision O6 of 15 November 2026, `PLAN.md` §1.1). The files below are binaries or generated sources; each
is listed with what it was made from. `tests/test_notice.py` checks that every tracked binary and generated file
appears here.

## Compiled GEMM kernels (`cubins/`)

Tensor-core GEMM kernels that the engine loads at run time as autotune candidates (`src/engine.cu`; never used
for models with long-range terms). They are compiled from NVIDIA CUTLASS headers (BSD-3-Clause, © 2017-2026
NVIDIA CORPORATION & AFFILIATES, <https://github.com/NVIDIA/cutlass>; licence text in `cubins/LICENSE.cutlass`).

Two sets are tracked. The `tf32x3_hyb_*` cubins are the ones the engine loads by default; they were built from
FastMACE's sources `fastmace/kernels/cutlass/tf32x3_sm80.cu` and `hybrid_f16.h` (MIT, © 2026 Fabian Zwingel)
with `-DFM_HYBRID_F16`. The `gemm_tc_*` cubins are LoupMD's own builds from `cubins/src/gemm_tc.cu` and
`cubins/src/hybrid_f16.h` (derived from the same FastMACE sources), with their recipe in
`cubins/src/build_gemm_tc.sh` and its record in `cubins/src/gemm_tc.build.txt`; the engine loads one of them only
when `LOUP_HYBRID_CUBIN` names it (testing, until they are gated and adopted).

| File | SHA-256 | Origin | Build record |
| --- | --- | --- | --- |
| `cubins/tf32x3_hyb_sm80.cubin` | `c5dd8728d23987ad8ae3883f5968956f665cd7a9ecabe7b8de27499a865e4d14` | byte-identical to FastMACE's `fastmace/kernels/cutlass/tf32x3_hyb_sm80.cubin` | FastMACE `tf32x3_sm80.build.txt`: CUTLASS v4.8.1, commit `2c85f497547eb441f0fe80bfb0fb2e6f9add1719`; nvcc 13.3 (V13.3.33); 11 November 2026 |
| `cubins/tf32x3_hyb_sm90.cubin` | `c4f97a4002dc53a4e2f7fc6c3afa7ac42edc04ff5b9e5b471f1d2d6e4cac7cd7` | byte-identical to FastMACE's `tf32x3_hyb_sm90.cubin` | as above |
| `cubins/tf32x3_hyb_sm120.cubin` | `9a949b7065906f5284e1190a0994576d1d186e3f723592488b54f81d59396f1f` | the same sources compiled for `sm_120`; not in FastMACE | **none.** `gemm_tc_hyb_sm120.cubin` below is its recorded rebuild; the default switches to it once it is gated (`PLAN.md` §1.3, D6) |
| `cubins/gemm_tc_hyb_sm80.cubin` | `62c046fa0ff8a3f20d070b1a8e3398d767f9cb16690c81ca6fb167a0ecd9ca39` | `cubins/src/gemm_tc.cu`: TF32 + FP16 hybrid, 6 tiles, with beta | `cubins/src/gemm_tc.build.txt`: CUTLASS commit `2c85f497547eb441f0fe80bfb0fb2e6f9add1719` (v4.8.1); nvcc 13.3 (V13.3.33) |
| `cubins/gemm_tc_3x_sm80.cubin` | `ec3767d5d0bfd486f70ac1640477bd0f42b5584eca319eba437307ce97f3f42e` | `cubins/src/gemm_tc.cu`: 3xTF32, 6 tiles, with beta | as above |
| `cubins/gemm_tc_hyb_sm90.cubin` | `640b2c1c2a471bfd3fb5125fa31fb44fb00b75ed71f08a73832c616f86d763c4` | `cubins/src/gemm_tc.cu`: TF32 + FP16 hybrid, 3 tiles | as above |
| `cubins/gemm_tc_hyb_sm120.cubin` | `7b6816a69d42a3f5375af98a049afcd26ced6549f7ddb8a3a083cfb5a75e8f75` | `cubins/src/gemm_tc.cu`: TF32 + FP16 hybrid, 3 tiles | as above |

What the hybrid kernels compute: the product of two float32 matrices, each element split into an 11-bit high
part and a low part. The high-by-high products run on TF32 tensor cores and the two cross products in one FP16
tensor-core instruction on parts that fit float16 exactly (`hybrid_f16.h`). Elements must stay below 32752 in
magnitude; the engine drops the hybrid for a handle whose result turns non-finite and evaluates again without it.
The `3x` cubin is CUTLASS's 3xTF32 without the FP16 instruction.

## Generated sources (`src/generated_*`)

Written by the generators in `python/`; not edited by hand. They hold kernel source specialised to model
architectures (channel counts, irreps, tensor-product paths, coupling coefficients). They hold no model weights:
those are read from a `.loup` bundle at run time.

| File | Generator | Made with |
| --- | --- | --- |
| `src/generated_kernels.cuh` | `python/gen_kernels.py`, from the architecture of `mace-omat-0-medium.model` | FastMACE's factored symmetric contraction (`fastmace/kernels/cuda/sc_factored.py`, `sc_generated.py`; MIT, © 2026 Fabian Zwingel) and the streaming schedule of FastMACE's `fastmace/kernels/oeq_streaming/` (`schedule.py`, adapted from OpenEquivariance pull request 223 by P. Fuchs, commit `ef051c3f9efa2d1daf74cfbe3beaad152235f47d`; BSD-3-Clause, © 2025 The Regents of the University of California, through Lawrence Berkeley National Laboratory). The generator takes the schedule from that module and emits the kernel text itself; MACE's `convert_e3nn_cueq` (MIT, © 2022 ACEsuit/mace) and e3nn (MIT) supply the coupling coefficients |
| `src/generated_kernels_signature.h` | `python/gen_kernels.py` | the signature string that bundles of the specialised path must carry |
| `src/generated_generic.cu` | `python/gen_generic.py`, from the architectures of the model files found at generation time | FastMACE's factored symmetric contraction (as above); e3nn's Wigner 3j coefficients; sympy for the spherical-harmonic gradient |
| `src/generated_generic.cuh` | `python/gen_generic.py` | as above |

## Code and data ported from FastMACE

FastMACE is MIT, © 2026 Fabian Zwingel, the same owner as LoupMD.

| File | Ported from (FastMACE commit `9b801229`) | Changes |
| --- | --- | --- |
| `loupmd/models.py` | `fastmace/model/labels.py` (the label table: release URLs, SHA-256, licences, heads and training functionals with their sources) and the checkpoint lookup, download and trust functions of `fastmace/model/loading.py` | errors are LoupMD's; the model folder variable is `LOUPMD_MODELS`; nothing is unpickled in this module |
| `loupmd/errors.py` | the message shape of `fastmace/errors.py` (`what`, `Why:`, `Next:`) | LoupMD's own classes |
| `loupmd/protocols.py` | `fastmace/protocols/schedule.py` and `profiles.py`: segments, `Schedule`, whole-step resolution (half up), the end-point target, `ScheduleState`, the profiles `slow`, `fast`, `ultrafast`, `anneal` | errors are `InputError`; profile files are TOML only; NPT, `pressure_GPa` and dispersion in a profile are refused; optimiser names are kept as data |
| `loupmd/quench.py` | `fastmace/protocols/quench.py`: the frame selection per target-temperature bin, the quench's settings, files and CSV columns | each frame is relaxed with `lm.optimize` on the host; the run's schedule and cadence are read from `metadata.json` |
| `loupmd/output.py` | the chunk scheme, file names and thermo header of `fastmace/io/output.py` and `io/trajectory.py`, and the durable-write helpers of `io/durable.py` | rows and frames come from the engine (device thermo buffer, `get_state`), not from torch tensors; a background thread writes them |
| `loupmd/quick.py`, `loupmd/integrators.py`, `loupmd/config.py` (`Output`) | the signature, argument rules and defaults of `fastmace/quick.py` (`md`, `RunResult`, initial velocities, default friction and output directory), the record names of `fastmace/integrators/__init__.py`, the `Output` record of `fastmace/io/output.py` | only what LoupMD's engine has (NVE, Langevin); FastMACE's planner arguments are refused by name |

The model files themselves are not part of LoupMD. Each label's licence (MIT or the Academic Software License) is
in the table and is shown by `loupmd models`.

## Modules ported from ASE (LGPL-2.1-or-later)

By owner decision O6 (15 November 2026) the modules below, which port classes of ASE 3.27 (LGPL-2.1-or-later,
© the ASE developers, <https://gitlab.com/ase/ase>) so that they agree with ASE to rounding, are distributed
under **LGPL-2.1-or-later**, not MIT. Each starts with `# SPDX-License-Identifier: LGPL-2.1-or-later` and a
provenance line. The licence text is `LICENSES/LGPL-2.1-or-later.txt`. They are plain Python source.

| LGPL file | Ported from (ASE 3.27) |
| --- | --- |
| `loupmd/optimize/lbfgs.py` | `ase/optimize/lbfgs.py` (`LBFGS` without line search) |
| `loupmd/optimize/fire.py` | `ase/optimize/fire.py` (`FIRE` without the downhill check) |
| `loupmd/optimize/cell.py` | `ase/filters.py` (`UnitCellFilter`'s cell mapping and `FrechetCellFilter`) |

The records of `loupmd/optimize/__init__.py` and `loupmd/integrators.py` take ASE's argument names and defaults
(an interface, not ported code) and are MIT. The Langevin and velocity-Verlet steps of `src/integrator.cu`
implement the published algorithms with ASE's coefficient conventions; they were written for the engine and are
MIT.

## Not yet in this file

Later stages add files that need an entry here when they land: the FastMACE modules vendored for export
(S2-T03), further modules ported from ASE (Stage 4: the remaining integrators), the D3
reference parameters (Stage 6D) and the benchmark structures (S2-T08, if the owner allows them in the
wheel). S7-T03 is the audit.
