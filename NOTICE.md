# Third-party notices

LoupMD is distributed under the MIT licence (`LICENSE`); files ported from ASE will carry LGPL-2.1-or-later
(owner decision O6 of 15 November 2026, `PLAN.md` §1.1). The files below are binaries or generated sources; each
is listed with what it was made from. `tests/test_notice.py` checks that every tracked binary and generated file
appears here.

## Compiled GEMM kernels (`cubins/`)

Tensor-core GEMM kernels that the engine loads at run time as autotune candidates (`HybCubin` in
`src/engine.cu`; never used for models with long-range terms). They are compiled from NVIDIA CUTLASS headers
(BSD-3-Clause, © 2017-2026 NVIDIA CORPORATION & AFFILIATES, <https://github.com/NVIDIA/cutlass>; licence text in
`cubins/LICENSE`) with FastMACE's sources `fastmace/kernels/cutlass/tf32x3_sm80.cu` and `hybrid_f16.h`
(MIT, © 2026 Fabian Zwingel), built with `-DFM_HYBRID_F16`.

| File | SHA-256 | Origin | Build record |
| --- | --- | --- | --- |
| `cubins/tf32x3_hyb_sm80.cubin` | `c5dd8728d23987ad8ae3883f5968956f665cd7a9ecabe7b8de27499a865e4d14` | byte-identical to FastMACE's `fastmace/kernels/cutlass/tf32x3_hyb_sm80.cubin` | FastMACE `tf32x3_sm80.build.txt`: CUTLASS v4.8.1, commit `2c85f497547eb441f0fe80bfb0fb2e6f9add1719`; nvcc 13.3 (V13.3.33); 11 November 2026 |
| `cubins/tf32x3_hyb_sm90.cubin` | `c4f97a4002dc53a4e2f7fc6c3afa7ac42edc04ff5b9e5b471f1d2d6e4cac7cd7` | byte-identical to FastMACE's `tf32x3_hyb_sm90.cubin` | as above |
| `cubins/tf32x3_hyb_sm120.cubin` | `9a949b7065906f5284e1190a0994576d1d186e3f723592488b54f81d59396f1f` | the same sources compiled for `sm_120`; not in FastMACE | **none in either repository yet.** The developing session is producing it (`PLAN.md` §1.3, D6) |

What the kernels compute: the product of two float32 matrices, each element split into an 11-bit high part and
a low part. The high-by-high products run on TF32 tensor cores and the two cross products in one FP16
tensor-core instruction on parts that fit float16 exactly (FastMACE `hybrid_f16.h`). Elements must stay below
32752 in magnitude.

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

The model files themselves are not part of LoupMD. Each label's licence (MIT or the Academic Software License) is
in the table and is shown by `loupmd models`.

## Not yet in this file

Later stages add files that need an entry here when they land: the FastMACE modules vendored for export
(S2-T03), the modules ported from ASE (Stage 4; LGPL-2.1-or-later under default
A6), the D3 reference parameters (Stage 6D) and the benchmark structures (S2-T08, if the owner allows them in the
wheel). S7-T03 is the audit.
