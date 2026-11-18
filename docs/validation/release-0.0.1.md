# Release verification: 0.0.1

The release is version 0.0.1 (the owner's decision of 18 November 2026, `PLAN.md` O34). Until that afternoon the
tree carried the version 0.1.0a1, and the checks below ran on builds that report it; the wheel of 0.0.1 and what
was run on it are in [the 0.0.1 wheel](#the-001-wheel).

The final checks of `PLAN.md` S7-T06 on the release branch, run again on the code after the merge of
`round6-hardening` (`PLAN.md` O30). Everything here was measured on one machine: an NVIDIA GeForce RTX 5060 Ti
(compute capability 12.0, 16 GB) in WSL 2 (Ubuntu) on Windows 11. No other GPU was used; the other GPUs of S7-T05
wait for the owner.

- Date: 18 November 2026.
- Code: `release/v1` with the merge `09d6daa`, the shared layer arena `2f98c9a` and the memory plan of `0171c3e`
  and `d58e935` (it adds the long-range models' buffers to the plan's count; no kernel changes).
- Library: `libloupmd.so`, SHA-256 `8c82bc7e3ee9762a...`, built for `sm_120` from that code; it reports
  `0.1.0a1+d1283291ef91a017`.
- Wheel: built from `aa44e42`, which adds the fixes the first release build and the first clean install asked for
  (`6b3313f`, `aa44e42`: the release scripts, the bundle cache's key, one error message; no C++/CUDA source
  changed). Its library, for six compute capabilities, has SHA-256 `dd9fdc04f4d0ee59...` and reports
  `0.1.0a1+488eb2ee43695540`.
- Release wheel: built from `f3e5d8d`, the commit tagged `v0.0.1`. It differs from `aa44e42` by the Python fixes
  of the documentation audit (`5b4e690`), the version string, the package metadata and the licence notices (the
  code that follows ASE is LGPL-2.1-or-later, `PLAN.md` O35); in the C++/CUDA sources only the version in
  `include/loupmd.h` and comments changed. Its library has SHA-256 `eeb796ff83c9e240...` and reports
  `0.0.1+f1c133ca134def66`.

The results of 16 November, on the branch before the merge, are in this file's history (`6d06428`).

## Summary

| Check | Command | Result |
| --- | --- | --- |
| CPU suite | `pytest -q tests/cpu` | 492 passed, 5 skipped at `aa44e42` (490 before its two new tests) |
| GPU, fast tier | `pytest -q tests/gpu -m fast` | 360 passed, 7 skipped (the NbCl5 structure file is not in the folder); the same on the wheel's library |
| GPU, full tier | `pytest -q tests/gpu -m full` | 48 passed |
| GPU, acceptance tier (gate per family, examples) | `pytest -q tests/gpu -m acceptance` | 15 passed, 1 skipped (the same file) |
| Gate per label, whole and tiled | `loupmd validate --model LABEL --set SET --capacity whole/tiled` | 37 passed: all 20 labels whole, the 17 without a long-range term tiled |
| A/B against the branch before the merge | `loupmd bench --model omat0-medium --suite standard --ab`, a b b a twice | faster: 0.927 (`NbCl5` 3x3x3), 0.961 (`SrTiO3` 10x10x10) |
| Memory plan against measurement, capacity | `docs/validation/capacity.md`, `STATUS_AND_PROFILE.md` §6.35 | plan within 0.3-5% of the device peak on 22 tiled plans; 10,985,000 atoms run with OMAT-0 medium, 20,480,000 with OMAT-0 small |
| Release build | `tools/release/make_release.sh` | wheel and source tarball of `aa44e42` built in the container; six compute capabilities; `twine check --strict` passed; nothing uploaded |
| Clean install of the built wheel | `pip install` of the wheel in a fresh environment without torch, then `loupmd doctor`, `loupmd bench --quick`, the quickstart's steps | passed (after two fixes); `loupmd validate` from the wheel's package: 12 of 12 passed |
| The wheel's library against the `sm_120` build | `loupmd bench --model omat0-medium --suite standard --ab`, a b b a twice | a tie: 1.002 (`NbCl5` 3x3x3), 1.000 (`SrTiO3` 10x10x10) |
| The 0.0.1 wheel with its export extra in a fresh environment | `pip install "<wheel>[export]"` from PyPI, then `loupmd doctor`, `loupmd bench --quick` from an empty cache, `loupmd md --check gate`, `loupmd validate`, the quickstart | passed: doctor 9 of 9, 1.708 ms/step (reference 1.704), the gate on five runs, the quickstart's five steps; the fast GPU tier on its library 360 passed, 7 skipped |
| The release from PyPI, after the upload | `pip install "loupmd[export]"` by name into a fresh environment, then `loupmd models --download`, `loupmd doctor`, `loupmd bench --quick`, `loupmd md --check gate`, the quickstart | passed: the file on PyPI is the tested wheel (the same SHA-256); doctor 9 of 9, 1.694 ms/step (reference 1.704), the gate, the quickstart's five steps |

## Gate per label

`loupmd validate --model LABEL --set SET` (the 3x float32-against-float64 rule of the accuracy gate) for every
label of `docs/reference/models.md`, once whole (`--capacity whole`) and once through the tiled call
(`--capacity tiled`). The `oxide` and `crystals` sets read their structures from the folders that
`LOUPMD_BENCH_STRUCTURES` names; `polar1-*`, `off23-large`, `off23b-medium` and `off24-medium` compute their
float64 reference on the CPU (`--ref-device cpu`). Every run passed. Worst force / stress ratio of each run (a
quantity passes up to 3, or below its floor):

| Set | Label | Whole | Tiled |
| --- | --- | --- | --- |
| `oxide` | `mp0-medium` | 0.79 / 0.52 | 0.74 / 1.06 |
| `oxide` | `mp0-large` | 1.34 / 0.87 | 1.35 / 0.69 |
| `oxide` | `mp0b3-medium` | 1.16 / 0.81 | 1.20 / 0.77 |
| `oxide` | `mpa0-medium` | 2.04 / 2.08 | 2.08 / 2.21 |
| `oxide` | `omat0-medium` | 1.85 / 1.24 | 1.86 / 1.29 |
| `oxide` | `omat0-small` | 1.35 / 1.34 | 1.33 / 1.44 |
| `oxide` | `matpes-pbe` | 1.04 / 1.55 | 1.09 / 1.58 |
| `oxide` | `matpes-r2scan` | 1.29 / 1.26 | 1.36 / 1.22 |
| `crystals` | `mh1-omat` | 0.84 / 0.31 | 0.88 / 0.30 |
| `crystals` | `mh1-mp` | 0.80 / 0.25 | 0.82 / 0.25 |
| `crystals` | `mh1-r2scan` | 0.86 / 0.31 | 0.90 / 0.34 |
| `crystals` | `polar1-s` | 0.96 / 2.15 | not applicable |
| `crystals` | `polar1-m` | 2.13 / 0.79 | not applicable |
| `crystals` | `polar1-l` | 2.25 / 0.80 | not applicable |
| `organic` | `omol0-xl` | 0.84 / 0.61 | 0.88 / 0.55 |
| `organic` | `off23-small` | 0.64 / 1.88 | 0.63 / 1.73 |
| `organic` | `off23-medium` | 0.35 / 0.64 | 0.37 / 0.56 |
| `organic` | `off23-large` | 0.42 / 0.97 | 0.47 / 0.97 |
| `organic` | `off23b-medium` | 1.02 / 0.51 | 0.97 / 0.50 |
| `organic` | `off24-medium` | 0.69 / 1.02 | 0.68 / 0.97 |

The tiled runs use default engines: since the merge a tiled system leaves the skip-chain fold off itself
(`PLAN.md` D85), and `omol0-xl` needs no setting to run in tiles. The `polar1-m` run was repeated once: its first
attempt hit the 400 s limit this check set per run while the reference was computed on the CPU, and passed in 14 s
with the reference cached.

The `MACE-POLAR-1` models have no tiled mode: their long-range term is a sum over the whole cell's reciprocal
space, which tiles do not split (`capacity="tiled"` refuses them; `PLAN.md` §11 has the post-release item). FastMACE
has no tiled mode for them either.

## A/B against the branch before the merge

`loupmd bench --model omat0-medium --suite standard --ab LOUP_LIB=<the library of 97e365d> -`: both sides run the
tip's Python code, side a on the library built before the merge and side b on the tip's, each run in a fresh
process, in the order a b b a, twice (ms/step):

| System | before the merge (a) | tip (b) | b / a |
| --- | --- | --- | --- |
| `NbCl5` 3x3x3 (1,944 atoms) | 1.863, 1.857, 1.866, 1.864 | 1.724, 1.726, 1.728, 1.729 | 0.927 |
| `SrTiO3` 10x10x10 (5,000 atoms) | 8.069, 8.096, 8.118, 8.128 | 7.780, 7.778, 7.789, 7.810 | 0.961 |

The sides do not overlap: the merged kernels are 7.3% and 3.9% faster on the standard benchmark. The shipped
reference numbers for this GPU (`loupmd/bench/reference.json`) were measured again on the tip's library, two runs
each: `omat0-medium` `NbCl5` 3x3x3 1.720 ms/step (1.881 before), `SrTiO3` 10x10x10 7.725 (8.243), the quick
`SrTiO3` 6x6x6 1.704 (1.833); `omat0-small` `SrTiO3` 10x10x10 4.276 (4.540).

(The earlier check compared the tip with the tag `release-stage6`, `PLAN.md` D83, on one library, since no
C++/CUDA source had changed between them; after the merge the comparison that says something is the library's.)

## The release build and the clean install

Docker Desktop runs again on the build machine (18 November 2026; `PLAN.md` D82), so
`tools/release/make_release.sh` ran for the first time. Nothing was uploaded.

### What the first run found

| Where | Defect | Fix |
| --- | --- | --- |
| `make_release.sh`, `build_wheel.sh` | In Git Bash the scripts turn the path conversion off for docker's arguments, and `git.exe` then does not take the root as `/d/...`: the build stopped at its first `git` call | `6b3313f`: the root is held as `D:/...`, which bash, git and docker take |
| `build_wheel.sh`, `Makefile` | The container got `WHEEL_ARCHS` as an empty string, make's `?=` kept it, and `nvcc` compiled without `-gencode` for its own default architecture (it stopped at `__reduce_max_sync`) | `6b3313f`: the container script unsets an empty list, and the Makefile takes an empty list as the default six |
| `loupmd/bundles.py` | The bundle cache's key hashes `src/generated_kernels_signature.h` with the exporters. A wheel has no `src` folder and a missing file counted as empty: a wheel's key left the kernels' signature out (a later wheel with regenerated kernels would have found the old bundles), and a bundle exported from a checkout of the same commit was not found | `aa44e42`: the wheel ships the header in `loupmd/_lib`, and the key reads it there when there is no checkout |
| `loupmd/bundles.py` | In an environment without the export extra, a model whose bundle is not in the cache ended with "the export ... failed" and the last lines of the exporter's traceback (`ModuleNotFoundError: No module named 'torch'`) | `aa44e42`: the miss names the missing modules, `pip install 'loupmd[export]'`, and the path a bundle exported elsewhere goes to |

### The artefacts

In `dist/release-0.1.0a1-aa44e42f5ac0/` on the build machine (`dist/` is not tracked):

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `loupmd-0.1.0a1-py3-none-manylinux_2_28_x86_64.whl` | 11,304,968 | `e90203d3a8fccfde4161fe297dea44a546bc54eb627be4f42c15ac56c5de7424` |
| `loupmd-0.1.0a1-src.tar.gz` (`git archive` of the commit) | 2,283,310 | `d96f311202b97931e89dd0d64a9edde1146a983ee14e49895223f9e7540a6307` |

- Build image (`tools/release/Dockerfile.wheel`): NVIDIA's CUDA 13.3.1 devel image on Rocky Linux 8 with
  gcc-toolset-13: `nvcc` 13.3.73, `g++` 13.3.1, glibc 2.28, Python 3.11.13.
- The library for compute capability 8.0, 8.6, 8.9, 9.0, 10.0 and 12.0 built in 76 s (`make -j4 wheel-lib`);
  `cuobjdump` lists 15 ELF images for each target.
- The library asks for symbol versions up to `GLIBC_2.14`, `GLIBCXX_3.4.22` and `CXXABI_1.3.11` (manylinux_2_28
  allows 2.28, 3.4.25 and 1.3.11). It needs `libcudart.so.13`, `libcublas.so.13` and `libcublasLt.so.13`, which
  come from the pinned NVIDIA wheels; for that reason `auditwheel show` names the tag `linux_x86_64`.
- The wheel has 112 files, none outside the package; `twine check --strict` passed.
- Two builds of one tree (that of `6b3313f`) gave libraries of different bytes (`2099e871...`, `9d733e0d...`): the
  build is not reproducible bit for bit, and the hashes above name these files only.

### The clean install

A new `python3 -m venv` in WSL 2 (Ubuntu, glibc 2.39, Python 3.12.3) and `pip install` of the wheel file alone:
69 s. It installed numpy 2.5.5, ASE 3.29.0, scipy 1.18.1 and `cuda-toolkit` 13.0.3 (`nvidia-cuda-runtime`
13.0.96, `nvidia-cublas` 13.1.1.3, `nvidia-cufft` 12.0.0.61); torch and mace cannot be imported there; the
environment takes 1.6 GB. Every command below ran the installed package from a folder outside the checkout.

| Step | Result |
| --- | --- |
| `loupmd doctor` | 9 checks, 0 failed, 1 warning (the export extra is not installed) |
| `loupmd bench --quick --model omat0-medium`, empty cache | exit 1 with: "no bundle of mace-omat-0-medium.model for elements [8, 22, 38] is in the cache, and this environment cannot export one. Why: the export extra is not installed (missing: torch, mace, e3nn). Next: Install it into this environment: pip install 'loupmd[export]'. Or copy a bundle ..." |
| the same, the user's cache (it holds the bundle a checkout of this code exported) | the bundle is found; `SrTiO3` 6x6x6: 1.733 ms/step (reference 1.704, ratio 1.017), on the system's CUDA 13.3 runtime |
| the same, with the loader path on the NVIDIA wheels' libraries (`runtime_info` names `nvidia/cu13/lib/libcudart.so.13`) | 1.723 ms/step (ratio 1.011) |
| the quickstart's `lm.md` (200 steps, `SrTiO3` 4x4x4, 320 atoms) | ran: 0.62 ms/step in the engine, report and output files written |
| the ASE calculator (`lm.calc`): energy, forces, stress | ran: -7.982533 eV/atom |
| `lm.md(..., dispersion="d3bj")`, 50 steps | ran; the automatic long-range check computed its split sums (cuFFT from the NVIDIA wheel) |

The first wheel (`6b3313f`, before the two fixes to `bundles.py`) read 1.738, 1.741 and 1.740 ms/step in the same
benchmark, with a bundle placed under its key by hand.

**The floor of the tag.** In a plain `rockylinux:8` container (Rocky Linux 8.10, glibc 2.28, Python 3.11.13, no
CUDA toolkit, no GPU) `pip install` of the wheel worked (numpy 2.4.6, scipy 1.17.1 there), and `loupmd doctor`
found the three CUDA libraries in the NVIDIA wheels and loaded `libloupmd` (`0.1.0a1+488eb2ee43695540`); its only
failure was the GPU check, as it must be there.

**The gate from the wheel's package.** `loupmd validate` needs the export extra. It ran with the wheel alone on
`PYTHONPATH` (`pip install --no-deps --target`), from the development environment's Python 3.10.12, whose torch
2.14.1+cu130, mace-torch 0.3.16 and cuequivariance 0.12.0 are the versions the extra pins, and with an empty bundle
cache: the wheel's own exporter wrote the 24 bundles, and the wheel's library ran them. All twelve runs passed
(worst energy / force / stress ratio; a quantity passes up to 3, or below its floor):

| Set | Label | Whole | Tiled |
| --- | --- | --- | --- |
| `oxide` | `omat0-medium` | 0.57 / 1.83 / 1.24 | 0.59 / 1.89 / 1.32 |
| `oxide` | `omat0-small` | 0.38 / 1.32 / 1.34 | 0.39 / 1.31 / 1.38 |
| `oxide` | `mpa0-medium` | 0.30 / 2.04 / 2.14 | not run |
| `crystals` | `mh1-omat` | 0.10 / 0.86 / 0.31 | 0.09 / 0.86 / 0.29 |
| `crystals` | `polar1-s` (reference on the CPU) | 0.00 / 0.84 / 2.18 | not applicable |
| `organic` | `omol0-xl` | 0.00 / 0.85 / 0.57 | 0.00 / 0.86 / 0.58 |
| `organic` | `off23-medium` | 0.01 / 0.35 / 0.54 | not run |
| `organic` | `off23-small` | not run | 0.00 / 0.67 / 1.73 |

The first `omol0-xl` whole run hit the 400 s limit this check set per run, while the exporter was still writing
that model's bundles; it passed in 26 s once they were in the cache.

**The tree's tests on the wheel's library.** `pytest -q tests/gpu -m fast` with `LOUP_LIB` on the installed wheel's
library: 360 passed, 7 skipped (445 s), as on the `sm_120` build.

**Speed of the wheel's library.** `loupmd bench --model omat0-medium --suite standard --ab - LOUP_LIB=<the wheel's
library>`: side a is the `sm_120` build of the same sources (`8c82bc7e...`), side b the wheel's library for six
targets, each run a fresh process, in the order a b b a, twice (ms/step):

| System | `sm_120` build (a) | wheel's library (b) | b / a |
| --- | --- | --- | --- |
| `NbCl5` 3x3x3 (1,944 atoms) | 1.759, 1.765, 1.759, 1.753 | 1.757, 1.768, 1.755, 1.772 | 1.002 |
| `SrTiO3` 10x10x10 (5,000 atoms) | 7.865, 7.902, 7.905, 7.913 | 7.887, 7.908, 7.890, 7.900 | 1.000 |

The sides overlap: the library with code for six compute capabilities is as fast on this GPU as the build for one.
Both sides read about 2% above the reference numbers measured in the morning (1.720 and 7.725), as the quick
benchmark above did. The machine was not quiet (another session compiled and ran its own GPU jobs between these
runs); the cause was not looked for, and `loupmd/bench/reference.json` keeps the morning's numbers.

**Not checked on this wheel.** `pip install "loupmd[export]"` into a fresh environment (the gate above used the
development environment's torch and mace; done on the 0.0.1 wheel, below); `loupmd build-kernels` from the wheel;
the wheel's code for compute capability 8.0, 8.6, 8.9, 9.0 and 10.0, which has not run on any GPU (the
workstations with other GPUs need the university network, which the build machine was not on).

## The 0.0.1 wheel

Three commits make the release out of the code checked above. `b920ecb` sets the version 0.0.1, adds the project
links to `pyproject.toml` (it had none) and writes `README.md` for the PyPI page. `ef7705b` makes the install page
and the quickstart pip-only. `f3e5d8d` puts the code that follows ASE under LGPL-2.1-or-later (the owner's decision
O35: licence headers, `NOTICE.md`, the README and the run report's licence line; in the C++/CUDA sources only
comments changed). `f3e5d8d` is the commit of the tag `v0.0.1`, and the wheel built from it with
`tools/release/make_release.sh v0.0.1` is the file for the upload. Before the build, `loupmd` was free on PyPI and
on TestPyPI (both answered 404 for the project on 18 November 2026). The owner uploaded the wheel to PyPI the same
day; what was checked after that is in [from PyPI](#from-pypi).

In `dist/release-0.0.1-f3e5d8d834af/` on the build machine:

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `loupmd-0.0.1-py3-none-manylinux_2_28_x86_64.whl` | 11,306,507 | `5faee689275cca50359002d4cb020da06cbaa2f9f89fbde78d52ac71f9c6225f` |
| `loupmd-0.0.1-src.tar.gz` (`git archive` of the commit) | 2,300,778 | `0f5d33dce268893a806ef5583b83970fca9bb41f3ab1577f6134a898e3380acb` |

The image and the checks are those of the build above: the library for six compute capabilities built in 72 s, 15
ELF images per target, symbol versions up to `GLIBC_2.14`, `GLIBCXX_3.4.22` and `CXXABI_1.3.11`, 112 files,
`twine check --strict` passed. The metadata (version 2.4) names the licence expression
`MIT AND BSD-3-Clause AND LGPL-2.1-or-later`, the four licence files and the three project links. Its library has
SHA-256 `eeb796ff83c9e240...` and reports `0.0.1+f1c133ca134def66`.

An earlier build of 0.0.1, from `b920ecb` before the licence change (`dist/release-0.0.1-b920ecbf3a91/`, SHA-256
`cf8d31c7...6857954`), passed the same checks the same afternoon (doctor 9 of 9; 1.693 and 1.690 ms/step; the same
gates; the quickstart; the fast tier 360 passed, 7 skipped). It carries the old licence notices and is not the file
to upload.

### A fresh environment with the export extra

This is what `pip install "loupmd[export]"` gives a new user, and the check the section above left open. A new
`python3 -m venv` in WSL 2 (Python 3.12.3) and `pip install --no-cache-dir "<the wheel file>[export]"`, every
dependency from PyPI: 268 s, 67 distributions, 5.9 GB, `pip check` finds no broken requirement. It installed torch
2.14.1 (which reports `2.14.1+cu130`), mace-torch 0.3.16, e3nn 0.4.4, cuequivariance 0.12.0, numpy 2.5.5, ASE
3.29.0 and `cuda-toolkit` 13.0.3: torch pins the same `cuda-toolkit` version as the package, so one set of CUDA
libraries serves both.

Every command ran the installed package from a folder outside the checkout, with an empty cache (`LOUPMD_CACHE`),
the model files from `LOUPMD_MODELS` and none of the development set-up's variables:

| Step | Result |
| --- | --- |
| `loupmd doctor` | 9 checks, 0 failed, 0 warnings |
| `loupmd bench --quick --model omat0-medium`, empty cache | the environment's own exporter wrote the bundle; `SrTiO3` 6x6x6: 1.708 ms/step (reference 1.704, ratio 1.002); 43 s with the export |
| the same again | 1.701 ms/step (ratio 0.998); 4 s |
| `loupmd md nacl.cif --model omat0-medium --temperature 300 --steps 2000 --repeat 3 3 3 --check gate` | the gate passed (energy 0.43, force 1.58, stress 1.21 of the 3 allowed, two rattled copies); 2,000 steps at 0.295 ms/step in the engine; report and output files written; 46 s. The report's licence line: "LoupMD: MIT, its engine library LGPL-2.1-or-later" |
| `loupmd validate --model omat0-medium --set oxide --capacity whole` | passed: 0.56 / 2.01 / 1.58 |
| the same, `--capacity tiled` | passed: 0.55 / 2.02 / 1.76 |
| `loupmd validate --model off23-small --set organic --capacity whole` | passed: 0.00 / 0.64 / 1.85 |
| `loupmd validate --model mh1-omat --set crystals --capacity whole` | passed: 0.10 / 0.83 / 0.28 |
| `loupmd models --download omat0-medium`, a second empty cache, `LOUPMD_MODELS` unset | 79,463,999 bytes in 5 s; the SHA-256 matches the label table |
| the quickstart's `loupmd md structure.cif --model omat0-medium --temperature 300 --steps 10000 --initial-velocities maxwell_boltzmann --seed 1 --output run`, with the downloaded model (rattled rock salt 3x3x3, 216 atoms) | ran: the export and 10,000 steps in 40 s, 0.308 ms/step in the engine |
| the quickstart's Python script (`lm.md`) | ran in 5 s: 0.298 ms/step in the engine |
| its ASE script (`lm.calc`, then `loupmd.ase.md.Langevin` for 10,000 steps) | ran in 7 s: -728.720242 eV, 0.302 ms/step in the engine; ASE 3.29 prints a deprecation notice for its own `MaxwellBoltzmannDistribution` |
| `loupmd relax structure.cif --model omat0-medium --fmax 0.01` | converged after 23 steps, largest force 0.0082 eV/A; `structure.relaxed.extxyz` written |
| the quickstart's Python relaxation, positions and then the cell (`FrechetCell`) | converged after 23 and after 30 steps |

The float64 references of these gates were computed in this environment, by its torch and mace, into its own empty
reference cache; the table of the 0.1.0a1 wheel above read the development cache's references. The stress ratios
of `omat0-medium` are larger here (1.58 and 1.76 against 1.24 and 1.32); both are within the limit of 3, and the
cause of the difference was not looked for.

Seen and left: a run with `--check gate` prints torch's and mace's own warnings before the report (about 25 lines:
`torch.jit` deprecation notices and the `weights_only` notice for the checkpoint the package has just verified by
its SHA-256).

**The floor of the tag.** In a plain `rockylinux:8` container (Rocky Linux 8.10, glibc 2.28, Python 3.11.13, no
CUDA toolkit, no GPU) `pip install` of the 0.0.1 wheel worked (numpy 2.4.6, scipy 1.17.1 there), and `loupmd
doctor` found the three CUDA libraries in the NVIDIA wheels and loaded `libloupmd` (`0.0.1+f1c133ca134def66`); its
only failure was the GPU check.

**The tree's tests on the 0.0.1 wheel's library.** `pytest -q tests/gpu -m fast` with `LOUP_LIB` on the installed
0.0.1 wheel's library: 360 passed, 7 skipped (440 s), as on the 0.1.0a1 builds.

**Not repeated on the 0.0.1 builds.** The full and acceptance tiers, the gate for every label and the A/B of the
wheel's library against the `sm_120` build: they ran on the 0.1.0a1 builds above, from the same C++/CUDA sources.
`tests/cpu` at `f3e5d8d`: 495 passed, 5 skipped.

**Not checked.** `loupmd build-kernels` from the wheel; the wheel's code for compute capability 8.0, 8.6, 8.9, 9.0
and 10.0, which has not run on any GPU; an upload to TestPyPI as a rehearsal (the upload went to PyPI directly).

### From PyPI

The owner uploaded the wheel on 18 November 2026 (PyPI's upload time: 16:09:14 UTC). PyPI lists one file for the
release, `loupmd-0.0.1-py3-none-manylinux_2_28_x86_64.whl` with 11,306,507 bytes and the SHA-256
`5faee689275cca50359002d4cb020da06cbaa2f9f89fbde78d52ac71f9c6225f`: the file of the table above. `release/v1` and
the tag `v0.0.1` (on `f3e5d8d`) were pushed to the repository right after, and the documentation site was rebuilt
from that push, so its install page and its quickstart name `pip install "loupmd[export]"` from the minute the
file was there. The tag is where the source of the LGPL library is offered.

Then the install by name: a new `python3 -m venv` in WSL 2 (Python 3.12.3, pip 26.3, no pip configuration) and
`pip install --no-cache-dir "loupmd[export]"`: 217 s, 67 distributions, 5.9 GB, `pip check` finds no broken
requirement, the same versions as in the environment above. pip took `loupmd` from the index (the installed
package has no `direct_url.json`), and the installed library has the SHA-256 of the tested wheel's
(`eeb796ff83c9e240...`).

Every step ran as a pip user runs it: `LOUPMD_MODELS` unset, the model from `loupmd models --download` in an
otherwise empty cache, none of the development set-up's variables, a folder outside the checkout.

| Step | Result |
| --- | --- |
| `loupmd models --download omat0-medium` | 79,463,999 bytes in 6 s; the SHA-256 matches the label table |
| `loupmd doctor` (with the model files' hashes) | 9 checks, 0 failed, 0 warnings |
| `loupmd bench --quick --model omat0-medium`, no bundle in the cache | the environment's own exporter wrote the bundle; `SrTiO3` 6x6x6: 1.694 ms/step (reference 1.704, ratio 0.994); 43 s with the export |
| the same again | 1.692 ms/step (ratio 0.993); 3 s |
| `loupmd md nacl.cif --model omat0-medium --temperature 300 --steps 2000 --repeat 3 3 3 --check gate` | the gate passed (energy 0.41, force 1.56, stress 1.27 of the 3 allowed, two rattled copies); 0.302 ms/step in the engine; 47 s |
| the quickstart's `loupmd md structure.cif ...` for 10,000 steps (rattled rock salt 3x3x3, 216 atoms) | ran in 6 s (the bundle was in the cache by then): 0.304 ms/step in the engine |
| the quickstart's Python script (`lm.md`) | ran in 5 s: 0.309 ms/step in the engine |
| its ASE script | ran in 7 s: -728.720242 eV, 0.303 ms/step in the engine |
| `loupmd relax structure.cif --model omat0-medium --fmax 0.01` | converged after 23 steps, largest force 0.0082 eV/A |
| the quickstart's Python relaxation, positions and then the cell | converged after 23 and after 30 steps |

Not run on this install: `loupmd validate` on the four pairs, the Rocky Linux 8 container and the fast GPU tier.
They ran on the same file before the upload (the sections above).

Seen and left: the system's CUDA 13.3 libraries are on this machine's loader path, and `loupmd.runtime_info()`
names them for the plain `loupmd` command, while torch brings the NVIDIA wheels' CUDA 13.0.3 libraries. After
these runs the cache holds stored GEMM plans for two cuBLASLt versions (`lt130101` and `lt130501` in the file
names), as the cache of the environment above does: the processes did not all load the same CUDA libraries. Which
process loaded which was not looked into; every run passed.

## Tests whose premise the merge changed

The 16 November run had one failure in the full tier, the sweep replay against the ordinary pass on MH-1 at
1.054e-7 eV/atom against a floor of 1e-7. Since the merge the ordinary whole-system pass folds the skip chain and the
sweeps do not, which moves forces by about 2e-6 eV/A and stress by up to 2e-8 eV/A^3 (`STATUS_AND_PROFILE.md`
§6.34); the tests that compare a tiled, domain or sweep result with a whole-system one at rounding-level bounds now
take the whole-system reference without the fold (`tests/gpu/gpu_support.py`, `unfolded`). No test failed in the
runs above.
