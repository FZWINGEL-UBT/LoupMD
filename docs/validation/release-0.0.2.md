# Release verification: 0.0.2

The release is version 0.0.2. The owner asked for it on 20 November 2026: "yes push and make sure all docs are up
to date and release the current version on pypi and github releases as 0.0.2" (`PLAN.md` O43). That ended O37,
under which 0.0.2 was to wait until the development checkout's optimisations were merged and verified. The merge landed
on 19 November (O38, O40), and the work after it is in `STATUS_AND_PROFILE.md` §6.36 to §6.41.

Everything here was measured on one machine: an NVIDIA GeForce RTX 5060 Ti (compute capability 12.0, 16 GB) in
WSL 2 (Ubuntu) on Windows 11. The CPU suite ran on the workstation leto.

- Date: 20 November 2026.
- Code: `release/v1` at `9f1442e`, the commit tagged `v0.0.2`. Since `v0.0.1` (`f3e5d8d`) it carries:
  - the export versions of O36 (D91);
  - the merge of `develop` (the new kernels, one kernel path, the run-time compiler, the LAMMPS package);
  - `run_style loupmd`'s integrators (O40);
  - out of memory under WSL 2 (`loup::device_malloc`) and the destroy-race fix (O41);
  - the tiled mode in LAMMPS (O42, LT0-LT5);
  - the barostats in tiled mode (LT6 (a), D99);
  - the documentation audit and the version (`9f1442e`).
  `CHANGELOG.md` lists them for users.
- Wheel: built from `9f1442e` by `tools/release/make_release.sh`. Its library has SHA-256 `89b9dc040e0565a7...`.

## Summary

| Check | Command | Result |
| --- | --- | --- |
| CPU suite | `pytest -q tests/cpu -n 24` on leto, `git archive` of `9f1442e` | 497 passed, 5 skipped |
| GPU, fast tier | `pytest -q tests/gpu -m fast`, the `sm_120` build of `ada48e8` | 380 passed, 1 failed, 51 skipped (the failure: below) |
| GPU, full tier | `pytest -q tests/gpu -m full`, the same library | 59 passed, 21 skipped (the LAMMPS files without `LOUPMD_LMP`) |
| LAMMPS | `tests/gpu/test_lammps.py`, `test_lammps_run_style.py`, `test_lammps_tiled.py` with `LOUPMD_LMP` | 41 passed; 24 passed |
| Documentation site | `mkdocs build --strict` | built |
| Release build | `tools/release/make_release.sh` | wheel and source tarball of `9f1442e` built in the container; six compute capabilities; `twine check --strict` passed; nothing uploaded |
| The 0.0.2 wheel with its export extra in a fresh environment | `pip install "<wheel>[export]"`, then `loupmd models --download`, `loupmd doctor`, `loupmd bench --quick` from an empty cache, `loupmd md --check gate`, the quickstart | passed: doctor 9 of 9, the gate, the quickstart's five steps (below) |
| Gates from the wheel's package | `loupmd validate`, its exporter writing the bundles | 6 of 6 passed (four labels, whole and tiled; below) |
| Barostat in tiled mode from the wheel | `lm.Simulation` with `MTK` on 5,000 atoms, `capacity="whole"` against `CapacityConfig(mode="tiled", receivers=2000)` | ran tiled; after 200 steps the volume agrees with the whole-cell run to 1e-8 and the positions to 4.2e-7 A |
| Rocky Linux 8 container, no GPU | `pip install` of the wheel, `loupmd doctor` | installs; the library loads (only the GPU check fails) |
| GPU, fast tier on the wheel's library | `pytest -q tests/gpu -m fast` with `LOUP_LIB` set to the installed library | 380 passed, 1 failed, 51 skipped (a timing test: below) |

`ada48e8` and `9f1442e` have the same C++/CUDA sources but for the version string in `include/loupmd.h`.

**The fast tier's failure.** `test_memory_plan.py` compares the predicted memory of a whole system with the drop
in free memory while the system is set up. At 20,480 atoms the drop read 1,963 MB against 2,073 MB predicted, while
the engine's own count of the same system was 2,072.9 MB. Run again, the file had 9 passed and 1 failed, at 40,000
atoms and the other way (4,287 MB measured, 3,980 MB predicted, own count 3,979.7 MB). The 0.0.2 changes do not
touch the whole system's memory. Under WSL 2 the free memory moves with what other programs on the card hold
(`STATUS_AND_PROFILE.md` §6.35). A native Linux workstation would settle it; none had a free GPU that evening.
On the wheel's library the same file passed.

**The fast tier's failure on the wheel's library.** `test_optimize_gpu.py::test_the_optimiser_s_own_time_at_5000_atoms`
requires the optimiser's own time to stay under 2% of the force calls. LBFGS read 2.02% (0.146 ms per step against
7.07 ms per force call). FIRE read 1.74%, with 8.65 ms per force call on the same structure, so the force calls'
times moved by a fifth between the two halves of the test. The test failed once before in the same way and passed on
a rerun (`PLAN.md` S6-D04). Both failures are timings or memory readings on the desktop card under WSL 2, which
since 20 November serves correctness checks only; all four workstations' GPUs were in use that evening.

## The release build

In `dist/release-0.0.2-9f1442eaf6cf/` on the build machine:

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `loupmd-0.0.2-py3-none-manylinux_2_28_x86_64.whl` | 12,775,565 | `bbf883be9fb77a039da2634282edb5f0a11d5e3a555bfba391181a61201acc99` |
| `loupmd-0.0.2-src.tar.gz` (`git archive` of the commit) | 2,557,938 | `71134cffcce61b9307ce1c796a0b350e813893acf4a2e7ebf7262560c619de1e` |

- The build image is the one of 0.0.1 (`tools/release/Dockerfile.wheel`: CUDA 13.3.1 on Rocky Linux 8,
  gcc-toolset-13).
- The library (48,750,856 bytes) holds 15 ELF images for each of compute capability 8.0, 8.6, 8.9, 9.0, 10.0 and
  12.0, plus one `sm_75` image, as 0.0.1's did.
- It asks for symbol versions up to `GLIBC_2.14`, `GLIBCXX_3.4.22` and `CXXABI_1.3.11`. It needs
  `libcudart.so.13`, `libcublas.so.13` and `libcublasLt.so.13` from the pinned NVIDIA wheels.
- The metadata names the licence expression `MIT AND BSD-3-Clause AND LGPL-2.1-or-later`, the four licence files,
  the three project links and `Requires-Python: >=3.10`.

## The 0.0.2 wheel

The wheel was installed with its export extra into a fresh environment, Python 3.12.3
(`pip install "<wheel>[export]"`, every dependency from PyPI). The extra brought torch 2.15.0+cu132, mace-torch
0.3.17 and e3nn 0.4.4, and `pip check` found no broken requirement. The model cache, the bundle cache and the
reference cache started empty, so the wheel's own exporter wrote every bundle and the wheel's library ran it.

- `loupmd models --download omat0-medium`: 8 s; the file's SHA-256 matches the label table.
- `loupmd doctor`: 9 checks, none failed; the library reports `0.0.2+b9f306bbe32d40c9`.
- `loupmd bench --quick` from the empty cache: ran in 50 s, the export included, and again in 5 s from the cache.
  Its step time is not a measurement here: the desktop card serves correctness checks only.
- `loupmd md nacl.cif --repeat 3 3 3 --check gate`, 2,000 steps: the gate passed (worst energy / force / stress
  ratio 0.41 / 1.56 / 1.08 on two rattled copies), 30 neighbour rebuilds, none dangerous.
- The quickstart's five steps (`docs/quickstart.md`): the command line, Python and the ASE calculator, each 10,000
  steps of NVT on 216 atoms; `loupmd relax` (23 steps to 0.0082 eV/A); the relaxation in Python (23 and 30 steps).

**The floor of the wheel.** In a plain `rockylinux:8` container (Rocky Linux 8.10, glibc 2.28, Python 3.11.13, no
CUDA toolkit, no GPU) `pip install` of the wheel worked (numpy 2.4.6, scipy 1.17.1, cuda-toolkit 13.2.2 there), and
`loupmd doctor` found the three CUDA libraries in the NVIDIA wheels and loaded `libloupmd` (`0.0.2+b9f306bbe32d40c9`);
its only failure was the GPU check.

**Gates from the wheel's package** (worst energy / force / stress ratio; a quantity passes up to 3, or below its
floor):

| Set | Label | Whole | Tiled |
| --- | --- | --- | --- |
| `oxide` | `omat0-medium` | 0.57 / 2.08 / 2.28 | 0.64 / 2.14 / 2.49 |
| `crystals` | `mh1-omat` | 0.13 / 0.85 / 0.38 | 0.10 / 0.87 / 0.46 |
| `crystals` | `polar1-s` (reference on the CPU) | 0.00 / 0.96 / 1.94 | not applicable |
| `organic` | `off23-small` | 0.00 / 0.63 / 2.12 | not run |

`omat0-medium` on `oxide` passed with less margin than before. Its worst row is the second copy of `SrTiO3` 3x3x3
(the larger rattle), at force 2.05 to 2.08 and stress 2.25 to 2.28 for all three kernel variants. 0.0.1 from its
wheel read 1.83 / 1.24 whole (the specialised kernels), and the generic kernels before the merge 1.96 / 1.71 with
references from the new export versions (`docs/validation/polar-stress-and-export-pins.md`). The cause was not
looked into.

**Bundles of the new export versions.** Of the bundles the wheel's exporter wrote, one has a key the development
cache also held (MACE-OFF23 small for H and O): the two files are byte-identical, written once by torch 2.15.0 and
mace-torch 0.3.17 and once by torch 2.14.1 and mace-torch 0.3.16. The other keys changed with the exporter's
sources since that cache was filled.

The first `polar1-s` run hit the check's own limit of 420 s after its float64 reference had been computed on the
CPU and cached; the run again took 59 s.

**Barostat in tiled mode.** `lm.Simulation` with `MTK` (300 K, 1 GPa, `tdamp_fs=100`, `pdamp_fs=500`) on `SrTiO3`
10x10x10 (5,000 atoms, rattled by 0.02 A, at rest), once with `capacity="whole"` and once with
`lm.config.CapacityConfig(mode="tiled", receivers=2000)`, which ran tiled. The volume after 50, 100, 150 and 200
steps was 61,754.67, 60,699.69, 60,476.93 and 61,489.99 A^3 in both runs (relative difference at most 9e-9), and
after 200 steps the positions differ by at most 4.2e-7 A. A first attempt passed the `CapacityConfig` to `lm.md`,
which takes the mode only and refused it. `docs/guide/large-cells.md` had shown it next to `lm.md`; it now names
`lm.Simulation`.

## Not checked

- Any GPU but the RTX 5060 Ti. The wheel's code for compute capability 8.0, 8.6, 8.9, 9.0 and 10.0 was compiled
  from the same sources and not run from the wheel.
- The acceptance tier and the gate for the sixteen labels not named above, on the code after the merge's
  verification (`STATUS_AND_PROFILE.md` §6.36: acceptance 16 passed on 19 November).
- `loupmd build-kernels` from the wheel.
- Timings and memory readings on a native Linux workstation: the two failures above.

## The release from PyPI

To follow the owner's upload.
