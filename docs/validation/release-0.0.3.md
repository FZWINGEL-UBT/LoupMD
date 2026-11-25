# Release verification: 0.0.3

The release is version 0.0.3. The owner asked for it on 24 November 2026 (`PLAN.md` O44): MACE-POLAR-1 had refused a
total charge and a spin multiplicity, "can you implement a fix on the release branch and test it and then push and
release it as 0.0.3?", then "also include the new a100 defaults in the 0.0.3 release".

The wheel was run on one machine: an NVIDIA GeForce RTX 5060 Ti (compute capability 12.0, 16 GB) in WSL 2 (Ubuntu)
on Windows 11. A build of the same sources for compute capability 8.9 was run on the workstation leeloo (RTX 4000
Ada, 20 GB, native Linux), and the CPU suite on the workstation leto.

- Date: 24 and 25 November 2026.
- Code: `release/v1` at `6d5e5d8`, the commit tagged `v0.0.3`. Since `v0.0.2` (`9f1442e`) it carries:
  - the branch `mh1-two-half-adjoint` (`bb2c973`, 41 commits): the two-half FP16 tensor-core kernels and the tuned
    defaults on compute capability 8.9, 8.6 and 8.0;
  - MACE-POLAR-1's total charge and spin multiplicity (`ca2aa8c`);
  - the version, the changelog and the documentation (`5a7edde`);
  - one test brought in line with the two-half kernels (`6d5e5d8`, below).
  `CHANGELOG.md` lists them for users.
- Wheel: built from `6d5e5d8` by `tools/release/make_release.sh`. Its library has SHA-256 `198b70f3b59f7193...`.

## Summary

| Check | Command | Result |
| --- | --- | --- |
| CPU suite | `pytest -q tests/cpu` on leto, `git archive` of `6d5e5d8` | 500 passed, 5 skipped |
| Documentation site | `mkdocs build --strict` | built |
| Release build | `tools/release/make_release.sh` | wheel and source tarball of `6d5e5d8` built in the container; six compute capabilities; `twine check --strict` passed; nothing uploaded |
| The 0.0.3 wheel with its export extra in a fresh environment | `pip install "<wheel>[export]"`, then `loupmd models --download`, `loupmd doctor`, `loupmd bench --quick` from an empty cache, `loupmd md --check gate`, the quickstart | passed: doctor 9 of 9, the gate, the quickstart's five steps (below) |
| Gates from the wheel's package | `loupmd validate`, its exporter writing the bundles | 6 of 6 passed after one was run again: `omat0-medium` on `oxide` failed once on its float32 yardstick (below) |
| MACE-POLAR-1 with a charge and a spin multiplicity, from the wheel | `loupmd validate --model polar1-s --set organic` with `--charge 1` and with `--charge -1 --spin-multiplicity 2`; `loupmd md` on 216 atoms neutral and with each | both gates passed (below); the three runs completed |
| Barostat in tiled mode from the wheel | `lm.Simulation` with `MTK` on 5,000 atoms, `capacity="whole"` against `CapacityConfig(mode="tiled", receivers=2000)` | ran tiled; the volumes agree to 5e-9 and the positions after 200 steps to 3.5e-7 A |
| GPU, fast tier on the wheel's library | `pytest -q tests/gpu -m fast` with `LOUP_LIB` set to the installed library | 386 passed, 1 failed, 51 skipped (a memory reading: below) |
| GPU, full tier on the wheel's library | `pytest -q tests/gpu -m full`, the same library | 60 passed, 21 skipped (the LAMMPS files without `LOUPMD_LMP`) |
| Compute capability 8.9: fast and full tier | the same two commands on leeloo, an `sm_89` build of `5a7edde` | fast 362 passed, 76 skipped; full 57 passed, 1 failed, 23 skipped; the failure is the test `6d5e5d8` changes, which then passed there |
| Compute capability 8.9: charged MACE-POLAR-1-M and -L | `tools/dev/test_gate.py --ref-device cpu` on leeloo | passed (below) |

`5a7edde` and `6d5e5d8` differ in `tests/gpu/test_tiled.py` only.

**The fast tier's failure.** `test_memory_plan.py::test_the_tiled_plan_is_the_calls_device_peak[0.25]` compares the
tiled plan's bytes with the call's device peak, which the engine reads as the least free memory of the whole card
during the call. It read 3,274 MB against a plan of 2,238 MB. The file run again by itself on the same library had
10 passed, with a peak of 2,168 MB for that plan. The reading counts whatever another program takes from the card
during the call, and the desktop's own programs held 2.3 GB of it that night. On leeloo the test passed in the fast
tier.

**The full tier's failure on compute capability 8.9.** `test_tile_gemm_buckets_time_fewer_shapes_and_keep_the_result`
asserts that size buckets time at most 60% of the GEMM shapes. Where the two-half kernels are the default they take
the row-dependent GEMMs, which are then not timed at all: the RTX 4000 Ada timed 56 of 90 shapes. With the three
two-half switches off it times 98 of 314. The test now sets them off on both sides (`6d5e5d8`); energy and forces
agreed either way.

## The release build

In `dist/release-0.0.3-6d5e5d86987a/` on the build machine:

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `loupmd-0.0.3-py3-none-manylinux_2_28_x86_64.whl` | 16,510,779 | `bb59fb338acddecac7dc4ad48a1cb2ca9c5a4117f18aa42a9d0ea35ae5913205` |
| `loupmd-0.0.3-src.tar.gz` (`git archive` of the commit) | 2,815,380 | `165aecb17ace521bff350a635927313d3fdb8755d008f66b44300c31b76dee4f` |

- The build image is the one of 0.0.1 and 0.0.2 (`tools/release/Dockerfile.wheel`: CUDA 13.3.1 on Rocky Linux 8,
  gcc-toolset-13).
- The library (62,826,144 bytes) holds 16 ELF images for each of compute capability 8.0, 8.6, 8.9, 9.0, 10.0 and
  12.0, plus one `sm_75` image. 0.0.2's held 15 for each.
- It asks for symbol versions up to `GLIBC_2.14`, `GLIBCXX_3.4.22` and `CXXABI_1.3.11`. It needs
  `libcudart.so.13`, `libcublas.so.13` and `libcublasLt.so.13` from the pinned NVIDIA wheels.
- The metadata names the licence expression `MIT AND BSD-3-Clause AND LGPL-2.1-or-later`, the four licence files,
  the three project links and `Requires-Python: >=3.10`.

## The 0.0.3 wheel

The wheel was installed with its export extra into a fresh environment, Python 3.12.3
(`pip install "<wheel>[export]"`, every dependency from PyPI). The extra brought torch 2.15.0+cu132, mace-torch
0.3.17 and e3nn 0.4.4, and `pip check` found no broken requirement. The model cache, the bundle cache and the
reference cache started empty, so the wheel's own exporter wrote every bundle and the wheel's library ran it.

- `loupmd models --download omat0-medium`: 4 s; the file's SHA-256 matches the label table.
- `loupmd doctor`: 9 checks, none failed; the library reports `0.0.3+d6410155687b00cf`.
- `loupmd bench --quick` from the empty cache: ran in 49 s, the export included, and again in 4 s from the cache.
  Its step time is not a measurement here: the desktop card serves correctness checks only.
- `loupmd md nacl.cif --repeat 3 3 3 --check gate`: the gate passed (worst energy / force / stress ratio
  0.41 / 1.54 / 1.23 on two rattled copies).
- The quickstart's five steps (`docs/quickstart.md`): the command line, Python and the ASE calculator, each 10,000
  steps of NVT on 216 atoms; `loupmd relax` (23 steps to 0.0082 eV/A); the relaxation in Python (23 and 30 steps).

**Gates from the wheel's package** (worst energy / force / stress ratio; a quantity passes up to 3, or below its
floor):

| Set | Label | Whole | Tiled |
| --- | --- | --- | --- |
| `oxide` | `omat0-medium` | 0.65 / 2.00 / 3.77 failed; five runs again 0.55 / 1.74 to 2.03 / 1.71 to 2.47 | 0.55 / 1.98 / 4.17 failed on the same references |
| `crystals` | `mh1-omat` | 0.10 / 0.82 / 0.33 | 0.10 / 0.84 / 0.33 |
| `crystals` | `polar1-s` (reference on the CPU) | 0.00 / 0.52 / 1.09 | not applicable |
| `organic` | `off23-small` | 0.00 / 0.63 / 1.77 | not run |
| `organic` | `polar1-s`, `--charge 1` | 0.00 / 0.14 / 0.33 | not applicable |
| `organic` | `polar1-s`, `--charge -1 --spin-multiplicity 2` | 0.00 / 0.15 / 0.34 | not applicable |

**`omat0-medium` on `oxide`.** The first run failed on the stress of the second copy of `SrTiO3` 3x3x3 (the larger
rattle), for all three kernel variants: LoupMD is 2.30e-7 eV/A^3 from MACE float64 there, and MACE float32 was
6.1e-8 from it in that run, a ratio of 3.75 (4.17 tiled, at 2.56e-7). The cell is the one 0.0.2 passed with the least
margin, at 2.28: LoupMD was 2.50e-7 off then and MACE float32 1.1e-7. So LoupMD's error is smaller than 0.0.2's and
the yardstick was smaller still. The float64 references of the two releases agree to 1e-16; the float32 ones, from
the same versions of torch and mace-torch on the same card, differ by up to 8.2e-8 in the stress.

Five more runs of the same wheel, each with its own reference cache:

| MACE references | LoupMD's stress error (eV/A^3) | Ratio | Worst force ratio |
| --- | --- | --- | --- |
| 0.0.2's file | 2.32e-7 | 2.12 | 2.03 |
| computed anew on the GPU | 2.34e-7 | 1.68 | 2.03 |
| computed anew on the GPU | 2.31e-7 | 2.10 | 1.88 |
| computed anew on the GPU | 2.34e-7 | 2.47 | 2.02 |
| computed anew on the CPU (`--ref-device cpu`) | 2.31e-7 | 2.18 | 1.74 |

All five passed. LoupMD's error is the same in every run (2.30e-7 to 2.34e-7); MACE float32's stress on this cell
moved between 6.1e-8 and 1.4e-7 from one process to the next, which is what decided the first run. Why MACE
float32 moves that much on this cell was not looked into. The tiled run was not repeated: its error against 0.0.2's
yardstick is a ratio of 2.3.

**MACE-POLAR-1 with a charge and a spin multiplicity.** New in 0.0.3. From the wheel, `polar1-s` on the `organic`
set (urea, naphthalene, paracetamol and 216 water molecules, two copies each) passed with `--charge 1` and with
`--charge -1 --spin-multiplicity 2`; the references are MACE with the same charge and spin. `loupmd md` on 216
atoms of NaCl ran 200 steps neutral, with `--charge 1` and with `--charge -1 --spin-multiplicity 2`.

The larger models were gated with `tools/dev/test_gate.py` on builds of the same engine sources (worst energy /
force / stress ratio):

| Model | Set | Charge, multiplicity | GPU | Result |
| --- | --- | --- | --- | --- |
| MACE-POLAR-1-S | `oxide` | +1, 1; 0, 3; 0, 1 | RTX 5060 Ti | passed |
| MACE-POLAR-1-M | `organic` | +1, 1 | RTX 5060 Ti | passed |
| MACE-POLAR-1-M | `oxide` | -1, 2 | RTX 5060 Ti | passed |
| MACE-POLAR-1-M | `organic` | -2, 3 | RTX 4000 Ada, reference on the CPU | passed (stress 0.19 on the cell below) |
| MACE-POLAR-1-L | `organic` | +1, 2 | RTX 4000 Ada, reference on the CPU | passed |

**References of the larger POLAR-1 models belong on the CPU.** On the RTX 5060 Ti the gate of MACE-POLAR-1-M with
charge -2 and multiplicity 3 failed on the stress of one water cell (2.8e-7 eV/A^3, a ratio of 8.2), with the
float64 reference computed on the GPU. The engine's stress there agrees with a reference of that cell computed by
itself (5e-9), with a finite difference of MACE's energy, and with itself on a fresh engine against one that had
run another cell (4e-10). A reference computed on the CPU is the same alone and in the gate's sequence (5.6e-18),
and with it the gate passed on leeloo. MACE float64 for POLAR-1-M does not fit a 20 GB card beside the engine
(out of memory on leeloo); on the 16 GB card under WSL 2 it ran by spilling into system memory. What went wrong in
that GPU reference was not found. Gate MACE-POLAR-1-M and -L with `--ref-device cpu`.

**Barostat in tiled mode.** `lm.Simulation` with `MTK` (300 K, 1 GPa, `tdamp_fs=100`, `pdamp_fs=500`) on `SrTiO3`
10x10x10 (5,000 atoms, rattled by 0.02 A, at rest), once with `capacity="whole"` and once with
`lm.config.CapacityConfig(mode="tiled", receivers=2000)`, which ran tiled. The volume after 50, 100, 150 and 200
steps was 61,754.67, 60,699.69, 60,476.93 and 61,489.99 A^3 in both runs, as with 0.0.2 (relative difference at most
5.3e-9), and after 200 steps the positions differ by at most 3.5e-7 A.

## Not checked

- The wheel on any GPU but the RTX 5060 Ti. Its code for compute capability 8.0, 8.6, 8.9, 9.0 and 10.0 was
  compiled from the same sources and not run from the wheel. The tuned defaults on 8.9, 8.6 and 8.0 were measured
  and gated on those GPUs with builds of the branch (`OPTIMISATION_LEDGER.md`), and the two tiers above ran on 8.9.
- The LAMMPS tests (`LOUPMD_LMP`) and the Rocky Linux 8 container without a GPU, which 0.0.2's record has.
- The acceptance tier, and the gate from the wheel for the labels not named above.
- `loupmd build-kernels` from the wheel.
- Timings: the desktop card serves correctness checks only. The step times in `CHANGELOG.md` are the branch's
  measurements.
