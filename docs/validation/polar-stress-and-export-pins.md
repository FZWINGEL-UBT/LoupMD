# MACE-POLAR-1 stress against MACE, and the export extra's versions

18 November 2026, evening. Everything here was measured on one machine: an NVIDIA GeForce RTX 5060 Ti (compute
capability 12.0, 16 GB) in WSL 2 on Windows 11.

## The question

A user compared the stress LoupMD 0.0.1 gives for a MACE-POLAR-1 model with MACE's own, from mace-torch 0.3.16 with
cuequivariance (the versions the export extra of 0.0.1 installs), on a water cell of 5,184 atoms, and found a
difference.

## The structure

`water_5184atoms.extxyz` (the user's file, not in the repository): 1,728 rigid water molecules in a cubic cell of
37.5487 A, 0.976 g/cm3. It is an exact 2x2x2 replication of a box of 648 atoms (the largest distance between the
file and the replicated box is 1e-14 A). MACE with cuequivariance does not fit this GPU at 5,184 atoms (float32
asks for more than 10 GiB), so the MACE side ran on the 648-atom box. LoupMD 0.0.1 gives the same stress on both
cells to the digits printed below.

## Result

`polar1-s` on the 648-atom box, stress in GPa (ASE's convention), LoupMD 0.0.1 as installed from PyPI:

| | xx | yy | zz | yz | xz | xy |
| --- | --- | --- | --- | --- | --- | --- |
| LoupMD 0.0.1 | -2.42816 | -2.56341 | -2.41164 | +0.01883 | -0.07400 | +0.05263 |
| MACE float64 energy under strain (central differences, h = 1e-5; mace-torch 0.3.16) | -2.42816 | -2.56341 | -2.41164 | +0.01883 | -0.07400 | +0.05263 |
| mace-torch 0.3.17, autograd stress (float32, e3nn backend) | -2.42816 | -2.56341 | -2.41163 | +0.01884 | -0.07401 | +0.05263 |
| mace-torch 0.3.16, autograd stress (float64, cuequivariance) | -2.44971 | -2.57972 | -2.43523 | +0.01632 | -0.07209 | +0.05334 |
| mace-torch 0.3.16, autograd stress (float32, cuequivariance) | -2.44971 | -2.57971 | -2.43523 | +0.01632 | -0.07209 | +0.05334 |

LoupMD's stress is the strain derivative of MACE's energy to 1e-5 GPa on every component (the difference
quotient with h = 1e-4 gives the same `xx` and `xy`). The autograd stress of mace-torch 0.3.16 is 0.016 to 0.024 GPa
off on the diagonal (about 1%) and about 0.002 GPa off the diagonal. mace-torch 0.3.17's agrees with LoupMD.

`polar1-m` on the same box: LoupMD 0.0.1 gives -2.42737, -2.57351, -2.39875, +0.01498, -0.06295, +0.05297 and
mace-torch 0.3.16 (float64, cuequivariance) -2.43629, -2.58065, -2.40237, +0.01196, -0.06059, +0.05123: 0.004 to
0.009 GPa apart on the diagonal. The difference quotient and mace-torch 0.3.17 did not fit under the 9 GiB this
check allowed PyTorch for `polar1-m`; `polar1-l` was not run.

The energies agree: LoupMD -449341.662318 eV, MACE float64 -449341.662339 eV (0.00003 meV/atom apart). MACE in
float32 gives -449343.09 eV: a total energy of this size loses 1.4 eV to single precision there.

## The cause

In mace-torch 0.3.16 the long-range part of `PolarMACE` builds its reciprocal-space sum from the reciprocal cell
and the volume of the input (`data["rcell"]`, `data["volume"]`), not from the strained cell, so the derivative
with respect to the strain misses the cell dependence of the k-space terms. mace-torch 0.3.17 derives both from
the strained cell (`mace/modules/extensions.py`). LoupMD's engine has the term, and its accuracy gate has always
judged the stress of these models against the difference quotient of MACE's float64 energy
(`loupmd/validate/references.py`), not against MACE's autograd stress.

## What 0.0.1 cannot do from a fresh install

MACE's POLAR-1 models are built on the package `graph_longrange`. The export extra of 0.0.1 does not install it,
so in an environment made by `pip install "loupmd[export]"` the export of a `polar1-*` model ends with
`ModuleNotFoundError: No module named 'graph_longrange'`. The runs above put the package on the path by hand
(version 0.4.0, and 0.4.4 from PyPI; both give the same LoupMD stress).

## The export extra after 0.0.1

The owner's decision of the same evening (`PLAN.md` O36): the extra installs mace-torch 0.3.17, and torch 2.15 if
that is possible. Both need Python 3.11 or newer, and torch 2.15.0 pins `cuda-toolkit` 13.2.2 where torch 2.14.1
pins 13.0.3, the version the engine's own dependency named. `pyproject.toml` now has:

- on Python 3.11 and newer: torch 2.15.0, mace-torch 0.3.17, and `cuda-toolkit[cublas,cudart,cufft]==13.2.2` for
  the engine;
- on Python 3.10: torch 2.14.1, mace-torch 0.3.16 and `cuda-toolkit` 13.0.3, as in 0.0.1 (the last versions for
  that Python);
- on both: `graph_longrange==0.4.4`, and cuequivariance 0.12.0 as before.

### What was checked

A test environment on Python 3.12 with torch 2.15.0 (`2.15.0+cu132`), mace-torch 0.3.17, graph_longrange 0.4.4,
cuequivariance 0.12.0 and `cuda-toolkit` 13.2.2 (nvidia-cublas 13.4.1.3), every package from PyPI, `pip check`
clean. LoupMD ran from the tree with its `sm_120` library (the sources of 0.0.1).

| Check | Result |
| --- | --- |
| `pip install --dry-run ".[export]"` from a copy of the tree, Python 3.12 | resolves: torch 2.15.0, mace-torch 0.3.17, graph_longrange 0.4.4, cuda-toolkit 13.2.2 |
| the same, Python 3.10 | resolves: torch 2.14.1, mace-torch 0.3.16, graph_longrange 0.4.4, cuda-toolkit 13.0.3 |
| export of every label of `docs/reference/models.md` (20 labels, `omat0-medium` also for its specialised kernels) with the new versions | all 21 bundles written |
| those 21 bundles against the ones the old versions export (torch 2.14.1, mace-torch 0.3.16; same thread count) | 21 of 21 files identical byte for byte |
| export with the cuequivariance ops package hidden (`omat0-small`, `polar1-s`) | works; the bundle differs from the one with the ops by rounding (at most 5.5e-9 of a tensor's largest value) |
| `loupmd validate`, bundles and float64 references made by the new versions: `omat0-medium` (`oxide`; generic, specialised, tiled), `mh1-omat` and `polar1-s`, `polar1-m` (`crystals`), `off23-small` and `omol0-xl` (`organic`) | all eight passed; worst force / stress ratios 1.96 / 1.71, 1.67 / 1.83, 2.02 / 1.84, 0.83 / 0.45, 1.20 / 1.97, 1.59 / 2.24, 0.65 / 1.81, 0.90 / 0.76 |
| the engine on the CUDA libraries of cuda-toolkit 13.2.2 | the gates above ran on them (their stored GEMM plans carry cuBLASLt 13.4.1 in the file name) |
| `loupmd bench --quick --model omat0-medium` on three CUDA runtimes, five interleaved rounds, same library and bundle | system CUDA 13.3: 1.711 to 1.744 ms/step (mean 1.726); wheels of cuda-toolkit 13.2.2: 1.698 to 1.747 (mean 1.731); wheels of cuda-toolkit 13.0.3: 1.707 to 1.751 (mean 1.732). No difference beyond the scatter of the runs |
| `pytest tests/cpu` | 496 passed, 5 skipped, in the new environment and in the development one (Python 3.10, the old versions) |
| `pytest tests/gpu -m fast` in the new environment | 333 passed, 1 failed, 33 skipped. The failure was a test's own stand-in for a random generator, which numpy 2.5 refuses (`np.full(None, ...)`); with it fixed `test_csvr_gpu.py` gives 16 passed. 26 of the skips were the D3 reference tests, whose reference packages the environment lacked; with them installed `test_d3.py` gives 46 passed, 7 skipped. That makes the development environment's 360 passed, 7 skipped; the tier was not run again in one piece |

The `omol0-xl` gate needed two attempts: the first ran into this check's limit of 800 s per gate while the bundle
was still being exported (its float64 reference alone had taken 11 minutes).

An environment made for 0.0.1 has no `graph_longrange`. A `polar1-*` export there now ends with a message that
names the package and the command to install it (`tests/cpu/test_cache.py`).

### Not checked

- No wheel was built from this change and no version number was set: it is in the tree as "Unreleased".
- `loupmd validate` for the twelve labels not named above, with references from the new versions. Their bundles are
  the same files as before.
- The full and acceptance GPU tiers, and `loupmd build-kernels`, in the new environment.
- An install on Python 3.10 (only pip's resolution was checked; its versions are those of 0.0.1 plus
  graph_longrange 0.4.4, with which the old versions exported `polar1-s` and `polar1-m` above).
- The difference quotient for `polar1-m` and `polar1-l`, and any GPU other than the RTX 5060 Ti.
