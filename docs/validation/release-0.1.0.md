# Release verification: 0.1.0

The final checks of `PLAN.md` S7-T06 on the release branch. Everything here was measured on one machine: an
NVIDIA GeForce RTX 5060 Ti (compute capability 12.0, 16 GB), driver 621.14, in WSL 2 (Ubuntu) on Windows 11. No
other GPU was used; the other GPUs of S7-T05 wait for the owner.

- Commit: `6d06428` (the tip when the checks ran; later commits change only `PLAN.md` and this page).
- Library: `libloupmd_d3n.so`, SHA-256 `7458a4c8045c3e24...`, built for `sm_120` from the same C++/CUDA sources
  as the tip (no file under `src/`, `include/` or `codegen/` changed after it was built).

**These results describe `release/v1` before the merge of `round6-hardening`** (`PLAN.md` O30: the release waits
for that merge, and the checks run again on the merged code).

## Summary

| Check | Command | Result |
| --- | --- | --- |
| CPU suite | `pytest -q tests/cpu` | 489 passed, 1 skipped |
| GPU, fast and full tiers | `pytest -q tests/gpu -m "not acceptance"` | 353 passed, 1 failed (`test_sweeps.py` MH-1, see below) |
| GPU, acceptance tier (gate per family, examples) | `pytest -q tests/gpu -m acceptance` | 16 passed (10 gates, 6 examples), 171 s |
| Gate per label, whole and tiled | `loupmd validate --model LABEL --set SET --capacity whole/tiled` | 37 passed: all 20 labels whole, the 17 without a long-range term tiled |
| A/B against the Stage 6 exit | `loupmd bench --model omat0-medium`, A B B A | tie: tip / `release-stage6` 1.0015 (`NbCl5` 3x3x3), 1.0001 (`SrTiO3` 10x10x10) |
| Clean install of the built wheel | `tools/release/make_release.sh`, then `pip install` in a fresh environment | **not run**: deferred to upload time (`PLAN.md` D82) |

## Gate per label

`loupmd validate --model LABEL --set SET` (the 3x float32-against-float64 rule of the accuracy gate) for every
label of `docs/reference/models.md`, once whole (`--capacity whole`) and once through the tiled call
(`--capacity tiled`). The `oxide` and `crystals` sets read their structures from the folder that
`LOUPMD_BENCH_STRUCTURES` names; the large models compute their float64 reference on the CPU
(`--ref-device cpu`). Every run passed:

| Set | Labels | Whole | Tiled |
| --- | --- | --- | --- |
| `oxide` | `mp0-medium`, `mp0-large`, `mp0b3-medium`, `mpa0-medium`, `omat0-medium`, `omat0-small`, `matpes-pbe`, `matpes-r2scan` | 8 passed | 8 passed |
| `crystals` | `mh1-omat`, `mh1-mp`, `mh1-r2scan` | 3 passed | 3 passed |
| `crystals` | `polar1-s`, `polar1-m`, `polar1-l` (CPU reference) | 3 passed | not applicable |
| `organic` | `omol0-xl`, `off23-small`, `off23-medium`, `off23-large`, `off23b-medium`, `off24-medium` (the last three with the CPU reference) | 6 passed | 6 passed |

The `MACE-POLAR-1` models have no tiled mode: their long-range term is a sum over the whole cell's reciprocal
space, which tiles do not split (`capacity="tiled"` refuses them; `PLAN.md` §11 has the post-release item). FastMACE
has no tiled mode for them either.

## A/B against the Stage 6 exit

The tag `v0.1.0a1` that S7-T06 names was never set (`PLAN.md` D83); the last stage tag is `release-stage6`
(`8a0ec4e`). No C++/CUDA source changed after it, so both sides run the same library and the A/B measures the Python
layer. `loupmd bench --model omat0-medium --suite standard` from a `git archive` of each tree, eight runs in the
order a b b a a b b a, each with a fresh GEMM plan cache (ms/step):

| System | `release-stage6` (a) | tip `6d06428` (b) | median b / a |
| --- | --- | --- | --- |
| `NbCl5` 3x3x3 (1,944 atoms) | 1.907, 1.897, 1.868, 1.883 | 1.892, 1.894, 1.888, 1.897 | 1.0015 |
| `SrTiO3` 10x10x10 (5,000 atoms) | 8.213, 8.223, 8.213, 8.216 | 8.211, 8.194, 8.220, 8.278 | 1.0001 |

A tie: the sides overlap on both systems.

## The wheel and the clean install

Not run. Docker Desktop, which hosts the `manylinux_2_28` build image, does not start on the build machine (stale
socket files in its run folder; `PLAN.md` D82). Nothing before the upload needs the wheel: the tests above run from
the source tree on a library built for the test GPU. The release build (`tools/release/make_release.sh`) and the
clean install of its wheel in an environment without torch run when the owner decides to upload (S7-T08).

## The failed full-tier test

`tests/gpu/test_sweeps.py::test_sweep_schedule_reproduces_the_ordinary_pass[mace-mh-1]` compares the per-layer
sweep replay (`LOUP_SWEEP_REPLAY=1`, a development path) with the ordinary pass on a strained, rattled 135-atom `SrTiO3` cell. Energy:
the replay differed by 1.054e-7 eV/atom against the test's floor of 1e-7, which the test sets at what two engines
with different GEMM plans differ by. Forces and stress were within their bounds. The same test has failed before
only inside full runs (`PLAN.md`, Stage 1's progress note). No C++/CUDA file changed since the Stage 6 exit.

Ten separate runs of that test alone, five with the suite's shared GEMM plan cache and five each with a fresh one:
10 passed. The failure is the margin of the test's floor against plan-to-plan float32 differences, not a fault of
the replay; the replay is a development path that no user setting reaches.
