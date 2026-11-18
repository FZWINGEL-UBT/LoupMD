# Release verification: 0.1.0

The final checks of `PLAN.md` S7-T06 on the release branch, run again on the code after the merge of
`round6-hardening` (`PLAN.md` O30). Everything here was measured on one machine: an NVIDIA GeForce RTX 5060 Ti
(compute capability 12.0, 16 GB) in WSL 2 (Ubuntu) on Windows 11. No other GPU was used; the other GPUs of S7-T05
wait for the owner.

- Date: 18 November 2026.
- Code: `release/v1` with the merge `09d6daa`, the shared layer arena `2f98c9a` and the memory plan of `0171c3e`
  and the commit that carries this page (it adds the long-range models' buffers to the plan's count; no kernel
  changes).
- Library: `libloupmd.so`, SHA-256 `8c82bc7e3ee9762a...`, built for `sm_120` from that code; it reports
  `0.1.0a1+d1283291ef91a017`.

The results of 16 November, on the branch before the merge, are in this file's history (`6d06428`).

## Summary

| Check | Command | Result |
| --- | --- | --- |
| CPU suite | `pytest -q tests/cpu` | 490 passed, 5 skipped |
| GPU, fast tier | `pytest -q tests/gpu -m fast` | 360 passed, 7 skipped (the NbCl5 structure file is not in the folder) |
| GPU, full tier | `pytest -q tests/gpu -m full` | 48 passed |
| GPU, acceptance tier (gate per family, examples) | `pytest -q tests/gpu -m acceptance` | 15 passed, 1 skipped (the same file) |
| Gate per label, whole and tiled | `loupmd validate --model LABEL --set SET --capacity whole/tiled` | 37 passed: all 20 labels whole, the 17 without a long-range term tiled |
| A/B against the branch before the merge | `loupmd bench --model omat0-medium --suite standard --ab`, a b b a twice | faster: 0.927 (`NbCl5` 3x3x3), 0.961 (`SrTiO3` 10x10x10) |
| Memory plan against measurement, capacity | `docs/validation/capacity.md`, `STATUS_AND_PROFILE.md` §6.35 | plan within 0.3-5% of the device peak on 22 tiled plans; 10,985,000 atoms run with OMAT-0 medium, 20,480,000 with OMAT-0 small |
| Clean install of the built wheel | `tools/release/make_release.sh`, then `pip install` in a fresh environment | **not run**: deferred to upload time (`PLAN.md` D82) |

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

## The wheel and the clean install

Not run. Docker Desktop, which hosts the `manylinux_2_28` build image, does not start on the build machine (stale
socket files in its run folder; `PLAN.md` D82). Nothing before the upload needs the wheel: the tests above run from
the source tree on a library built for the test GPU. The release build (`tools/release/make_release.sh`) and the
clean install of its wheel in an environment without torch run when the owner decides to upload (S7-T08).

## Tests whose premise the merge changed

The 16 November run had one failure in the full tier, the sweep replay against the ordinary pass on MH-1 at
1.054e-7 eV/atom against a floor of 1e-7. Since the merge the ordinary whole-system pass folds the skip chain and the
sweeps do not, which moves forces by about 2e-6 eV/A and stress by up to 2e-8 eV/A^3 (`STATUS_AND_PROFILE.md`
§6.34); the tests that compare a tiled, domain or sweep result with a whole-system one at rounding-level bounds now
take the whole-system reference without the fold (`tests/gpu/gpu_support.py`, `unfolded`). No test failed in the
runs above.
