# symmetrix main (b907f04) vs LoupMD: force-call timing and 3x gate, MACE-OMAT-0 medium, RTX 5060 Ti

16 November 2026.

## Versions under test

- **symmetrix** `main` at b907f04 (10 November 2026), which is v0.3.0 plus build fixes.
  - Its MACE foundation code is the 1771c84 commit ("add foundation model machinery") plus a JSON `format_version` 3 check.
  - Only the `_mace_foundation` Python module was built (`scripts/sx_build.sh`, no Kokkos).
  - Flags are those of `symmetrix/source/cpp/mace_foundation/CMakeLists.txt` in a Release build:
    - C++20, `-O3 -march=native -fopenmp`;
    - `-ffast-math -fno-finite-math-only` for `cpu.cpp`;
    - nvcc `--expt-relaxed-constexpr --use_fast_math -arch=sm_120`.
  - Module sha256 `c1cd8d1a…`.
- **LoupMD** installed `libloupmd.so` = `libloupmd_s106a.so` (sha256 `9e866c32…`), branch `round6-hardening`.
- **Hardware and software:** RTX 5060 Ti (driver 621.14), CUDA 13.3, mace-torch 0.3.16, torch 2.14.1.

## Model extraction

`scripts/sx_extract.py` runs the steps of `symmetrix_extract_mace`: `extract_mace_data` with `r_min` 0.5, no basis compression, written with `indent=4`. Each structure family gets its own species set.

Spline points:
- **1024:** the default of both the CLI and `extract_mace_data()` in the code.
- **256:** the default that `pair_symmetrix/README.md` states.

| Species (Z) | JSON at 1024 points | JSON at 256 points |
|---|---|---|
| 8, 22, 38 | 849 MB | 226 MB |
| 11, 17 | 431 MB | 119 MB |
| 17, 41 | 431 MB | 119 MB |

## Timing (`scripts/job_speed.sh`, `sx_speed.py`)

**Systems:** SrTiO3 6³, 10³ and 14³, each a snapshot after 500 LoupMD Langevin steps at 300 K.

**One force call** is:
- the library's Python entry point: LoupMD `LoupEngine.evaluate(x, cell, compute_stress=True)`, symmetrix `MACEFoundation.compute(x, Z, cell, pbc)`;
- positions host→device, then energy, forces and virial/stress device→host;
- symmetrix replays its own CUDA graph.

**Protocol:**
- 20 untimed warm-up calls.
- 100 timed calls, each on the snapshot plus fresh 0.003 Å Gaussian noise, so no neighbour-list rebuild and no caching.
- Reported time is the median.
- Every configuration runs in its own process, in one order and then in the reverse order.

| Implementation | 1,080 atoms ms | 5,000 atoms ms | 13,720 atoms ms | Ratio to LoupMD default (1,080 / 5,000 / 13,720) |
|---|---|---|---|---|
| LoupMD default (specialised) | 2.18 | 8.69 | 23.65 | 1.00 |
| LoupMD generic | 1.98 | 8.56 | 24.08 | 0.91 / 0.99 / 1.02 |
| symmetrix float32, 1024 points | 3.11 | 13.91 | 36.82 | 1.42 / 1.60 / 1.56 |
| symmetrix float32-mixed, 1024 points | 2.57 | 11.62 | 31.95 | 1.18 / 1.34 / 1.35 |
| symmetrix float32, 256 points | 3.01 | 13.52 | 36.11 | 1.38 / 1.56 / 1.53 |
| symmetrix float32-mixed, 256 points | 2.61 | 11.81 | 31.86 | 1.19 / 1.36 / 1.35 |

- **Repeatability:** the two passes agree within 0.7 ms (pass 1 / pass 2 per row in `speed_summary.md`).
- **Spline points barely change symmetrix's time:** 1024 and 256 are within 3.5%.
- **Cross-check:** symmetrix forces agree with LoupMD's within 3.5e-5 to 8.0e-5 eV/Å (max abs), and energies within 0.003 meV/atom.
- **LoupMD context, same snapshots:**

  | LoupMD | 1,080 atoms ms | 5,000 atoms ms | 13,720 atoms ms |
  |---|---|---|---|
  | Device-only force pass | 1.8–1.9 | 8.3–8.5 | 23.1–23.3 |
  | MD step (Langevin) | 1.8 | 8.1–8.3 | 22.6–23.0 |

- **Not measured:** symmetrix's MD loop. It has none outside LAMMPS, and the `symmetrix/mace/foundation` LAMMPS pair style was not built.

## 3x float32 gate (`scripts/job_gate.sh`, `sx_gate.py`)

**Set and rule:**
- `python/test_gate.py`'s oxide set: NaCl 2×2×2, SrTiO3 3×3×3 and NbCl5 2×2×2, two strained and rattled configurations each.
- Its cached MACE float64/float32 references.
- Its rule: dE ≤ max(3 dE32, 0.01 meV/atom), dF ≤ max(3 dF32, 1e-5), dS ≤ max(3 dS32, 1e-7).

**How each was run:**
- symmetrix through its ASE calculator (`symmetrix.experimental.MACEFoundation`, skin 1 Å).
- LoupMD through `LoupEngine.evaluate`.

| Candidate | Worst dF / dF32 | Worst dS / dS32 | Worst dE (meV/atom) | 3x gate |
|---|---|---|---|---|
| LoupMD default | 1.66 | 2.49 | 0.00224 | pass |
| LoupMD generic | 2.00 | 1.80 | 0.00189 | pass |
| symmetrix float64, 1024 | 0.03 | 0.02 | 0.00000 | pass |
| symmetrix float32, 1024 | 0.22 | 0.23 | 0.00012 | pass |
| symmetrix float32-mixed, 1024 | 0.75 | 0.85 | 0.00013 | pass |
| symmetrix float64, 256 | 2.08 | 1.30 | 0.00010 | pass |
| symmetrix float32, 256 | 2.12 | 1.63 | 0.00019 | pass |
| symmetrix float32-mixed, 256 | 2.81 | 1.52 | 0.00004 | pass |

Per-configuration rows are in `gate_summary.txt`.

- **At 1024 points, symmetrix's float32 path is roughly 8x closer to MACE float64 than LoupMD on this set:**
  - its worst ratios are 0.22 (force) and 0.23 (stress), against LoupMD's 1.66–2.00 and 1.80–2.49;
  - it is closer than MACE's own float32 (ratio 1);
  - its mixed mode stays below 1.
- **At 256 points the spline sets the error:** float64 already reaches 2.08. The mixed mode then reaches 2.81 of the 3.0 limit.
- **LoupMD's ratios are mostly its own approximations:** S103 showed the 8-byte packed tables account for most of OMAT-0's margin.
- **Not covered:** the gate structures have no pair below about 1.8 Å, so short-range spline error is not tested here.

## Rerun

```bash
bash scripts/sx_build.sh                     # clone ~/src/symmetrix-b907f04, build ~/src/symmetrix-eval/b907f04
# then, under the local-GPU reservation:
bash scripts/job_extract.sh SrTiO3           # ~/src/symmetrix-eval/json
bash scripts/queue_all.sh                    # job_speed.sh, then job_gate.sh (which extracts NaCl and NbCl5)
```

## Files

- `speed.jsonl`: raw timing rows.
- `speed_summary.md`: timing tables.
- `gate_summary.txt`: gate output.
- `logs/`: job output.
- `scripts/`: build, extraction, timing and gate scripts.
