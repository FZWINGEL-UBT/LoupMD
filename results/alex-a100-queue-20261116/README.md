# A100 validation queue, 16 November 2026

Nineteen single-GPU jobs on NHR Alex (A100-SXM4-40GB, partition `a100`, 1 hour each). Together they validate on
compute capability 8.0 every LoupMD change measured so far only on the RTX 5060 Ti (S81-S106). Seven were queued
earlier in the day (`s83t` to `s100c`); the twelve `s106*` jobs were added at 18:30. When this file was written,
all nineteen were PENDING (reason `Priority`), and no job had started.
On 17 November at 08:18 CET six jobs had finished and thirteen were still pending; see [Results](#results-17-november-2026).

## Checking

```bash
ssh alex 'bash /home/atuin/v130cb/v130cb18/loup-s64/alex64/check_queue.sh'
```

`check_queue.sh` (a copy is next to this file) reads `queue.txt` and prints two things:

- The Slurm state of every job, from `sacct`.
- For each job log that exists, its key lines:
  - **`ab_libs.py` library A/B:** the `MEAN` line. `b/a` is the step-time ratio of side b to side a, so below 1
    means b is faster. The two values in brackets are the two run orders. A change counts as faster only when
    both are below 1. Locally, the same library on both sides differed by up to 0.5%, so smaller ratios are ties.
  - **`ab_sides.py` in-process A/B:** the `ratio to side 0` lines, with the forward and reverse order.
  - **Gates:** one summary line per gate run, in the form `rows N, failing F, worst force X, worst stress Y`.
    Failing rows are printed in full. The 3x rule fails a row at a ratio of 3.0.
  - **Errors:** error lines and the `#####` end markers. A time-out shows as
    `CANCELLED ... DUE TO TIME LIMIT`.

The NHR jump host refuses connections made in quick succession, so use one `ssh` call per check. If a call is
refused, wait about 30 s and try again.

To read one full log:

```bash
ssh alex 'cat /home/atuin/v130cb/v130cb18/loup-s64/runs/slurm-loup-<name>-<jobid>.out'
```

The working directory is `W=/home/atuin/v130cb/v130cb18/loup-s64`. Under it:

| Path | Contents |
| --- | --- |
| `$W/alex64/<name>.sh` | Job scripts (copies in `scripts/` next to this file) |
| `$W/runs/slurm-loup-<name>-<jobid>.out` | Job logs |
| `$W/src/libloupmd_<tag>.so` | Libraries |
| `$W/pynew` | Python wrapper and test scripts |
| `$W/gate_refs` | Gate reference cache (`LOUP_GATE_CACHE`) |

`pynew` matches `python/` at `a0d04cd6`: `test_gate.py` and `gen_generic.py` were synced at 18:30.

Every job starts with a fresh GEMM plan cache. `--same-plans` A/Bs share one tuned plan set between the two sides.

## Libraries

| Tag | sha256 (first 16) | Sources |
| --- | --- | --- |
| `s81` | `6962b4bdc72a2d06` | S78 tree with the cc 8.0 defaults and the lane-group neighbour kernels |
| `s84` | `8d839b6f5d489f69` | `s81` plus the POLAR dot fold and the own-SGEMM extensions (active on cc 8.0 for long-range models, which load no tensor-core cubin) |
| `s86` | `f8304b7c5ea65695` | `round6-hardening` `9f8cac77`: S85 / S86 POLAR launch trims |
| `s93a` | `51e7b9af55159773` | Scratch build, not committed: `s86` with `first_bwd_edge_kernel<COLS = 2>` capped at 6 blocks / SM on `sm_80` (80 registers) |
| `s93b` | `9e4867439a454cf4` | Scratch build, not committed: `s93a` with `gconv_fwd_L1` R1 at C = 128 capped at 5 blocks / SM on `sm_80` (96 registers) |
| `s100` | `93d7a3481640d822` | `round6-hardening` `87019ef5` (S100); later commits up to `a0d04cd6` change documentation and S101 only |
| `s106` | `8738d051f47b2d59` | The sources of `round6-hardening` `82b11207`. Built from `a0d04cd6` plus `$W/bsrc/src/s106_worktree.patch` (sha256 `86617c84e8820c97`) before that commit: S105 (`first_fwd_kernel` without its density-only warp) and S106 (Langevin first half with one thread per normal draw) |

Every library was built on the Alex login node with CUDA 13.2 for `sm_80`: `make CUDA_ARCH=sm_80` in `$W/bsrc/src`.

## Jobs

| Job ID | Name | Libraries | What it measures | Decision it informs |
| --- | --- | --- | --- | --- |
| 5012939 | `s83t` | `s81` | In-process A/B (`ab_sides.py`, 3 rounds) of the generic conv-forward variant tuner `LOUP_CONV_TUNE=1` (default on `sm_120` only). Systems: OMAT-0 medium generic SrTiO3, MP-0 medium SrTiO3, OFF24-M water (5,184 atoms), OFF23-M water216. Also prints the per-variant times. | Faster in both orders on these families: enable `GenericModel::tune_conv_variants` on cc 8.0 (`device_is_cc120()` check, `src/generic.cu`), then gate. Otherwise keep it on `sm_120` only. |
| 5012942 | `s84c` | `s81` -> `s84` | Library ABBA, 2 rounds: POLAR-1-S / M / L NaCl216 (fresh plans); OFF23-S water216 on shared plans as the no-change control. | POLAR not slower in both orders: keep S82-S84 on cc 8.0. Otherwise find which of the dot fold and the own-SGEMM extensions costs time, and gate it off on cc 8.0. |
| 5014323 | `s86c` | `s84` -> `s86` | Library ABBA on shared plans, 3 rounds: POLAR-1-S / M / L NaCl216 (S85 / S86 launch trims). | Not slower: no action. These changes are already default on every GPU. |
| 5014324 | `s86g` | `s86` | 3x gates: POLAR-1-S / M / L crystals. | Any failing row: bisect S82-S86 on the A100 before relying on POLAR there. |
| 5014425 | `s93c` | `s86` -> `s93a`, `s86` -> `s93b` | Library ABBA on shared plans, 2 rounds: OMAT-0 medium generic SrTiO3, MP-0 medium SrTiO3, OFF23-M water216. | If `s93b` (or `s93a`) is faster in both orders on all three, add its `sm_80` register caps to the sources (`LOUP_FIRST_BWD_EDGE_MINB2` in `src/generic.cu`; generator minimum blocks for `gconv_fwd_L1` R1) and gate them. Otherwise drop. |
| 5014761 | `s96c` | `s86` | OMAT-0 medium specialised path, separate processes, 3 rounds: receiver-owned `w'` (`LOUP_WPRIME_RCV=1`) against the default (edge-tiled below 512 atoms off `sm_120`). Systems: SrTiO3 3x3x3 (135 atoms), 4x4x4 (320), 2x2x2 (40). | Receiver-owned faster in both orders: extend `LoupHandle::use_wprime_rcv()` (`src/engine.cu`, now `is_sm120 \|\| N >= 512`) to cc 8.0. |
| 5015963 | `s100c` | `s86` -> `s100` | Library ABBA on shared plans, 2 rounds: OMAT-0 medium generic SrTiO3, MP-0 SrTiO3, OFF23-M water216. These cover S96 (`sm_120` only) and S97-S100 (generic pass glue on every GPU). Also the OMAT-0 path comparison at `s100` (SrTiO3, NbCl5) and 3x gates: OMAT-0 generic oxide, OFF23-S organic. | S97-S100 must not be slower on the A100. If they are, restrict them to `sm_120`. Record the result in the ledger (`gen-pass-glue`) and STATUS. |
| 5017365 | `s106c` | `s100` -> `s106` | Library ABBA on shared plans, 2 rounds: OMAT-0 medium generic SrTiO3, MP-0 SrTiO3, OFF23-M water216, OFF23-S water216, MH-1 NaCl216, POLAR-1-S NaCl216. | S105 / S106 must not be slower on the A100. S101 does not change `sm_80` code. |
| 5017366 | `s106p` | `s106` | OMAT-0 medium path comparison in separate processes, 2 rounds: specialised (side a) against generic (side b). Systems: SrTiO3 5,000 / 1,080 / 320 atoms, NbCl5. | Input for the owner's choice of the default OMAT-0 medium path on the A100. On the 5060 Ti, generic leads at every size (0.964-0.993 at S100). |
| 5017367 | `s106g1` | `s106` | 3x gates: OMAT-0 medium generic oxide, OMAT-0 medium specialised oxide, MP-0 medium oxide. | See the gate sweep note below. |
| 5017368 | `s106g2` | `s106` | MPtrj oxide, OMAT-0 small oxide, MPA-0 medium oxide. | As above. |
| 5017369 | `s106g3` | `s106` | MP-0b3 medium oxide, MatPES-PBE and MatPES-r2SCAN OMAT fine-tunes crystals. | As above. |
| 5017370 | `s106g4` | `s106` | MH-1 crystals (`omat_pbe` head), MH-1 `matpes_r2scan` crystals, MH-1 `omol` organic. | As above. |
| 5017371 | `s106g5` | `s106` | OFF23-M, OFF23-S, OFF24-M organic. | As above. |
| 5017372 | `s106g6` | `s106` | OFF23-L, OFF23b-M organic. | As above. |
| 5017373 | `s106g7` | `s106` | OMOL-0 XL organic, neutral and charge +1 doublet. | As above. |
| 5017374 | `s106g8` | `s106` | POLAR-1-S, POLAR-1-M crystals. | As above. |
| 5017375 | `s106g9` | `s106` | POLAR-1-L crystals. | As above. |
| 5017376 | `s106g10` | `s106` | `LOUP_CHAIN_FOLD=0`: OMAT-0 medium generic oxide, MH-1 crystals. | As above. |

The gate sweep (`s106g1`-`s106g10`) gates every family at the cc 8.0 defaults. S97-S100, S105 and S106 are active
on every GPU, so they count as validated on cc 8.0 only once every row passes. A failing family should be re-gated
with `s100`, then `s86`, to bisect.

## Results (17 November 2026)

The first six jobs ran between 04:51 and 06:04 CET on node `a0701` (A100-SXM4-40GB, one GPU each). Every one
completed with exit code 0. `s84c`, `s86c`, `s86g`, `s93c` and `s96c` overlapped there on different GPUs. Their logs
are in `logs/`, and the numbers are recorded in `OPTIMISATION_LEDGER.md` under the entries named below.

In the A/B rows, the two values in brackets are the two run orders.

| Job | Result | Decision (from the table above) | Ledger entry |
| --- | --- | --- | --- |
| `s83t` | Tuner against the default, with a second default side as noise: OMAT-0 generic SrTiO3 `1.003` (noise `1.002`), MP-0 medium SrTiO3 `0.998` (`0.998`), OFF24-M water `0.996` (`0.997`), OFF23-M water216 `1.024` (`1.013`). R5 was the slowest variant on every system. | Not faster on these families: the tuner stays on `sm_120` only. | `gen-conv-fwd-variants` |
| `s84c` | POLAR-1-S / M / L NaCl216 `0.795` (`0.796` / `0.794`), `0.802` (`0.809` / `0.795`), `0.814` (`0.814` / `0.814`). Control OFF23-S water216 `1.001` (`1.018` / `0.983`). | Faster in both orders: S82-S84 stay on cc 8.0. | `pol-dot-fold`, `a100-own-ext-nb-groups` |
| `s86c` | POLAR-1-S `0.989` (`0.988` / `0.989`), M `0.988` (`0.979` / `0.997`), L `1.011` (`1.018` / `1.005`). | S and M faster. L is slower in both orders, so the "not slower" condition fails for L. Kept on by the owner's decision (17 November). | `pol-bias-ln-fuse` |
| `s86g` | 29 rows, none failing. Worst force / stress: POLAR-1-M 1.29 / 1.49 (18 rows), L 1.32 / 0.80 (6), S 0.47 / 0.72 (5; NaNbO3_c0 left out as unstable). | No failing row: no bisect needed. | `pol-bias-ln-fuse` |
| `s93c` | `s93b`: OMAT-0 generic `0.986` (`0.987` / `0.985`), MP-0 medium `0.984` (`0.984` / `0.984`), OFF23-M water216 `1.015` (`1.023` / `1.007`). `s93a`: `1.000`, `0.998`, `1.003`. | Not faster on all three: dropped. | `s93-sm80-caps` |
| `s96c` | Receiver-owned `w'` against the default: 40 / 135 / 320 atoms `0.994` (`0.996` / `0.992`), `0.989` (`0.988` / `0.989`), `0.968` (`0.969` / `0.967`). | Faster in both orders at every size, so the criterion for extending `use_wprime_rcv()` to cc 8.0 is met. Done in S114 (owner's decision, 17 November). | `wp-receiver-owned` |

Still pending at 08:18 CET (reason `Priority`): `s100c` and the twelve `s106*` jobs. Slurm's estimated starts were
18 November 04:55 (`s100c`) to 08:02 CET; `s106g8`-`s106g10` had no estimate yet.

## Round of 18 November (S115-S128)

### The S106 sweep's two failing stress rows

All nineteen jobs above finished (log: the owner's `alex-a100-jobs-20261118.txt`). Every gate passed except two OMAT-0 medium
stress rows on `SrTiO3_3x3x3_c1` (limit 3.00): the specialised path with `wprime_rcv=1, wprime_threads=512` (3.07)
and the generic path with `LOUP_CHAIN_FOLD=0` (3.38 / 3.14 / 3.01 across the three conv choices). The folded generic
sweep passed (worst stress 2.92). The absolute stress error there is about 3.4e-7 eV/A^3, and the ratio divides by
MACE float32's own error (`max(dS32, 1e-7 / 3)`), so a row near the floor moves with the atomic accumulation order.
Since S114, `wprime_rcv=1` is the cc 8.0 default, so the first row is a default configuration. `q121r_s106` and
`q121r_s121` repeat both gates three times per library to measure that spread before any change is considered.

### Libraries (built on the Alex login node, CUDA 13.2, `sm_80`, `alex_build.sh` in `scripts/`)

| Tag | sha256 (first 16) | Sources | Python |
| --- | --- | --- | --- |
| `s121` | `41795c411f9b159d` | `round6-hardening` `990b261` (S115-S121; S119 / S121 compile their per-path forms on `sm_80`) | `$W/py121` |
| `s123` | `c72262b5bc907931` | `990b261` + S122-S124 (the sources of `5fb1099` and the S122 part of `f799b54`) | `$W/pys123` |
| `s123x80` | `0f55589e94b60333` | `s123` with every `sm_120`-only block of `src/generated_generic.cu` also compiled for `sm_80` (test build: all 72 `__CUDA_ARCH__ == 1200` conditions widened, i.e. the Cartesian forms of S119 / S121 / S122, S48's x-gradient rewrite and the S40 / S101 RED edge-vector updates; the same set the sm_86 / sm_89 test builds measured) | `$W/pys123x80` |

S125, S127 and S128 (`f799b54`) are not in these libraries; a later round builds them.

### Jobs (submitted 18 November about 20:55 CET; held by a maintenance reservation, `ReqNodeNotAvail`)

| Job IDs | Names | Libraries | What it measures | Decision it informs |
| --- | --- | --- | --- | --- |
| 5030343-44 | `q121c1`, `q121c2` | `s106` -> `s121` | Library ABBA on shared plans, 2 rounds: POLAR-1-S / M / L NaCl216, POLAR-1-S NbCl5 (S115, S118); OFF23-S / M / L water216, MP-0 and MPtrj SrTiO3 (S116), OMAT-0 generic SrTiO3 (control) | S115-S121 not slower on cc 8.0 |
| 5030345-55 | `q121g1`-`q121g11` | `s121` | 3x gates of every family at the cc 8.0 defaults; `g10`: `LOUP_CHAIN_FOLD=0`; `g11`: `LOUP_POLAR_FOLD=0` (POLAR-1-S / M / L) | Every row passes: S115-S121 validated on cc 8.0 |
| 5030356-57 | `q121r_s106`, `q121r_s121` | `s106`, `s121` | OMAT-0 medium specialised oxide gate and generic oxide gate with `LOUP_CHAIN_FOLD=0`, three times each | The spread of the two failing rows above |
| 5030381 | `q123c1` | `s121` -> `s123` | Library ABBA: POLAR-1-S / L NaCl216, POLAR-1-S NbCl5 (S123 / S124), POLAR-1-M NaCl216 (control) | S123 / S124 not slower on cc 8.0 |
| 5030382 | `q123c2` | `s123` -> `s123x80` | Library ABBA: OFF23-L / OFF23-M water216, OFF24-M water, MP-0 and OMAT-0 generic SrTiO3, POLAR-1-L NaCl216, OMOL-0 XL water216 | That set faster on cc 8.0 (sm_86: OFF23-L `0.938`, POLAR-1-L `0.985`, OMAT-0 generic `0.991`). Since S129 (19 November) it is the default on every GPU, so this run confirms it on the A100; a regression there would exclude cc 8.0 again |
| 5030383-85 | `q123g1`-`q123g3` | `s123x80`, `s123` | 3x gates: the x-degree-1 and x-degree-2 Cartesian families on `s123x80`; POLAR-1-S / L on `s123` on both fold paths | Accuracy of the above on cc 8.0 |

## Local gate reference (5060 Ti)

Worst force / stress ratio of the S100 library's full sweep (ledger `gen-pass-glue`):

| Family | Worst force / stress |
| --- | --- |
| OMAT-0 medium specialised | 1.67 / 2.50 |
| OMAT-0 medium generic | 2.02 / 1.80 |
| MP-0 medium | 0.83 / 0.48 |
| MPtrj | 1.37 / 0.74 |
| MH-1, `omat_pbe` / `matpes_r2scan` / `omol` heads | 0.85 / 0.31, 0.86 / 0.35, 0.33 / 0.42 |
| OFF24-M | 0.70 / 0.55 |
| OFF23-M | 0.34 / 0.58 |
| OFF23b-M | 0.98 / 1.21 |
| OFF23-S | 0.64 / 1.74 |
| OFF23-L | 0.42 / 1.19 |
| OMAT-0 small | 1.21 / 1.25 |
| MPA-0 medium | 2.02 / 1.84 |
| OMOL-0 | 0.79 / 0.87 |
| POLAR-1-S / M / L | 0.69 / 0.64, 2.17 / 1.20, 1.85 / 0.71 |
| `LOUP_CHAIN_FOLD=0`: OMAT-0 generic / MH-1 | 2.01 / 1.74, 0.87 / 0.29 |

Earlier A100 runs (`s83g`, library `s81`) gave the following, all passing:

- OMAT-0 medium specialised 1.70 / 2.92.
- OMAT-0 medium generic 2.01 / 2.82.
- MP-0 medium 0.70 / 1.41.

POLAR-1-M's worst row varies from run to run: 2.17 to 2.71 on the 5060 Ti across identical builds, because
atomic accumulation order changes.

## Risks

- **Time-outs:** gate runs took 5 to 30 minutes each on the A100 (longer when several jobs share a node).
  `s81g2` timed out after two OMOL-0 gates. A gate job that ends with `DUE TO TIME LIMIT` needs its remaining gates
  resubmitted: copy the script, delete the finished `g ...` lines and `sbatch` it.
- **A/B noise:** the A/B jobs share nodes with other users' work. Treat a ratio within 0.5% of 1, or with mixed
  orders, as a tie.
