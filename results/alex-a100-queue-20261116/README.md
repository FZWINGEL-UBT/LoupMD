# A100 validation queue, 16 November 2026

Nineteen single-GPU jobs on NHR Alex (A100-SXM4-40GB, partition `a100`, 1 hour each). Together they validate on
compute capability 8.0 every LoupMD change measured so far only on the RTX 5060 Ti (S81-S106). Seven were queued
earlier in the day (`s83t` to `s100c`); the twelve `s106*` jobs were added at 18:30. When this file was written,
all nineteen were PENDING (reason `Priority`), and no job had started.

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
