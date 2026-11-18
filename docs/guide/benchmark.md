# Benchmark on your machine

`loupmd bench` times MD steps with a model on fixed systems and prints ms/step, us per atom-step and ns/day. The
GPU must be idle.

```bash
loupmd bench --model omat0-medium --quick
loupmd bench --model omat0-medium --json bench.json
```

## What is timed

Each system runs Langevin dynamics in the engine's own loop: 300 K, 1 fs, friction 0.01/fs, a neighbour-list skin
of 0.5 A, starting velocities from a Maxwell-Boltzmann draw with seed 42. After the warm-up steps, the steps run in
blocks, and the result is the median of the blocks' ms/step.

| Protocol | Warm-up steps | Blocks | Steps per block | Use |
|---|---|---|---|---|
| `standard` | 100 | 6 | 256 | the default; the numbers of the shipped references |
| `fastmace` | 300 | 4 | 300 | comparisons with FastMACE's published tables |
| `quick` | 50 | 2 | 128 | one short look |

The bundle is exported and loaded before the timing starts, and no output is written. The tuning of a new system
(GEMM plans, kernel variants) happens before the timed blocks. The timing is the engine alone; an `lm.md` run adds
its output and uses its own timestep (0.5 fs by default) and an automatic skin.

## Suites and systems

| Suite | Systems | Atoms |
|---|---|---|
| `standard` (default) | `NbCl5 3x3x3`, `SrTiO3 10x10x10` | 1,944; 5,000 |
| `quick` (`--quick`) | `SrTiO3 6x6x6` | 1,080 |
| `families` | one system for the model's family: `SrTiO3 10x10x10` (MP, OMAT, MatPES, MH-1), `NaCl 3x3x3` (POLAR-1), `water 216` (OFF, OMOL-0) | 5,000; 216; 648 |

`--system NAME` (repeatable) times named systems instead: `NbCl5 3x3x3`, `SrTiO3 10x10x10`, `SrTiO3 6x6x6`,
`NaCl 3x3x3`, `water 216` and `water 216 2x2x2` (5,184 atoms). `--protocol` overrides the suite's protocol. The
`capacity` suite is refused in this version.

SrTiO3 and NaCl are generated in code, and the water box ships with the package. `NbCl5_R2SCAN.cif` does not ship
yet: put it in a folder named by `LOUPMD_BENCH_STRUCTURES` (several folders are separated as in `PATH`), or run a
suite that needs no file.

## Reading the table

| Column | Meaning |
|---|---|
| `ms/step` | median over the blocks |
| `us/atom-step` | ms/step per atom, in microseconds |
| `ns/day` | simulated time per day at the protocol's timestep |
| `spread` | (largest - smallest block) / median |
| `T / K` | the temperature at the end; it should be near 300 K |
| `dangerous` | always 0: the engine checks the neighbour list at every step and rebuilds it before a force call could use a stale one. The column is kept for the record's format |
| `status` | `ok`, or `UNSTABLE` when the temperature is not finite |

An `UNSTABLE` row is not a valid timing, and the command exits with code 1.

## Busy GPUs

`loupmd bench` refuses to time when another process is on the GPU, or when nvidia-smi reports more than 20%
utilisation before the run (under WSL nvidia-smi lists no processes). `--force` times anyway; the record and a
warning then say what was there.

## Comparing with the shipped references

The package ships `loupmd/bench/reference.json`: timings measured with `loupmd bench` itself, each naming the GPU,
the model file's SHA-256, the system, the protocol, the library build and the day. When a row matches an entry on all
four of GPU name, model hash, system and protocol, the table adds `reference` and `ratio` (measured / reference) and
names the entry. The shipped entries are for the NVIDIA GeForce RTX 5060 Ti:

| Model | System | Protocol | ms/step |
|---|---|---|---|
| `omat0-medium` | NbCl5 3x3x3 | standard | 1.720 |
| `omat0-medium` | SrTiO3 10x10x10 | standard | 7.725 |
| `omat0-medium` | SrTiO3 6x6x6 | quick | 1.704 |
| `omat0-small` | SrTiO3 10x10x10 | standard | 4.276 |

A reference is a figure to compare with, not a promise. Another GPU model has no reference.

`--compare mace` also times mace-torch's ASE calculator (float32, on the GPU) under ASE's Langevin on the same
structure: 10 warm-up steps, then the median of 3 blocks of 20 steps. It needs the export extra. The printed ratio
is mace-torch's step time over LoupMD's. A comparison that fails is recorded with its error.

## The record

`--json FILE` (or `-` for standard output) writes the record, schema `loupmd.bench/1`: the LoupMD and library
versions, the GPU's name, compute capability and driver, the model's label, file, SHA-256 and head, the protocol,
one row per system with every block's time, and the GPU processes and utilisation seen before the run.

## A/B comparisons

Two settings differ by a few percent at most, which is the size of the noise between runs. `--ab` runs the same
benchmark once per side, each in a fresh process, in the order a b b a, so drift and order count equally on both
sides. A side is a list of environment variables, `ENV=VAL[,ENV=VAL...]`, or `-` for the environment as it is:

```bash
loupmd bench --model omat0-medium --ab - LOUP_NO_HYBRID=1 --json ab.json
```

It prints every run, the mean ms/step per side and system, and each mean's ratio to the first side (schema
`loupmd.bench-ab/1`).

- Keep the GPU idle and the system the same for all runs.
- The engine stores its GEMM plans per GPU and library build; the plan chosen can move small cells by a few
  percent. Give each side its own plan folder to measure that noise, for example
  `--ab LOUP_CACHE_DIR=/tmp/plans_a LOUP_CACHE_DIR=/tmp/plans_b,LOUP_NO_HYBRID=1`.
- Check the final temperatures: a side that did not stay near 300 K timed different dynamics.
- Call a difference real only when it holds in both orders.
