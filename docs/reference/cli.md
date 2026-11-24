# Command-line reference

Generated from the `loupmd` parser by `tools/dev/gen_cli_reference.py`; do not edit by hand. Every command also
prints this with `--help`.

Exit codes: 0 success, 1 a LoupMD error (its message is printed without a traceback unless `--debug` is given),
2 a usage error.

## `loupmd`

Molecular dynamics with MACE potentials on a C++/CUDA engine.

```text
loupmd [-h] [--version] [--debug] COMMAND ...
```

| Argument | Meaning |
| --- | --- |
| `--version` | show program's version number and exit |
| `--debug` | show the traceback of an error |

### `loupmd doctor`

Checks the GPU and driver, the CUDA runtime libraries, libloupmd and its interface version, the tensor-core GEMM kernels for the card, the cache root, the model files' hashes and the export extra. Changes nothing.

```text
loupmd doctor [-h] [--no-hashes]
```

| Argument | Meaning |
| --- | --- |
| `--no-hashes` | do not hash the model files (they can be large) |

### `loupmd models`

The model labels LoupMD knows, with the released file, head, training functional, licence and SHA-256 of each.

```text
loupmd models [-h] [--download LABEL] [--long]
```

| Argument | Meaning |
| --- | --- |
| `--download` `LABEL` | download the label's checkpoint into the cache (checked against the table's SHA-256) |
| `--long` | full SHA-256, URL and the source of each functional |

### `loupmd cache`

LoupMD's cache: downloaded models, exported bundles, GEMM plans, gate references and locally built libraries.

```text
loupmd cache [-h] ACTION ...
```

#### `loupmd cache list`

every cached file with its size and last use

```text
loupmd cache list [-h] [--kind {models,bundles,plans,gate_refs,lib}]
```

| Argument | Meaning |
| --- | --- |
| `--kind` `KIND` | only this kind (repeatable) (one of: models, bundles, plans, gate_refs, lib) |

#### `loupmd cache size`

bytes per kind

```text
loupmd cache size [-h]
```

#### `loupmd cache prune`

Removes bundles, plans and gate references (what LoupMD can make again); models and libraries only when named with --kind.

```text
loupmd cache prune [-h] [--older-than DAYS] [--max-size SIZE]
                          [--kind {models,bundles,plans,gate_refs,lib}] [--dry-run]
```

| Argument | Meaning |
| --- | --- |
| `--older-than` `DAYS` | remove what was not used for DAYS days |
| `--max-size` `SIZE` | then remove the least recently used files until the kinds fit SIZE (500M, 1G, ...) |
| `--kind` `KIND` | prune this kind (repeatable; default: bundles, plans, gate_refs) (one of: models, bundles, plans, gate_refs, lib) |
| `--dry-run` | show what would be removed |

### `loupmd bench`

Times Langevin MD (300 K, 1 fs, skin 0.5 A, friction 0.01/fs) with a model on the systems of a suite and prints ms/step, us per atom-step and ns/day. The GPU must be idle.

```text
loupmd bench [-h] --model MODEL [--head HEAD] [--checkpoint PATH] [--sha256 HEX]
                    [--trusted] [--download] [--suite {standard,quick,families,capacity}]
                    [--quick] [--system NAME] [--protocol {standard,fastmace,quick}] [--device N]
                    [--force] [--compare {mace}] [--ab SIDE [SIDE ...]] [--json FILE]
```

| Argument | Meaning |
| --- | --- |
| `--model` `MODEL` | a label (see `loupmd models`), a checkpoint path or a URL |
| `--head` `HEAD` | the head of a multi-head model |
| `--checkpoint` `PATH` | the file of a label, when it is not in LOUPMD_MODELS |
| `--sha256` `HEX` | the stated hash of a file that is not in the label table |
| `--trusted` | load a file outside the label table without a hash |
| `--download` | download a label's checkpoint if it is missing |
| `--suite` `SUITE` | standard: NbCl5 3x3x3 and SrTiO3 10x10x10 (default); quick: one system, under a minute; families: one system for the model's family (one of: standard, quick, families, capacity) |
| `--quick` | short for --suite quick |
| `--system` `NAME` | time this system instead of a suite's (repeatable): NbCl5 3x3x3, SrTiO3 10x10x10, SrTiO3 6x6x6, NaCl 3x3x3, water 216, water 216 2x2x2 (one of: NbCl5 3x3x3, SrTiO3 10x10x10, SrTiO3 6x6x6, NaCl 3x3x3, water 216, water 216 2x2x2) |
| `--protocol` `PROTOCOL` | standard: 100 warm-up steps, median of 6 x 256; fastmace: 300, 4 x 300; quick: 50, 2 x 128 (default: the suite's) (one of: standard, fastmace, quick) |
| `--device` `N` | the GPU index (default 0) |
| `--force` | time even if the GPU is in use; the record says so |
| `--compare` `COMPARE` | also time another engine on the same structure (mace: mace-torch's ASE calculator) (one of: mace) |
| `--ab` `SIDE` | development: run the benchmark once per side in fresh processes in the order a b b a; a side is ENV=VAL[,ENV=VAL...] or - for no change |
| `--json` `FILE` | write the record (loupmd.bench/1) to FILE, or - for stdout |

### `loupmd validate`

Compares LoupMD's energy, forces and stress with MACE float64 on strained and rattled structures. A quantity passes when its deviation is at most three times MACE float32's own deviation, or below 0.01 meV/atom, 1e-5 eV/A, 1e-7 eV/A^3. Needs the export extra (torch, mace-torch). With --dispersion: LoupMD's D3 term against simple-dftd3 (the dftd3 package) at the same settings on the structures, within 1e-6 eV/atom, 1e-5 eV/A, 1e-7 eV/A^3; the model is not run. Exit code 1 when the check fails.

```text
loupmd validate [-h] [--model MODEL] [--dispersion PRESET] [--functional NAME]
                       [--head HEAD] [--checkpoint PATH] [--sha256 HEX] [--trusted] [--download]
                       [--set {oxide,organic,crystals}] [--structure FILE] [--copies N]
                       [--rattle A] [--ref-device {auto,cuda,cpu}] [--no-ref-cache]
                       [--include-unstable] [--specialised] [--capacity {whole,tiled,wavefront}]
                       [--charge CHARGE] [--spin-multiplicity SPIN_MULTIPLICITY] [--device N]
                       [--json FILE]
```

| Argument | Meaning |
| --- | --- |
| `--model` `MODEL` | a label (see `loupmd models`), a checkpoint path or a URL; with --dispersion a label, for its training functional |
| `--dispersion` `PRESET` | check this dispersion preset (`loupmd dispersion list`) instead of the model |
| `--functional` `NAME` | with --dispersion: the functional whose D3 parameters to use (default: the model's) |
| `--head` `HEAD` | the head of a multi-head model |
| `--checkpoint` `PATH` | the file of a label, when it is not in LOUPMD_MODELS |
| `--sha256` `HEX` | the stated hash of a file that is not in the label table |
| `--trusted` | load a file outside the label table without a hash |
| `--download` | download a label's checkpoint if it is missing |
| `--set` `SET` | oxide: NaCl, SrTiO3, NbCl5 (default); organic: urea, naphthalene, paracetamol, water; crystals: NaCl, NaNbO3, NbCl5 R2SCAN cells (one of: oxide, organic, crystals) |
| `--structure` `FILE` | gate rattled copies of your own structure instead of a set (repeatable) |
| `--copies` `N` | rattled copies per structure (default 2) |
| `--rattle` `A` | their displacement width in A (default 0.05) |
| `--ref-device` `REF_DEVICE` | where MACE computes the references; cpu for models whose float64 pass does not fit the GPU (default: the GPU if there is one) (one of: auto, cuda, cpu) |
| `--no-ref-cache` | compute the references again and overwrite the cached ones |
| `--include-unstable` | also gate configurations on which the model itself is not float32-reproducible |
| `--specialised` | refused: the kernels specialised to MACE-OMAT-0 medium (an option of 0.0.1) were retired, and every model runs the generic kernels |
| `--capacity` `CAPACITY` | whole: one call per structure (default); tiled: tile by tile, about four tiles per structure, as large cells run; wavefront: tiled in the wavefront order with the row pool (two-layer models) (one of: whole, tiled, wavefront) |
| `--charge` `CHARGE` | total charge of the cell in e (MACE-OMOL-0, MACE-POLAR-1) |
| `--spin-multiplicity` `SPIN_MULTIPLICITY` | 2S + 1 (models with a spin embedding) |
| `--device` `N` | the GPU index (default 0) |
| `--json` `FILE` | write the record (loupmd.gate/1) to FILE, or - for stdout |

### `loupmd build-kernels`

The library has the kernels of the models its release was built from. A model with another symmetric contraction needs its own, compiled here with nvcc into a kernel plugin in the cache; the engine then finds it by itself. Needs the CUDA toolkit and the export extra.

```text
loupmd build-kernels [-h] [--checkpoint PATH] [--sha256 HEX] [--trusted] [--arch sm_XY]
                            [--force]
                            MODEL
```

| Argument | Meaning |
| --- | --- |
| `MODEL` | a label (see `loupmd models`) or a checkpoint path |
| `--checkpoint` `PATH` | the file of a label, when it is not in LOUPMD_MODELS |
| `--sha256` `HEX` | the expected SHA-256 of a file outside the label table |
| `--trusted` | load a file outside the label table without a hash |
| `--arch` `sm_XY` | the GPU architecture to compile for (default: GPU 0's) |
| `--force` | compile even where the library or the cache already has the kernels |

### `loupmd md`

Molecular dynamics on a structure with a MACE model. Writes thermo rows, a trajectory, a restart and metadata.json into the output directory and prints a one-screen report.

```text
loupmd md [-h] [--structure FILE] --model MODEL [--head HEAD] [--checkpoint PATH]
                 [--sha256 HEX] [--trusted] [--download] [--charge CHARGE]
                 [--spin-multiplicity SPIN_MULTIPLICITY] [--ensemble ENSEMBLE] [--pressure GPA]
                 [--cell-mode {hydrostatic,full}] [--thermostat {langevin,csvr}]
                 [--friction PER_FS] [--tau FS] [--temperature K] [--schedule SEGMENTS]
                 [--profile NAME] [--timestep FS] [--steps STEPS] [--repeat A B C] [--seed SEED]
                 [--initial-velocities HOW] [--check CHECK] [--device N]
                 [--capacity {auto,whole,tiled,wavefront}] [--skin A] [--output DIR]
                 [--thermo-every N] [--trajectory-every N] [--restart-every N]
                 [--dispersion PRESET] [--d3-functional NAME]
                 [structure]
```

| Argument | Meaning |
| --- | --- |
| `structure` | the structure file (anything ASE reads) |
| `--structure` `FILE` | the structure file, as FastMACE spells it (instead of the positional argument) |
| `--model` `MODEL` | a label (see `loupmd models`), a checkpoint path or a URL |
| `--head` `HEAD` | the head of a multi-head model |
| `--checkpoint` `PATH` | the file of a label, when it is not in LOUPMD_MODELS |
| `--sha256` `HEX` | the stated hash of a file that is not in the label table |
| `--trusted` | load a file outside the label table without a hash |
| `--download` | download a label's checkpoint if it is missing |
| `--charge` `CHARGE` | total charge of the cell in e (MACE-OMOL-0, MACE-POLAR-1) |
| `--spin-multiplicity` `SPIN_MULTIPLICITY` | 2S + 1 (models with a spin embedding) |
| `--ensemble` `ENSEMBLE` | NVT (default), NVE or NPT (Langevin BAOAB at --pressure; experimental) |
| `--pressure` `GPA` | the target pressure in GPa of --ensemble NPT (default 0) |
| `--cell-mode` `CELL_MODE` | --ensemble NPT: hydrostatic (default: the cell keeps its shape) or full (one of: hydrostatic, full) |
| `--thermostat` `THERMOSTAT` | the NVT thermostat: langevin (default) or csvr (Bussi's velocity rescaling, which needs --tau) (one of: langevin, csvr) |
| `--friction` `PER_FS` | Langevin's friction per fs (default about 0.002) |
| `--tau` `FS` | the thermostat's time constant in fs (csvr; no default) |
| `--temperature` `K` | the thermostat's target |
| `--schedule` `SEGMENTS` | ramps and holds in order, separated by commas: ramp:START_K:END_K:PS and hold:K:PS, for example ramp:0:400:3,hold:400:1,ramp:400:0:24 |
| `--profile` `NAME` | a shipped profile (slow, fast, ultrafast, anneal) or a profile file: its schedule and output cadences, and for a profile at constant pressure NPT at its pressure |
| `--timestep` `FS` | the timestep in fs (default 0.5) |
| `--steps` `STEPS` | the number of steps (optional with a schedule or a profile) |
| `--repeat` `A B C` | the supercell |
| `--seed` `SEED` | the noise seed (default: drawn and recorded) |
| `--initial-velocities` `HOW` | keep (default), maxwell_boltzmann or a temperature in K: what a structure at rest starts with |
| `--check` `CHECK` | skip (default), gate or allow_unavailable: the accuracy gate before the first step |
| `--device` `N` | the GPU index (default 0) |
| `--capacity` `CAPACITY` | auto (default): the whole cell when it fits the GPU, else tiles (every integrator; with a dispersion term NVE and Langevin only, no barostat); whole, tiled or wavefront (tiles in the wavefront order, two-layer models) force one (one of: auto, whole, tiled, wavefront) |
| `--skin` `A` | the neighbour-list skin in A, or auto (default): chosen from how far the atoms move per step; it decides speed only |
| `--output` `DIR` | the output directory (default <structure stem>.loupmd) |
| `--thermo-every` `N` | thermo rows every N steps (default 20) |
| `--trajectory-every` `N` | trajectory frames every N steps (default 20; 0: none) |
| `--restart-every` `N` | restart points every N steps (default 10000; 0: only at the end) |
| `--dispersion` `PRESET` | a dispersion preset (`loupmd dispersion list`): the D3 term on every step |
| `--d3-functional` `NAME` | with --dispersion: the functional whose D3 parameters to use (default: the model's) |

### `loupmd init`

Prints a commented TOML file of every argument of `loupmd md`, with the ones given here filled in: loupmd init md STRUCTURE --model M ... > run.toml

```text
loupmd init [-h] KIND ...
```

#### `loupmd init md`

a run file for `loupmd run`

```text
loupmd init md [-h] [--structure FILE] --model MODEL [--head HEAD] [--checkpoint PATH]
                      [--sha256 HEX] [--trusted] [--download] [--charge CHARGE]
                      [--spin-multiplicity SPIN_MULTIPLICITY] [--ensemble ENSEMBLE]
                      [--pressure GPA] [--cell-mode {hydrostatic,full}]
                      [--thermostat {langevin,csvr}] [--friction PER_FS] [--tau FS]
                      [--temperature K] [--schedule SEGMENTS] [--profile NAME] [--timestep FS]
                      [--steps STEPS] [--repeat A B C] [--seed SEED] [--initial-velocities HOW]
                      [--check CHECK] [--device N] [--capacity {auto,whole,tiled,wavefront}]
                      [--skin A] [--output DIR] [--thermo-every N] [--trajectory-every N]
                      [--restart-every N] [--dispersion PRESET] [--d3-functional NAME]
                      [structure]
```

| Argument | Meaning |
| --- | --- |
| `structure` | the structure file (anything ASE reads) |
| `--structure` `FILE` | the structure file, as FastMACE spells it (instead of the positional argument) |
| `--model` `MODEL` | a label (see `loupmd models`), a checkpoint path or a URL |
| `--head` `HEAD` | the head of a multi-head model |
| `--checkpoint` `PATH` | the file of a label, when it is not in LOUPMD_MODELS |
| `--sha256` `HEX` | the stated hash of a file that is not in the label table |
| `--trusted` | load a file outside the label table without a hash |
| `--download` | download a label's checkpoint if it is missing |
| `--charge` `CHARGE` | total charge of the cell in e (MACE-OMOL-0, MACE-POLAR-1) |
| `--spin-multiplicity` `SPIN_MULTIPLICITY` | 2S + 1 (models with a spin embedding) |
| `--ensemble` `ENSEMBLE` | NVT (default), NVE or NPT (Langevin BAOAB at --pressure; experimental) |
| `--pressure` `GPA` | the target pressure in GPa of --ensemble NPT (default 0) |
| `--cell-mode` `CELL_MODE` | --ensemble NPT: hydrostatic (default: the cell keeps its shape) or full (one of: hydrostatic, full) |
| `--thermostat` `THERMOSTAT` | the NVT thermostat: langevin (default) or csvr (Bussi's velocity rescaling, which needs --tau) (one of: langevin, csvr) |
| `--friction` `PER_FS` | Langevin's friction per fs (default about 0.002) |
| `--tau` `FS` | the thermostat's time constant in fs (csvr; no default) |
| `--temperature` `K` | the thermostat's target |
| `--schedule` `SEGMENTS` | ramps and holds in order, separated by commas: ramp:START_K:END_K:PS and hold:K:PS, for example ramp:0:400:3,hold:400:1,ramp:400:0:24 |
| `--profile` `NAME` | a shipped profile (slow, fast, ultrafast, anneal) or a profile file: its schedule and output cadences, and for a profile at constant pressure NPT at its pressure |
| `--timestep` `FS` | the timestep in fs (default 0.5) |
| `--steps` `STEPS` | the number of steps (optional with a schedule or a profile) |
| `--repeat` `A B C` | the supercell |
| `--seed` `SEED` | the noise seed (default: drawn and recorded) |
| `--initial-velocities` `HOW` | keep (default), maxwell_boltzmann or a temperature in K: what a structure at rest starts with |
| `--check` `CHECK` | skip (default), gate or allow_unavailable: the accuracy gate before the first step |
| `--device` `N` | the GPU index (default 0) |
| `--capacity` `CAPACITY` | auto (default): the whole cell when it fits the GPU, else tiles (every integrator; with a dispersion term NVE and Langevin only, no barostat); whole, tiled or wavefront (tiles in the wavefront order, two-layer models) force one (one of: auto, whole, tiled, wavefront) |
| `--skin` `A` | the neighbour-list skin in A, or auto (default): chosen from how far the atoms move per step; it decides speed only |
| `--output` `DIR` | the output directory (default <structure stem>.loupmd) |
| `--thermo-every` `N` | thermo rows every N steps (default 20) |
| `--trajectory-every` `N` | trajectory frames every N steps (default 20; 0: none) |
| `--restart-every` `N` | restart points every N steps (default 10000; 0: only at the end) |
| `--dispersion` `PRESET` | a dispersion preset (`loupmd dispersion list`): the D3 term on every step |
| `--d3-functional` `NAME` | with --dispersion: the functional whose D3 parameters to use (default: the model's) |

### `loupmd run`

Runs a TOML run file (see `loupmd init md`) or the metadata.json of a finished run. With --resume, continues the run of a directory from its restart.

```text
loupmd run [-h] [--resume DIR] [--steps STEPS] [file]
```

| Argument | Meaning |
| --- | --- |
| `file` | a run file (.toml) or a metadata.json |
| `--resume` `DIR` | continue the run in DIR from its restart file |
| `--steps` `STEPS` | with --resume: the number of further steps (default: what is left of the run's steps or schedule) |

### `loupmd relax`

Relaxes a structure at fixed cell with ASE's LBFGS or FIRE until the largest force on an atom is below --fmax, and writes it with its energy and forces as extended XYZ.

```text
loupmd relax [-h] --model MODEL [--head HEAD] [--checkpoint PATH] [--sha256 HEX]
                    [--trusted] [--download] [--charge CHARGE]
                    [--spin-multiplicity SPIN_MULTIPLICITY] [--optimizer {lbfgs,fire}]
                    [--fmax EV_PER_A] [--steps STEPS] [--maxstep A] [--repeat A B C]
                    [--dispersion PRESET] [--d3-functional NAME] [--device N] [--output FILE]
                    structure
```

| Argument | Meaning |
| --- | --- |
| `structure` | the structure file (anything ASE reads) |
| `--model` `MODEL` | a label (see `loupmd models`), a checkpoint path or a URL |
| `--head` `HEAD` | the head of a multi-head model |
| `--checkpoint` `PATH` | the file of a label, when it is not in LOUPMD_MODELS |
| `--sha256` `HEX` | the stated hash of a file that is not in the label table |
| `--trusted` | load a file outside the label table without a hash |
| `--download` | download a label's checkpoint if it is missing |
| `--charge` `CHARGE` | total charge of the cell in e (MACE-OMOL-0, MACE-POLAR-1) |
| `--spin-multiplicity` `SPIN_MULTIPLICITY` | 2S + 1 (models with a spin embedding) |
| `--optimizer` `OPTIMIZER` | default lbfgs (one of: lbfgs, fire) |
| `--fmax` `EV_PER_A` | default 0.05 eV/A |
| `--steps` `STEPS` | at most this many optimiser steps |
| `--maxstep` `A` | the longest step (default 0.2 A) |
| `--repeat` `A B C` | the supercell |
| `--dispersion` `PRESET` | a dispersion preset (`loupmd dispersion list`): the D3 term in every force call |
| `--d3-functional` `NAME` | with --dispersion: the functional whose D3 parameters to use (default: the model's) |
| `--device` `N` | the GPU index (default 0) |
| `--output` `FILE` | the relaxed structure (default <structure stem>.relaxed.extxyz) |

### `loupmd quench`

Picks trajectory frames of a finished run per bin of the thermostat's target temperature (on each ramp, evenly spaced, plus the final frame), relaxes each at fixed cell and writes the minima, a CSV row per frame and a summary into DIR/quench/. An interrupted quench continues.

```text
loupmd quench [-h] [--min-K MIN_K] [--bin-K BIN_K] [--per-bin PER_BIN] [--fmax FMAX]
                     [--optimizer {gpu-LBFGS,ase-LBFGS,LBFGS,gpu-FIRE,FIRE}]
                     [--max-steps MAX_STEPS] [--name NAME] [--device N]
                     DIR
```

| Argument | Meaning |
| --- | --- |
| `DIR` | the output directory of a run made with a schedule |
| `--min-K` `MIN_K` | the lowest target temperature picked from |
| `--bin-K` `BIN_K` | the width of a temperature bin |
| `--per-bin` `PER_BIN` | frames per bin and ramp |
| `--fmax` `FMAX` | relax until the largest force is below this, eV/A |
| `--optimizer` `OPTIMIZER` | default gpu-LBFGS (FastMACE's name; LoupMD's LBFGS) (one of: gpu-LBFGS, ase-LBFGS, LBFGS, gpu-FIRE, FIRE) |
| `--max-steps` `MAX_STEPS` | optimiser steps per frame at most |
| `--name` `NAME` | the quench's folder in DIR (default quench) |
| `--device` `N` | the GPU index (default 0) |

### `loupmd dispersion`

Dispersion corrections: what the presets mean, which summation methods exist and where each is wrong, and the parameter sets with their sources.

```text
loupmd dispersion [-h] WHAT ...
```

#### `loupmd dispersion list`

the presets, dampings, methods and parameter sets

```text
loupmd dispersion list [-h]
```
