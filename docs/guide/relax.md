# Relaxation

LoupMD relaxes structures with ports of ASE 3.27's `LBFGS` and `FIRE`, which take ASE's steps, and of its
`FrechetCellFilter` for the cell. `loupmd relax` relaxes one structure file; `lm.optimize.relax` is the Python call;
`loupmd quench` relaxes picked frames of a finished MD run.

## `loupmd relax`

```bash
loupmd relax structure.cif --model omat0-medium --fmax 0.01
```

It relaxes the positions at fixed cell until the largest force on an atom is below `--fmax` (default 0.05 eV/A) or
`--steps` optimiser steps were taken, and writes the structure with its energy and forces as extended XYZ to
`--output` (default `<structure stem>.relaxed.extxyz` in the current directory). `--optimizer` is `lbfgs` (default)
or `fire`; `--maxstep` is the longest step, default 0.2 A. `--repeat A B C` relaxes a supercell. `--dispersion
PRESET` (with `--d3-functional NAME` as in `loupmd md`) adds the D3 term to every force call. It prints the
result, the number of force calls and the time, and exits with 1 when the relaxation did not converge.
[The CLI reference](../reference/cli.md) lists every flag. The command relaxes at fixed cell only;
the cell is relaxed from Python.

## `lm.optimize.relax`

```python
import loupmd as lm
from ase.io import read

ff = lm.load("omat0-medium")
result = lm.optimize.relax(ff, read("structure.cif"), lm.optimize.LBFGS(), fmax=0.01)
print(result)                  # converged after ... steps: largest force ... eV/A, energy ... eV
relaxed = result.atoms         # with energy and forces on a single-point calculator
```

`relax(force_field, atoms, optimizer=None, *, fmax=0.05, steps=100_000_000, skin=0.5, dispersion=None,
callback=None)`. The input `atoms` is not changed. `optimizer` defaults to `LBFGS()`. `dispersion` adds the D3 term
to the energy, forces and stress, as `lm.calc(dispersion=)` does ([dispersion](dispersion.md)); without it the
relaxation has none, whatever term another calculator on the same force field uses. `callback(step, positions, energy, max_force)` is called
after every force call. The result is a `RelaxResult` with `atoms`, `steps`, `converged`, `max_force` (eV/A),
`energy` (eV), `force_calls`, and the time in the force calls (`force_s`) and in the optimiser (`optimizer_s`).

| record | fields and defaults (ASE's) |
|---|---|
| `LBFGS` | `maxstep_A=0.2` (at most 1.0, as ASE), `memory=100`, `damping=1.0`, `alpha=70.0`, `cell_filter=None` |
| `FIRE` | `dt=0.1`, `maxstep_A=0.2`, `dtmax=1.0`, `Nmin=5`, `finc=1.1`, `fdec=0.5`, `astart=0.1`, `fa=0.99`, `a=0.1`, `cell_filter=None` |

ASE's `use_line_search` for LBFGS is not offered, and FIRE's `downhill_check=True` is refused. FIRE's `dt` and
`dtmax` are in ASE time units, as in ASE: they are step-size parameters, not physical times.

**Where the steps run.** LBFGS at fixed cell runs inside the engine on the GPU: the history and its products stay on
the device, so a step costs little more than its force call. FIRE, and every relaxation with a cell filter, takes
its steps on the host around the engine's force call.

## Convergence

The relaxation stops when the largest force on a free atom is below `fmax`, or after `steps` optimiser steps;
`converged` says which. A structure already below `fmax` takes no step. With a cell filter the three cell rows
count for `fmax` too, as in ASE. A non-finite energy or force stops the relaxation with `NonFiniteError`.

## Fixed atoms

Atoms under ASE's `FixAtoms` do not move, and their forces do not count for `fmax`. Every other ASE constraint is
refused by name. The cell cannot be relaxed while atoms are fixed.

## Relaxing the cell

`cell_filter=lm.optimize.FrechetCell(...)` relaxes the positions and the cell together, as ASE's
`FrechetCellFilter`:

```python
cell = lm.optimize.FrechetCell(scalar_pressure_GPa=0.0)
result = lm.optimize.relax(ff, atoms, lm.optimize.LBFGS(cell_filter=cell), fmax=0.01)
```

Its fields are ASE's: `scalar_pressure_GPa` (ASE's `scalar_pressure`, in GPa; the enthalpy E + p V is minimised),
`hydrostatic_strain`, `constant_volume`, `mask` (six Voigt flags or a 3x3 of flags) and `exp_cell_factor` (`None`:
the number of atoms). The result's atoms carry the stress as well. The cell filter is refused with fixed atoms and
for models with long-range terms (MACE-POLAR-1), whose long-range part has no stress the filter can use.

The ASE drop-ins `loupmd.ase.optimize.LBFGS` and `FIRE` take an ASE `Atoms`, or one wrapped in ASE's
`FrechetCellFilter` (ASE's other filters are refused), and run these optimisers ([ASE](ase.md)).

## Relaxing before MD

`Simulation.relax(optimizer=None, *, fmax=0.05, steps=...)` relaxes the starting structure at fixed cell before the
first step, with the run's dispersion term. The run then starts from the relaxed positions with the structure's
velocities; the call is refused once the run has started, and so is an optimiser with a cell filter (relax the
cell with `lm.optimize.relax` first and make the `Simulation` from `result.atoms`). `lm.md` and `loupmd md` do not relax: a profile's `relax_fmax` is recorded only
([molecular dynamics](md.md)).

```python
sim = lm.Simulation(atoms, ff, integrator=lm.integrators.Langevin(temperature_K=300, friction_per_fs=0.002),
                    output=lm.Output("run/"))
print(sim.relax(fmax=0.01))
sim.run(10_000)
```

## Quenching a finished run

`loupmd quench DIR` takes a finished run made with a schedule (`--schedule` or `--profile`) and a trajectory, picks
frames by the thermostat's target temperature and relaxes each to its minimum at the frame's cell. The selection is
FastMACE's: from `--min-K` up to the highest target in bins of `--bin-K`, on each ramp (heating and cooling)
separately, `--per-bin` frames per bin evenly spaced over the bin's steps, plus the final frame.

```bash
loupmd md structure.cif --model omat0-medium --profile anneal --timestep 1 --initial-velocities 10
loupmd quench structure.loupmd
```

The settings come from the command line, else the defaults (FastMACE's, which are also the `anneal` profile's). A
profile's own `quench` table is not read in this version:

| flag | default |
|---|---|
| `--min-K` | 500 K |
| `--bin-K` | 100 K |
| `--per-bin` | 2 |
| `--fmax` | 0.01 eV/A |
| `--optimizer` | `gpu-LBFGS` (also `ase-LBFGS`, `LBFGS`: LoupMD's LBFGS; `gpu-FIRE`, `FIRE`: LoupMD's FIRE) |
| `--max-steps` | 5,000 optimiser steps per frame |

The model is the run's, loaded again from its `metadata.json`, and a run with dispersion is quenched with the
same D3 term. The run must be complete: a run stopped inside its
schedule is refused (finish it with `loupmd run --resume DIR`), and so is a run without a schedule or a trajectory.
The results go into `DIR/quench/` (`--name` chooses another folder):

- `settings.json`: the six settings, written first;
- `captured.traj`: the picked MD frames;
- `quenched.traj`: their minima;
- `quench.csv`: one row per frame, written after every frame: phase, bin, step, target and MD temperature, MD and
  quenched energy per atom, converged, optimiser steps, time, largest force and volume per atom;
- `quench.json`: the settings and the summary, written last.

An interrupted quench continues after its last row when it is run again with the same settings; other settings are
refused (choose another `--name`). A complete quench is returned as it is. The command prints the range and mean of
the quenched energies per ramp and exits with 1 when any frame did not converge. From Python,
`loupmd.quench.quench_run(directory, force_field, settings)` does the same.
