# ASE

LoupMD has three ASE front ends: a calculator, drop-in MD classes and drop-in optimisers. The calculator fits any
ASE workflow but makes one force call per ASE step. The drop-in classes keep ASE's arguments and hand the whole loop
to the engine.

## The calculator

```python
import loupmd as lm
from ase.io import read

atoms = read("structure.cif")
atoms.calc = lm.calc("omat0-medium")
energy = atoms.get_potential_energy()      # eV
forces = atoms.get_forces()                # eV/A
stress = atoms.get_stress()                # Voigt order, eV/A^3
```

`lm.calc(model, ...)` takes what `lm.load` takes: a label, a checkpoint path or a URL, with `head`, `checkpoint`,
`sha256`, `trusted`, `download`, `charge`, `spin_multiplicity` and `device` (default GPU 0). See
[models](models.md). It also takes:

- `skin`: the neighbour-list skin in A (default 0.5).
- `dispersion`: a D3 preset such as `"d3bj"`, or an `lm.Dispersion` ([dispersion](dispersion.md)).

The calculator is `loupmd.ase.LoupCalculator`. Its properties are `energy`, `free_energy` (the same number),
`forces`, `stress` and `energies`:

- The stress is computed only when ASE asks for it.
- `atoms.get_potential_energies()` gives each atom's energy (its MACE node energy plus its E0; they sum to the
  energy). It costs one more energy-only pass. It needs the generic kernels and a short-range model, so MACE-POLAR-1
  has none.
- The cell must be periodic in all three directions. A molecule or a slab goes into a box with vacuum and
  `pbc=True`; anything else is refused before the export.

The model file is found and checked when the calculator is made. The bundle and the engine are made at the first
calculation, for the structure's elements. The engine is kept while the atomic numbers stay. A change of their order
or count sets the system again on the same engine. An element outside the loaded bundle gets a new bundle and engine.

### One model, several calculators

`lm.calc` also takes a `ForceField` from `lm.load`. All calculators made from one `ForceField` share its engine, so
the model is exported and loaded once. Each calculator sets its own dispersion term on the engine before it runs.

```python
ff = lm.load("omat0-medium")
plain = lm.calc(ff)
with_d3 = lm.calc(ff, dispersion="d3bj")
```

The load options (`head`, `checkpoint`, `sha256`, `trusted`, `download`, `charge`, `spin_multiplicity`, `device`)
are refused with a `ForceField`: they belong to `lm.load`.

### What the calculator is for

ASE calls a calculator once per step and moves the atoms itself, so each call copies positions to the GPU and
forces back. Use it for relaxations with ASE's tools, phonons, NEB and comparisons with other codes. For MD use
`lm.md`, `lm.Simulation` ([MD](md.md)) or the drop-in classes below, where the engine owns the loop.

With `dispersion=`, `long_range="auto"` runs the plain sum, because the calculator has no setup structure to choose
on. Name `long_range="split"` for the split sum.

## Drop-in MD classes

A script written for `ase.md` runs after its import line changes and the model is named:

```python
from ase import units
from loupmd.ase import Langevin

dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=300, friction=0.01 / units.fs, model="omat0-medium",
               trajectory="md.traj", logfile="md.log", loginterval=10)
dyn.attach(report, interval=100)
dyn.run(1000)
```

Every class takes ASE's arguments in ASE's units, and these keywords besides:

- `model=` (required): a label, a checkpoint path or an `lm.load` result; `head=` and the load options as for
  `lm.calc`.
- `seed=`: the thermostat's noise seed. The noise is drawn on the GPU.
- `output=`: a directory or an `lm.Output` for LoupMD's own output ([output](../reference/output.md)).
- `check=`: the accuracy gate before the first step, `"skip"` by default (see [models](models.md)).

| Class | ASE arguments | Notes |
|---|---|---|
| `VelocityVerlet` | `timestep` | NVE |
| `Langevin` | `timestep`, `temperature_K` (or `temperature` in eV), `friction`, `fixcm=True` | |
| `Bussi` | `timestep`, `temperature_K`, `taut` | refuses a structure at rest |
| `NoseHooverChainNVT` | `timestep`, `temperature_K`, `tdamp`, `tchain=3`, `tloop=1` | |
| `NVTBerendsen` | `timestep`, `temperature_K` (or `temperature`), `taut`, `fixcm=True` | |
| `NPTBerendsen` | `temperature_K`, `pressure_au`, `compressibility_au`, `taut` (500 fs), `taup` (1000 fs), `fixcm=True` | experimental |
| `Inhomogeneous_NPTBerendsen` | as `NPTBerendsen`, with `mask=(1, 1, 1)` | experimental; the angles stay |
| `LangevinBAOAB` | `temperature_K`, `externalstress` (a scalar), `hydrostatic=False`, `T_tau`, `P_tau` | experimental |
| `IsotropicMTKNPT` | `temperature_K`, `pressure_au`, `tdamp`, `pdamp`, `tchain`, `pchain`, `tloop`, `ploop` | experimental; chains of at most 8 links |
| `MTKNPT` | as `IsotropicMTKNPT`, with the full cell | experimental |

What differs from ASE:

- `rng=`, `comm=` and `master=` are refused unless `None`. `restart=` and unknown keywords are refused.
- `Langevin(fixcm=True)` keeps the centre of mass at rest with the mass-weighted projection of the noise. ASE's
  projection differs from it for unequal masses.
- `VelocityVerlet`, `Bussi` and `NoseHooverChainNVT` remove no centre-of-mass motion, as in ASE. They count 3 N
  degrees of freedom less three for each fixed atom; ASE's chain counts 3 N even with fixed atoms.
- `NVTBerendsen` and `NPTBerendsen` with `fixcm=True` count three degrees of freedom fewer than ASE's classes.
- `NPTBerendsen` refuses ASE's deprecated `pressure=` (bar) and `compressibility=` (1/bar). It makes one force
  call per step, so the first kick of a step uses the forces of the cell before the barostat scaled it.
- `LangevinBAOAB` refuses a run without `temperature_K` or `externalstress`, a stress tensor as target, `P_mass`,
  `P_mass_factor` other than 1 and `disable_cell_langevin=True`. With the full cell the barostat's drag follows
  Quigley and Probert.

The integrators are described in [integrators](../reference/integrators.md) and [NPT](npt.md).

### Observers

`trajectory`, `logfile`, `loginterval` and `append_trajectory` mean what they mean in ASE: an ASE trajectory file
and ASE's `MDLogger` lines. `attach(function, interval)` calls a function at step 0 and every `interval` steps.

`run(steps)` hands the engine the steps between two observer calls in one piece; the piece is the greatest common
divisor of the observers' intervals. Before every observer call and at the end of `run`, `atoms` holds the current
positions, momenta and a calculator with the energy and forces. Each observer stops the engine, so a long run with
`loginterval=1` is slow. LoupMD's own `output=` is written without stopping the run.

`dyn.summary` is the run report so far, `dyn.simulation` the underlying `lm.Simulation`. `dyn.close()` (or a
`with` block) closes the trajectory, the log and the output.

## Optimisers

`loupmd.ase` also has `LBFGS` and `FIRE` with ASE's arguments:

```python
from loupmd.ase import LBFGS

opt = LBFGS(atoms, trajectory="relax.traj", model="omat0-medium")
converged = opt.run(fmax=0.05)
```

They run `lm.optimize`, which ports ASE 3.27's LBFGS and FIRE and takes the same steps. LBFGS at fixed cell runs
on the GPU inside the engine, so a step costs little more than its force call. FIRE, and any relaxation with a cell
filter, takes its steps on the host around the engine's force call.

- `logfile` is standard output by default (`None` for none); `trajectory` writes an ASE trajectory.
- For the cell, wrap the atoms in `FrechetCellFilter(atoms, scalar_pressure=...)` or pass
  `cell_filter=lm.optimize.FrechetCell(...)`. ASE's other filters are refused by name.
- Not offered: `restart=`, `use_line_search=True`, FIRE's `maxmove` (use `maxstep`) and `position_reset_callback`.

ASE's own optimisers also work with `atoms.calc = lm.calc(...)`, with one calculator call per step.
[Relaxation](relax.md) covers `lm.optimize` and `loupmd relax`.
