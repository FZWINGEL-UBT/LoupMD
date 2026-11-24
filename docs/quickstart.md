# Quickstart

This page runs one simulation three ways: from the command line, from Python, and through ASE. The run is 10,000
steps of NVT molecular dynamics at 300 K with a Langevin thermostat and a 0.5 fs timestep (5 ps), with the
`omat0-medium` model. Then it shows what the run wrote, and a short relaxation.

## Install

```bash
pip install "loupmd[export]"
loupmd doctor
loupmd models --download omat0-medium
```

The first line installs LoupMD from PyPI with the export extra (PyTorch and MACE), which a model needs the first
time it runs on a new set of elements. `loupmd doctor` checks the GPU, the driver and the libraries, and the last
line fetches the model file once (79 MB) into the cache. [Install](install.md) says what the machine needs.

You also need a structure file that ASE reads, here `structure.cif`.

## One run, three ways

Each variant writes to the directory `run`, and a new run refuses a directory that already holds output: delete
`run` or change the name before trying the next one.

### Command line

```bash
loupmd md structure.cif --model omat0-medium --temperature 300 --steps 10000 \
    --initial-velocities maxwell_boltzmann --seed 1 --output run
```

`loupmd md` prints the report when the run ends. The NVT thermostat is Langevin by default, with a friction of
about 0.002/fs. `--timestep` defaults to 0.5 fs. Without `--output` the directory is `structure.loupmd`.

### Python

```python
import loupmd as lm

result = lm.md("structure.cif", model="omat0-medium", temperature_K=300, steps=10_000,
               initial_velocities="maxwell_boltzmann", seed=1, output="run")
print(result.summary)
result.simulation.close()
```

`lm.md` takes the same settings as `loupmd md`, under the names `temperature_K`, `timestep_fs`, `steps`, `seed`,
`output`, `ensemble`, `capacity`, `dispersion` and so on. [The API reference](reference/api.md) lists them all.

### ASE

```python
import loupmd as lm
from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from loupmd.ase.md import Langevin

atoms = read("structure.cif")
model = lm.load("omat0-medium")
atoms.calc = lm.calc(model)                  # an ASE calculator: energy, forces, stress
print(atoms.get_potential_energy())

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=300, friction=0.02, model=model, seed=1, output="run",
               trajectory="md.traj", logfile="md.log", loginterval=100)
dyn.run(10_000)
print(dyn.summary)
dyn.close()
```

`lm.calc` is an ordinary ASE calculator. It runs one force evaluation per call, so use it for single points,
relaxations, phonons and NEB, and with any ASE tool.

The classes of `loupmd.ase.md` take ASE's arguments and units and run on LoupMD's engine: change the import line of
an ASE script and add `model=`. `friction=0.02` is in inverse ASE time units, the same friction as the default of
`lm.md`. The engine owns the loop and stops only for observers: `trajectory=`, `logfile=` and functions attached
with `dyn.attach(function, interval=N)` are called every `loginterval` (or `interval`) steps with `atoms` up to date.
`output=` writes LoupMD's own files without stopping the run, and is the faster choice for long runs.

`Langevin`, `VelocityVerlet`, `Bussi`, `NoseHooverChainNVT` and `NVTBerendsen` hold the volume constant;
`LangevinBAOAB`, `IsotropicMTKNPT`, `MTKNPT`, `NPTBerendsen` and `Inhomogeneous_NPTBerendsen` change the cell. See
[ASE](guide/ase.md) and [integrators](reference/integrators.md).

## Reading the result

`lm.md` returns a `RunResult`:

| Field | What it holds |
|---|---|
| `summary` | the report as text (`print(result)` prints it too) |
| `plan` | the execution plan: whole cell or tiles, kernels, memory |
| `gate` | the accuracy gate's status and record |
| `simulation` | the `Simulation`, which can run on, save a restart and must be closed |
| `metadata` | the dict written to `metadata.json` |
| `output` | the output directory |

The output directory holds:

| File | Content |
|---|---|
| `thermo.0.csv`, `thermo.1.csv`, ... | one row every `--thermo-every` steps (default 20): `step`, `time_ps`, `phase`, `target_K`, `temperature_K`, `potential_eV_atom`, `kinetic_eV_atom`, `total_eV_atom`; NPT runs add `volume_A3_atom` and `pressure_GPa` |
| `trajectory.0.traj`, ... | an ASE trajectory, a frame every `--trajectory-every` steps (default 20; 0 for none) |
| `restart.npz` | the restart, every `--restart-every` steps (default 10,000) and at the end |
| `simulation.json` | what a resume needs to load the model and continue |
| `metadata.json` | the record of the run: arguments, model, plan, gate, timing; `loupmd run metadata.json` repeats it |

A new pair of thermo and trajectory files starts at every restart point. To read them joined:

```python
from loupmd.output import read_frames, read_thermo

header, rows = read_thermo("run")          # rows are lists of strings
frames = read_frames("run")                # Atoms with momenta, energy and forces
```

`ase.io.read("run/trajectory.0.traj", index=":")` reads one file. [Output files](reference/output.md) has the
details, and [restarts](guide/restart.md) shows how to continue a run (`loupmd run --resume run`).

The report has these blocks:

```text
LoupMD 0.0.3: MD run, status ...
Model and structure     the label, file, SHA-256 and head; atoms, elements and cell lengths in A
Physics                 the ensemble, integrator, timestep in fs, steps and seed
Plan                    whole cell or tiles, the GPU, memory, kernels, the neighbour skin in A, the precision
Gate                    the accuracy gate, or "skipped"
Setup                   the bundle, and the seconds for export or cache, engine and the first force evaluation
Speed                   ms/step, ns/day, million atom-steps/s, neighbour rebuilds
Licence                 the model's licence and LoupMD's
Warnings                anything to know, for example how the starting velocities were drawn
Output                  the directory and its files
```

The gate is skipped by default. `--check gate` (Python: `check="gate"`) compares energies, forces and stress with
MACE in float64 on rattled copies of your structure before the first step; `loupmd validate` runs the same check on
its own. See [models](guide/models.md).

## A short relaxation

`loupmd relax` relaxes the positions at fixed cell until the largest force is below `--fmax` (default 0.05 eV/A):

```bash
loupmd relax structure.cif --model omat0-medium --fmax 0.01
```

It writes `structure.relaxed.extxyz` in the current directory (`--output FILE` to change it), with the energy and
forces. `--optimizer fire` uses FIRE instead of LBFGS; `--maxstep` (default 0.2 A) limits a step. The exit status
is 1 when it did not converge.

In Python, `lm.optimize.relax` relaxes a copy of the structure; with a `FrechetCell` filter the cell relaxes too:

```python
import loupmd as lm
from ase.io import read, write

model = lm.load("omat0-medium")
atoms = read("structure.cif")

result = lm.optimize.relax(model, atoms, lm.optimize.LBFGS(), fmax=0.01)
print(result)                        # converged after N steps: largest force ..., energy ...

cell = lm.optimize.LBFGS(cell_filter=lm.optimize.FrechetCell(scalar_pressure_GPa=0.0))
result = lm.optimize.relax(model, result.atoms, cell, fmax=0.01)
write("relaxed.extxyz", result.atoms)
```

`RelaxResult` has `atoms`, `steps`, `converged`, `max_force` (eV/A), `energy` (eV), `force_calls`, `optimizer_s`
and `force_s`. LBFGS at fixed cell runs on the GPU inside the engine. `from loupmd.ase import LBFGS, FIRE` gives
ASE-style optimisers for ASE scripts. See [relaxation](guide/relax.md).

## Where to go next

- [Molecular dynamics](guide/md.md): thermostats, schedules and profiles, run files (`loupmd init md`).
- [Constant pressure](guide/npt.md) and [restarts](guide/restart.md).
- [Models](guide/models.md): the families, heads, charge and spin, files outside the label table.
- [Large cells](guide/large-cells.md), [dispersion](guide/dispersion.md), [Monte Carlo](guide/monte-carlo.md).
- [Benchmarks](guide/benchmark.md) (`loupmd bench`) and the [FAQ](faq.md).
- Reference: [command line](reference/cli.md), [Python API](reference/api.md), [errors](reference/errors.md).
