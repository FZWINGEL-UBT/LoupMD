# Molecular dynamics

`lm.md` and `loupmd md` run MD on a structure in one call. They check every argument before the model is loaded,
run the steps on the GPU, write the output directory and print a one-screen report. `lm.Simulation` is the same run
taken step by step, with events between the calls.

```python
import loupmd as lm

result = lm.md("structure.cif", model="omat0-medium", temperature_K=300, steps=10_000)
print(result)                      # the report; result.metadata is what metadata.json holds
```

```bash
loupmd md structure.cif --model omat0-medium --temperature 300 --steps 10000
```

## Ensembles and thermostats

| `ensemble=` / `--ensemble` | integrator |
|---|---|
| `"NVT"` (default) | Langevin (`lm.integrators.Langevin`) |
| `"NVE"` | velocity Verlet from the structure's momenta; `temperature_K` and `schedule` are refused |
| `"NPT"` | Langevin BAOAB with a barostat, experimental; see [constant pressure](npt.md) |

The Langevin friction defaults to 0.02 per ASE time unit, about 0.002/fs (`--friction PER_FS` changes it). On the
command line `--thermostat csvr --tau FS` selects Bussi's velocity rescaling (CSVR); its time constant has no
default. CSVR rescales the velocities it finds, so a structure at rest is refused: draw velocities first (below).

`integrator=` takes any `lm.integrators` record and then decides the ensemble: `NVE`, `Langevin`, `CSVR`,
`NoseHooverChain`, `Berendsen` (not canonical: for equilibration), and the barostats `LangevinBAOAB`, `MTK` and
`Berendsen` with a pressure. A thermostat's temperature, friction and time constant have no defaults. A
`temperature_K=` or `timestep_fs=` that differs from the record's fields is an error.

```python
nhc = lm.integrators.NoseHooverChain(timestep_fs=1.0, temperature_K=300, tdamp_fs=100)
result = lm.md("structure.cif", model="omat0-medium", integrator=nhc, steps=10_000)
```

`fixcm="auto"` (every record but `LangevinBAOAB`) keeps the centre of mass at rest, so the temperature counts three
degrees of freedom fewer. It is off when the structure has atoms under ASE's `FixAtoms`; those atoms stay in place
and count no degrees of freedom. Any other ASE constraint is refused.

## Timestep, velocities and seed

- **Timestep:** `timestep_fs` / `--timestep`, default 0.5 fs.
- **Initial velocities:** `initial_velocities="keep"` (default) leaves a structure at rest at rest.
  `"maxwell_boltzmann"` draws Maxwell-Boltzmann momenta at the run's first target; a temperature in K draws them
  at that temperature (NVE has no target, so it takes only this form). Velocities are drawn only for a structure
  at rest: a structure that already moves keeps its momenta, and the report says which happened. A schedule that
  starts at 0 K needs a temperature in K here.
- **Seed:** `seed=` / `--seed` fixes the thermostat's noise and the drawn velocities (their own stream). Without it
  a seed is drawn and written to `metadata.json` and `simulation.json`, so the run can be repeated. GPU sums in
  float32 differ in their last bits from run to run, so two runs with the same seed agree closely, not bitwise.

## Temperature schedules and profiles

A schedule is a list of ramps and holds in ps. On a ramp every step runs at the target of its end point, so the
target changes every step; on a hold it stays. Each segment is rounded to whole steps (half up); a segment that
rounds to 0 steps and a jump in temperature between segments are refused.

```python
from loupmd import protocols

schedule = protocols.ramp(0, 400, ps=3) + protocols.hold(400, ps=1) + protocols.ramp(400, 0, ps=24)
result = lm.md("structure.cif", model="omat0-medium", schedule=schedule, initial_velocities=10.0)
```

```bash
loupmd md structure.cif --model omat0-medium --schedule ramp:0:400:3,hold:400:1,ramp:400:0:24 \
    --initial-velocities 10
```

`ramp(start_K, end_K, rate_K_per_ps=...)` gives a ramp by its rate instead of its length. With a schedule `steps`
is optional: without it the whole schedule runs; more steps than it has are refused. `temperature_K` and a schedule
exclude each other. The thermo file's `phase` column says `heating`, `hold` or `cooling` (`constant` without a
schedule), and a row and a frame are also written at the end of every segment. After the last segment the
thermostat stays at the last target.

A profile is a named schedule with its output cadences. `--profile NAME` takes a shipped profile or a `.toml` file:

| profile | schedule | thermo and frames every |
|---|---|---|
| `slow` | 0 to 400 K in 30 ps, 400 to 0 K in 240 ps | 100 steps |
| `fast` | 0 to 400 K in 3 ps, 400 to 0 K in 24 ps | 100 steps |
| `ultrafast` | 0 to 400 K in 1 ps, 400 to 0 K in 8 ps | 100 steps |
| `anneal` | 0 to 1,000 K in 20 ps, 1,000 to 250 K in 250 ps | 20 steps |

A profile file has `name`, `[[segments]]` tables (`kind = "ramp"` with `start_K`, `end_K` and `ps` or
`rate_K_per_ps`; `kind = "hold"` with `temperature_K` and `ps`) and optionally `sample_interval`,
`trajectory_interval`, `relax_fmax`, `quench`, `ensemble` and `pressure_GPa`. A profile with `ensemble = "NPT"` runs
at its constant pressure ([constant pressure](npt.md)); pressure schedules are refused. A profile's `relax_fmax` is
recorded only: `loupmd md` does not relax first. Its `quench` settings are used by
[`loupmd quench`](relax.md#quenching-a-finished-run). From Python, `lm.md(structure, model=...,
**lm.protocols.PROFILES["fast"].md_options())` runs a profile's schedule and ensemble, without its cadences.

## Neighbour skin

`skin="auto"` (the default of `lm.md` and `loupmd md`) chooses the neighbour-list skin from how far the atoms move
in one step: eight times the largest step with a margin, on a 0.05 A grid from 0.5 to 1.5 A. It is first estimated
from the target temperature and the lightest atom, then corrected once from what the engine measured after 200 steps
near the run's highest target. `--skin A` fixes it; `lm.Simulation` takes a fixed 0.5 A unless given `skin="auto"`.
The engine checks its list against the skin in every step, so the skin decides speed only, never the result.

## Output

The output directory (default `<structure stem>.loupmd/`, or `loupmd-run/` for an in-memory `Atoms`) receives:

- `thermo.<n>.csv`: rows every `--thermo-every` steps (default 20) with step, time, phase, target, temperature and
  potential, kinetic and total energy per atom;
- `trajectory.<n>.traj`: ASE trajectory frames every `--trajectory-every` steps (default 20; 0: none);
- `restart.npz` and `simulation.json`: the restart, every `--restart-every` steps (default 10,000; 0: only at the
  end), see [restarts](restart.md);
- `metadata.json`: the run's record, including the arguments it was called with.

Every restart point closes a chunk and starts the next pair of files `<n + 1>`. A new run refuses a directory that
already holds chunks. From Python the cadences are an `lm.Output("run/", thermo_every=20, trajectory_every=20,
restart_every=10_000)` passed as `output=`; `loupmd.output.read_thermo` and `read_frames` join the chunks.
[The output reference](../reference/output.md) lists the files and columns.

## The accuracy gate

`check="gate"` (`--check gate`) runs the accuracy gate on rattled copies of the structure before the first step:
energy, forces and stress against MACE in float64 must be within three times MACE float32's own deviation, or below
0.01 meV/atom, 1e-5 eV/A and 1e-7 eV/A^3. A failed gate raises `GateFailed` and nothing runs. The gate needs the
export extra (torch, mace-torch); without it `check="gate"` raises `GateUnavailable`, and `check="allow_unavailable"`
records that and runs on. `check="skip"` is the default. `gate=loupmd.config.GateConfig(...)` sets the rattle
(0.05 A), the number of copies (2) and where the references run. A resumed run does not repeat the gate.

## Run files

`loupmd init md` takes the arguments of `loupmd md` and prints a commented TOML file of every key, with the given
ones filled in. `loupmd run FILE` runs it; nothing runs at `init`.

```bash
loupmd init md structure.cif --model omat0-medium --temperature 300 --steps 10000 > run.toml
loupmd run run.toml
```

```toml
structure = "structure.cif"
model = "omat0-medium"
timestep_fs = 1.0
seed = 3
initial_velocities = "maxwell_boltzmann"   # CSVR needs a structure that moves

[output]
directory = "run"
thermo_every = 10

[integrator]
type = "CSVR"
tau_fs = 100.0

[[schedule]]
kind = "hold"
temperature_K = 300.0
ps = 10.0
```

The keys are those of `lm.md`; unknown keys are refused by name. `[integrator]` takes any `lm.integrators` record by
`type` with its fields; `timestep_fs` and the temperature may be left to the keys above it (with a schedule, its
first target). `profile` and `[[schedule]]` exclude each other. The `arguments` block of a finished run's
`metadata.json` is the same table, so `loupmd run DIR/metadata.json` repeats a run (after its output directory is
changed: the old one holds the run).

## Stepwise control: `lm.Simulation`

```python
from ase.io import read

atoms = read("structure.cif")
ff = lm.load("omat0-medium")
sim = lm.Simulation(atoms, ff, integrator=lm.integrators.Langevin(temperature_K=300, friction_per_fs=0.002),
                    output=lm.Output("run/"), seed=1, skin="auto")
sim.run(5_000)                                 # returns a RunSummary: steps, ms_per_step, rebuilds, ...
sim.run(schedule=lm.protocols.ramp(300, 600, ps=10))
sim.save_restart()
sim.close()
```

`Simulation.step` counts every step of the run, and `sim.atoms` is the current structure with its energy and forces.
Nothing is exported before the first `run`. A schedule passed to `run` moves the thermostat's target; a second one
is refused while the first has steps left. `run(mc=...)` adds Monte Carlo moves ([Monte Carlo](monte-carlo.md)).
`sim.relax()` relaxes the starting structure at fixed cell before the first step ([relaxation](relax.md)).

Between two `run` calls `sim.update(...)` changes the run as one transaction: everything is checked, then applied
and evaluated once; if any part fails the run is left as it was. Its keywords:

| keyword | method | what changes |
|---|---|---|
| `positions` (A, `(N, 3)`) | `set_positions` | the positions; momenta stay |
| `momenta` (ASE units) | `set_momenta` | the momenta |
| `masses` (amu) | `set_masses` | the masses; momenta stay, as in ASE |
| `species` | `set_species` | the elements, within those of the run's bundle (`elements=` adds more); a changed atom takes the new element's mass |
| `total_charge`, `spin_multiplicity` | `set_charges` | OMOL-0's inputs; the model is loaded again. Per-atom charges are refused |
| `friction_per_fs` | `set_friction` | the Langevin or BAOAB friction |

Every change but the friction sets the engine's state anew; the thermo rows stay on their cadence. Before the first
`run` an event changes the starting structure. Events are refused while a run is going and after `close()`.
