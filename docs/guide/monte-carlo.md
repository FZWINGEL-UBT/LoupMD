# Monte Carlo between MD steps

`lm.mc` has three Monte Carlo moves: `Swap`, `Transmute` and `Displace`. They run every few MD steps of a run, or as
blocks between two `run` calls. Each attempt costs one energy-only pass of the engine (no gradients), and the
Metropolis rule is applied on the host.

```python
import loupmd as lm

moves = [lm.mc.Swap(pairs=[("Cu", "Au")], every=100, attempts=20),
         lm.mc.Displace(delta_A=0.1, every=50, attempts=10)]
result = lm.md("alloy.cif", model="omat0-medium", temperature_K=600, steps=20_000, mc=moves)
print(result.metadata["mc"]["counters"])
```

## The moves

Species are element symbols or atomic numbers. Every move has `every` (MD steps between blocks, default 1) and
`attempts` (attempts per block, default 1). Fixed atoms (ASE's `FixAtoms`) are never chosen.

| Move | What one attempt does | Accepted with |
|---|---|---|
| `Swap(pairs)` | picks a pair type uniformly from `pairs`, then one atom of each species uniformly, and exchanges them | min(1, exp(-dE / kT)) |
| `Transmute(species, mu=None)` | picks an atom of the set uniformly, then another element of the set uniformly | min(1, exp(-(dE - dmu) / kT)) |
| `Displace(delta_A)` | moves one atom uniformly within a sphere of radius `delta_A` (A) | min(1, exp(-dE / kT)) |

**Swap.** The two atoms exchange positions and velocities (FastMACE's `velocities="exchange"`, the only choice).
Each atom keeps its index, element and mass, so the engine needs no new setup. The configuration is the one a
species swap gives; only the atoms' labels differ. A pair whose species are not both in the structure is skipped; a
`Swap` with no usable pair is refused.

**Transmute** (semi-grand canonical). `mu` maps species to configurational chemical potentials in eV; a species
left out has 0, and dmu is mu(new) - mu(old). The atom takes the new element's standard mass, and its velocity is
rescaled to keep its kinetic energy (`velocities="rescale"`, the only choice). Each attempt sets the engine's system
for the trial composition; that time is counted in `setup_s`. Every element of `species` must be in the run's
bundle: `lm.md` adds them itself, and `lm.Simulation` takes them as `elements=[...]`.

**Displace.** The step is fixed, which keeps detailed balance; `adaptive=True` is refused. Choose `delta_A` from a
short trial run; about 30% acceptance is usual.

## Moves during a run

`lm.md(mc=...)` and `Simulation.run(steps, mc=...)` take one move or a list. After every `move.every`-th step,
counted by `Simulation.step`, the move makes `move.attempts` attempts at the thermostat's target temperature at that
step (with a schedule, the schedule's target there). The moves run in their list order, before that step's trajectory
frame and restart point.

- An NVE run has no temperature, so `mc=` is refused there.
- The random stream of each block follows from the run's seed, the step and the move's place in the list. A resumed
  run makes the same attempts as one that was not interrupted.
- `lm.md` refuses FastMACE's `min_mc_temperature_K=`. To leave out part of a schedule, split it into
  `Simulation.run` calls with and without `mc=`.

In a run file the moves are a list of tables, each naming its move:

```toml
[[mc]]
move = "swap"
pairs = [["Cu", "Au"]]
every = 100
attempts = 20
```

The command line has no flag for moves; use a run file with `loupmd run` ([MD](md.md)).

## Blocks between runs

`lm.mc.swap`, `lm.mc.transmute` and `lm.mc.displace` make one block of attempts on a `Simulation` that is not
running:

```python
sim = lm.Simulation(atoms, lm.load("omat0-medium"), integrator=lm.integrators.Langevin(
    timestep_fs=1.0, temperature_K=600, friction_per_fs=0.01), output=lm.Output("run"))
sim.run(steps=1000)
block = lm.mc.swap(sim, lm.mc.Swap(pairs=[("Cu", "Au")], attempts=500), seed=1)
print(block.counters.acceptance, block.energy_eV)
sim.run(steps=1000)
```

Each takes `temperature_K=` (default: the thermostat's), `seed=` and `attempts=` (default: the move's). It returns
the block's counters and the potential energy after it, in eV. The state the chain ends in is set on the engine
once; with nothing accepted the engine's state is put back unchanged.

## Counters

Each kind of move has counters: `attempted`, `accepted`, `rejected_nonfinite` (trial energies that were not finite;
rejected, never patched) and `setup_s` (transmute only). `acceptance` is accepted / attempted.

They are summed over the run in `Simulation.mc_counters` (by `"swap"`, `"transmute"`, `"displace"`), in
`metadata.json` under `mc` with the moves themselves, and in the restart file, so a resumed run continues them.
`RunResult.metadata["mc"]` holds the same. See [output](../reference/output.md).

## Dispersion

With `dispersion=` the D3 term is part of every energy-only pass, so the Metropolis rule sees the model plus D3.
The term walks every pair within 40 bohr, and on small cells it can cost as much as the model, so each attempt
costs more ([dispersion](dispersion.md)).
