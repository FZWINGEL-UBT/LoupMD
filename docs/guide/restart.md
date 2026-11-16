# Restarts and resuming a run

A run with an output directory writes a restart at every restart point. `Simulation.resume` and `loupmd run --resume`
continue the run from the last one, in the same directory. There is no `loupmd md --resume`.

## When a restart is written

- Every `restart_every` steps (`--restart-every N`, `lm.Output(restart_every=...)`; default 10,000).
  `--restart-every 0` (`restart_every=None`) writes none during the run.
- At the end of every `lm.md` and `loupmd md` call.
- Whenever `Simulation.save_restart()` is called. A `Simulation` without output needs a path:
  `sim.save_restart("restart.npz")`.

`close()` writes no restart. A run stopped by a non-finite energy, a diverged cell or a killed process resumes from
the last restart point before the stop.

## Files

- `restart.npz`: the step, the engine's state block, the seed, the model's SHA-256, atomic numbers, masses, cell,
  positions and velocities, the fixed atoms, the schedule's position, the thermostat's target, the Monte Carlo
  counters, and where the output chunks end.
- `simulation.json`: how to make the run again: the model (label or path, head, charge, spin multiplicity,
  SHA-256), the integrator record, the output cadences, the seed, the skin, the elements of the bundle, the
  execution mode (whole or tiled) and the dispersion record.

Both are written to a temporary file, fsynced and renamed, so a crash leaves the previous version in place. A
restart point is also a chunk boundary: the rows and frames up to it are closed and fsynced, and the next
`thermo.<n>.csv` and `trajectory.<n>.traj` are opened. A variable-cell run's restart also holds its reference
volume V0 ([constant pressure](npt.md)).

## Resuming

```python
import loupmd as lm

sim = lm.Simulation.resume("structure.loupmd")
sim.run(10_000)
sim.save_restart()
sim.close()
```

```bash
loupmd run --resume structure.loupmd               # what is left of the run's steps or schedule
loupmd run --resume structure.loupmd --steps 5000  # 5,000 more steps
```

`Simulation.resume(directory)` reads `simulation.json` and `restart.npz`, loads the model again (a label through the
label table, a file by its path with the recorded SHA-256) and checks everything before it changes anything.
`force_field=` passes a loaded model instead; `restart=` names another restart file. The returned `Simulation` is
at the restart's step and runs on with `run`, as a new one does.

`loupmd run --resume DIR` without `--steps` runs the rest of the schedule, or the rest of the steps the run was
started with; a run with nothing left is refused and asks for `--steps N`. It then writes a restart, updates
`metadata.json` (with the warning "resumed at step N") and prints the report.

## What a resume reproduces

- **The state:** positions, velocities, the cell and the integrator's own state (a barostat's momentum, for
  example) come back from the state block bitwise.
- **The noise:** the seed is the run's, and the thermostat's random stream goes on where it stopped. Monte Carlo
  attempts draw from a stream fixed by the seed and the step, so a resumed run makes the same attempts.
- **The schedule:** its position and target; the thermostat stays at the last target of a finished schedule.
- **The output:** rows and frames go on at the same cadences. The chunks after the restart point (written by the
  stopped run after its last restart) are deleted, and the run continues in the next chunk, so the joined thermo
  file and trajectory have no gap and no duplicate.
- **The plan:** the execution mode recorded (whole cell or tiles); a tiled restart cannot be read by the whole-cell
  engine, nor the reverse ([large cells](large-cells.md)).

On the test engine of `tests/cpu` a run stopped and resumed writes the same rows and frames as one run straight
through. On the GPU, float32 sums differ in their last bits from run to run, so a resumed run follows the
uninterrupted one closely rather than bitwise. In `tests/gpu`, 500 NVE steps, a restart and 500 more end within
1e-3 A of 1,000 steps in one go. A Langevin state exported after 50 steps and imported into a new engine stays
within 1e-4 A of the uninterrupted run over 100 more steps, while a run restarted from its positions and velocities
alone, with the noise stream started anew, is far off.

A resume does not run the accuracy gate again. It keeps the skin recorded in `simulation.json`. A dispersion term
with `long_range="auto"` repeats its setup check, so it may choose another method when two are close in time.

## What must match

`RestartMismatch` is raised, and nothing in the directory is changed, when:

- `restart.npz` or `simulation.json` is missing, or either has a format this version does not read
  (`loupmd.restart/1`, `loupmd.simulation/1`);
- the model cannot be loaded again (pass `force_field=`), or its SHA-256 is not the one the restart was written
  with;
- the state block does not fit the restart's own record (another number of atoms, for example);
- an output chunk up to the restart point is missing, or the chunks hold another number of rows or frames than the
  restart recorded (they were changed after it was written);
- the chunks written after the restart cannot be deleted (a read-only directory or file).

The directory may be moved: a resume writes into the directory it was given. A new run, as opposed to a resume,
refuses a directory that already holds output chunks.

[The errors reference](../reference/errors.md) lists every error with its message.
