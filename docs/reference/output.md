# Output directory

A run with output writes into one directory. `lm.md` and `loupmd md` default to `./<structure stem>.loupmd/` for a
structure file and `./loupmd-run/` for an `ase.Atoms`; `--output DIR` or `output=` names another. `lm.Simulation`
writes only with `output=lm.Output(directory, ...)`. The writers are `loupmd/output.py`, `loupmd/simulation.py` and
`loupmd/provenance.py`. File names and formats are FastMACE's.

| File | Format | Written by | When |
| --- | --- | --- | --- |
| `thermo.<n>.csv` | CSV with a header row | every run with output | rows every `thermo_every` steps, in chunks |
| `trajectory.<n>.traj` | ASE trajectory | every run with output | frames every `trajectory_every` steps, in chunks |
| `restart.npz` | NumPy `.npz` (no pickles), format `loupmd.restart/1` | every run with output | every `restart_every` steps and at `save_restart()` |
| `simulation.json` | JSON, format `loupmd.simulation/1` | every run with output | at the start; again after `Simulation.update` changes the integrator, the state or the composition |
| `metadata.json` | JSON, schema `loupmd.metadata/1` | `lm.md`, `loupmd md`, `loupmd run` | at the end of the run |

The one-screen report is printed to standard output by `loupmd md` and `loupmd run`, and is `RunResult.summary`
of `lm.md`. It is not written to a file; it is made from `metadata.json`, so the two agree.

`metadata.json`, `simulation.json` and `restart.npz` are written atomically: a temporary file next to the target is
fsynced and renamed over it, then the directory is fsynced. A crash leaves the previous version in place.

## Cadences

`lm.Output` holds the directory and the cadences, in MD steps (FastMACE's record and defaults):

| Field | Default | `loupmd md` flag | Meaning |
| --- | --- | --- | --- |
| `directory` | required | `--output DIR` | the output directory, kept as given |
| `thermo_every` | 20 | `--thermo-every N` | thermo rows every N steps; always written |
| `trajectory_every` | 20 | `--trajectory-every N` | frames every N steps; `None` (`0` on the command line and in a run file): no trajectory |
| `restart_every` | 10000 | `--restart-every N` | restart points every N steps; `None` (`0` on the command line and in a run file): no periodic restart |

Steps count from the start of the run (`Simulation.step`), across resumes. A shipped profile (`--profile`) sets its
own cadences.

- The starting structure gives the first row (step 0) and, with a trajectory, the first frame.
- Rows fall on multiples of `thermo_every`; frames on multiples of `trajectory_every`.
- With a schedule, the end of each ramp or hold also gets a row (when it falls between two thermo steps) and a frame.
- Restart points fall on multiples of `restart_every`. `lm.md`, `loupmd md` and `loupmd run` also write one at the
  end of the run, so `restart_every=None` means "only at the end" there. `lm.Simulation` alone writes one at the end
  only when `save_restart()` is called.

## Chunks

Thermo rows and trajectory frames are written in chunks: `thermo.0.csv` and `trajectory.0.traj`, then
`thermo.1.csv` and `trajectory.1.traj`, and so on. Every restart point is a chunk boundary: the current pair is
closed and fsynced, and the next pair is opened. Each chunk is complete on its own: every thermo chunk starts with
the header, and a chunk without rows or frames still has both files.

The restart file stores `output_chunk` (the last closed chunk), `output_rows` and `output_frames` (the rows and
frames in chunks 0 to `output_chunk`). A resume checks those chunks before it changes anything, deletes the later
chunks and continues with chunk `output_chunk + 1`. Missing or changed chunks raise `RestartMismatch`. Nothing is ever
rewritten. A new run refuses a directory that already holds chunks (`InputError`).

Rows and frames go to disk on a background thread while the engine runs the next steps. The device writes thermo
rows inside the captured step and the host reads them once per chunk of steps.

## Thermo rows

Columns of `thermo.<n>.csv` (`THERMO_HEADER`):

| Column | Unit | Meaning |
| --- | --- | --- |
| `step` | | step number from the start of the run |
| `time_ps` | ps | `step` times the timestep |
| `phase` | | the schedule's phase at this step; `constant` for a run without a schedule |
| `target_K` | K | the thermostat's target at this step; `nan` for NVE |
| `temperature_K` | K | instantaneous temperature, 2 E_kin / (N_f k_B) |
| `potential_eV_atom` | eV/atom | potential energy per atom |
| `kinetic_eV_atom` | eV/atom | kinetic energy per atom |
| `total_eV_atom` | eV/atom | potential plus kinetic energy per atom |

A variable-cell run (NPT) adds two columns (`NPT_COLUMNS`):

| Column | Unit | Meaning |
| --- | --- | --- |
| `volume_A3_atom` | A^3/atom | cell volume per atom |
| `pressure_GPa` | GPa | minus a third of the stress trace plus the ideal-gas part 2 E_kin / (3 V) |

N_f is the number of degrees of freedom: 3 N, less 3 with the centre of mass at rest and less 3 per fixed atom
([Integrators](integrators.md)). A non-finite value in any column but `target_K` stops the run with `NonFiniteError`
before the row is written.

## Trajectory frames

`trajectory.<n>.traj` is an ASE trajectory. Each frame is an `Atoms` with positions (A), momenta (from the
velocities and masses), the cell and boundary conditions, and a single-point calculator holding the potential energy
(eV) and forces (eV/A). A variable-cell run's frames carry their cell. After a species change
(`Simulation.update`, Monte Carlo transmute moves) the frames carry the new elements and masses.

## `restart.npz`

A NumPy archive, read with `allow_pickle=False`. Arrays:

| Key | Content |
| --- | --- |
| `format` | `loupmd.restart/1` |
| `step` | step of the restart point |
| `blob` | the engine's exported state block |
| `model_sha256` | SHA-256 of the model file |
| `seed` | the noise seed |
| `numbers`, `masses`, `pbc`, `cell`, `fixed` | atomic numbers, masses, periodicity, cell (A), fixed-atom mask |
| `positions`, `velocities` | positions (A) and velocities (ASE units) |
| `potential_energy` | eV |
| `temperature_K` | the thermostat's temperature (`nan` without one) |
| `schedule` | JSON of the schedule's position, or `null` |
| `mc_counters` | JSON of the Monte Carlo counters by move kind |
| `output_chunk`, `output_rows`, `output_frames` | the chunk indices above (`-1`, 0, 0 without output) |
| `reference_volume_A3` | variable-cell runs only: the reference volume (A^3) |

`save_restart(path)` writes a restart elsewhere; a simulation without an output directory needs the path.

## `simulation.json`

The run record that a resume reads to rebuild the simulation. Keys:

| Key | Content |
| --- | --- |
| `format` | `loupmd.simulation/1` |
| `loupmd` | package version |
| `model` | `label`, `path`, `head`, `sha256`, `charge`, `spin_multiplicity` |
| `integrator` | the `IntegratorConfig` fields ([Integrators](integrators.md)) |
| `output` | the `lm.Output` fields |
| `seed`, `skin`, `atoms`, `elements` | noise seed, neighbour-list skin (A), atom count, the bundle's atomic numbers |
| `capacity` | `mode` (`whole`, `tiled` or `wavefront`), `receivers`, `memory_fraction` |
| `dispersion` | the dispersion entry (below), or `null` |

## `metadata.json`

What the run was, enough to repeat it. `loupmd run metadata.json` repeats a run from its `arguments`. Top-level keys
(`provenance.collect`):

| Key | Content |
| --- | --- |
| `schema` | `loupmd.metadata/1` |
| `created` | UTC time, ISO 8601 |
| `status` | the simulation's status at the end |
| `loupmd` | `version`, `git_commit` (`null` for an installed package) |
| `software` | `python`, `numpy`, `ase`, `platform` |
| `gpu` | `index`, `name`, `compute_capability`, `driver` |
| `model` | `label`, `file`, `path`, `sha256`, `head`, `licence`, `functional`, `charge`, `spin_multiplicity` |
| `bundle` | file name of the `.loup` bundle |
| `structure` | `source`, `repeat`, `atoms`, `elements` (symbol: count), `fixed_atoms`, `cell`, `pbc` |
| `integrator` | the `IntegratorConfig` fields |
| `schedule` | the resolved schedule (segments with `kind`, `phase`, `start_K`, `end_K`, `ps`, `steps`), or `null` |
| `output` | the `lm.Output` fields |
| `seed`, `skin_A`, `skin` | noise seed; skin in A; the skin record (`mode` `auto` or `fixed`, with the estimate and the measured largest step for `auto`) |
| `plan` | `execution` (`whole` or `tiled`), `precision`, `kernels` (`generic` or `specialised`), `kernel_signature`, `interaction_layers`, `r_max_A`, `memory`, `library`; tiled runs add `receivers`, `tiles`, `integrator_on` (`device` or `host`), `wavefront`, `requested` (the `capacity` mode asked for) and `tiled_because` (`asked` or `memory`). `memory` is the capacity plan's record in bytes: `whole_bytes`, `free_bytes`, `tiled_global_bytes`, `tile_workspace_bytes`, `host_bytes` (pinned host memory) with `host_lists` and `host_state` (whether the tile lists and the layer state are there) |
| `dispersion` | the dispersion entry (below), or `null` |
| `gate` | `status` (`skipped`, `passed`, `failed`, `unavailable`) and the gate's record |
| `step` | the last step |
| `counters` | `steps`, `rebuilds`, `pipelined_steps`, `dangerous_builds` (always 0: the engine checks the neighbour list at every step) |
| `mc` | Monte Carlo `moves` and `counters`, or `null` |
| `timing` | `setup_s`, `wall_s`, `ms_per_step`, `engine_ms_per_step`, `ns_per_day`, `atom_steps_per_s` |
| `arguments` | the arguments of `lm.md` (the run file's table) |
| `warnings` | list of strings |

`provenance.validate(metadata)` lists the problems of a record against the schema; an empty list means valid.

### The `dispersion` entry

With a dispersion correction ([Dispersion](../guide/dispersion.md)), `metadata.json` and `simulation.json` hold the
`lm.Dispersion` record's fields except `preset`: `method`, `damping`, `functional`, `parameters`, `three_body`,
`three_body_cutoff_bohr`, `pair_cutoff_bohr`, `cn_cutoff_bohr`, `smoothing`, `long_range`, `mesh_spacing_A`,
`update_every`, `auto_methods`. Added keys:

| Key | Content |
| --- | --- |
| `parameter_source` | where the parameters come from, `given by the user`, or `null` |
| `long_range_chosen_by` | `automatic, at setup` for `long_range="auto"`, else `the user` |
| `long_range_used` | the summation method the run uses |
| `long_range_choice` | the setup check of `long_range="auto"` (below); absent when the method was given |
| `cost` | `ms_per_evaluation` on the starting structure, and `three_body_ms` with the three-body term |

`long_range_choice` holds:

| Key | Content |
| --- | --- |
| `reference` | the reference the candidates are checked against (the split sum at tight settings), or `null` when the engine is not the library's |
| `bound` | the plain sum's deviations from the reference, which a candidate may not exceed (present when candidates were checked) |
| `candidates` | one row per method: `name`, `long_range`, `pair_cutoff_bohr`, `checked`, `kept`, and, when checked, `mesh_spacing_A`, `ms`, `energy_meV_per_atom`, `max_force_eV_per_A`, `stress_GPa`; `why` for a dropped or unchecked one |
| `left_out` | methods not tried, each with `method` and `why` |
| `chosen` | the name of the method used |
| `setup_s` | seconds the check took (present when candidates were timed) |

## Reading the output back

| Function | Returns |
| --- | --- |
| `loupmd.output.read_thermo(directory)` | `(header, rows)`: the header and every thermo row over all chunks, in order; rows are lists of strings |
| `loupmd.output.read_frames(directory)` | every trajectory frame over all chunks, in order, as `Atoms` with momenta, energy and forces |
| `loupmd.output.chunk_files(directory)` | every chunk file, by chunk index |
| `ase.io.read("trajectory.0.traj", index=":")` | the frames of one chunk |
| `lm.Simulation.resume(directory)` | a `Simulation` continued from `restart.npz` ([Restart](../guide/restart.md)) |

```python
import json
from loupmd.output import read_frames, read_thermo

header, rows = read_thermo("run")
temperatures = [float(row[header.index("temperature_K")]) for row in rows]
frames = read_frames("run")
metadata = json.load(open("run/metadata.json"))
print(metadata["plan"]["execution"], metadata["timing"]["ns_per_day"])
```

`loupmd run --resume DIR` continues a run from its restart and rewrites `metadata.json` at the end.

## Quench output

`loupmd quench DIR` writes into `DIR/quench/` (`--name` changes the folder): `settings.json`, `captured.traj` (the
picked MD frames), `quenched.traj` (their minima), `quench.csv` (one row per frame) and `quench.json` (settings and
summary, written last). See [Command-line reference](cli.md).
