# Coming from FastMACE

LoupMD keeps FastMACE's argument names, units, file names and run-file keys wherever a feature exists in both. A
FastMACE script needs the changes on this page and no others; `tests/cpu/test_from_fastmace.py` runs FastMACE's
quickstart with exactly these changes.

## The import line and the command

| FastMACE | LoupMD |
| --- | --- |
| `import fastmace as fm` | `import loupmd as lm` |
| `fastmace md`, `fastmace run`, `fastmace doctor`, ... | `loupmd md`, `loupmd run`, `loupmd doctor`, ... |
| `fastmace init FLAGS > run.toml` | `loupmd init md FLAGS > run.toml` |
| `from fastmace.ase import Langevin` (and the other drop-ins) | `from loupmd.ase import Langevin` |
| `from fastmace.io.output import read_frames, read_thermo` | `from loupmd.output import read_frames, read_thermo` |

Everything else in these lines stays: `lm.md(...)` and `lm.load(...)` take FastMACE's arguments,
`loupmd md --structure NaCl.cif` is accepted beside `loupmd md NaCl.cif`, `--dispersion d3bj --d3-functional NAME`
works as in FastMACE, and a run file's `schedule = "anneal"` sets the profile's schedule alone, as there (LoupMD's
`profile = "anneal"` also sets the profile's output cadences).

FastMACE asks you to import it before torch. LoupMD does not import torch while it runs, so the order does not
matter.

## Arguments that are refused

Each raises `PlanRefused` or `InputError`, and the message says what to use instead.

- `precision="float64"`, `allow_reference=` / `fallback=` and `Workload(reproducible=True)`: see
  [what LoupMD leaves out](not-in-loupmd.md).
- FastMACE's execution switches `mode=`, `runner=`, `engine=`, `neighbours=` and `calibrate=`: LoupMD has one
  execution path.
- `min_mc_temperature_K=`: run the Monte Carlo moves with `mc=` over the whole run, or split the schedule into runs
  with and without moves ([Monte Carlo](monte-carlo.md)).
- A CSVR thermostat on a structure at rest. FastMACE runs it, and the structure stays at rest, because CSVR rescales
  the velocities it finds. LoupMD refuses it: add `initial_velocities = "maxwell_boltzmann"` (or a temperature in K)
  to the run file, or draw velocities first.

## Files and caches

| | FastMACE | LoupMD |
| --- | --- | --- |
| Default output directory | `<structure stem>.fastmace/` | `<structure stem>.loupmd/` |
| Files in it | `thermo.<n>.csv`, `trajectory.<n>.traj`, `restart.npz`, `simulation.json`, `metadata.json` | the same names, chunks and cadences |
| Model folder | `FASTMACE_MODELS` | `LOUPMD_MODELS` |
| Cache root | `FASTMACE_CACHE` | `LOUPMD_CACHE` (default `~/.cache/loupmd`); see [environment](../reference/environment.md) |
| ASE drop-ins | write FastMACE's own files too (`run.fastmace/` for `trajectory="run.traj"`) | write LoupMD's own files only with `output=` |

- **Restarts are not interchangeable.** `restart.npz` and `simulation.json` carry the formats `loupmd.restart/1`
  and `loupmd.simulation/1`; `lm.Simulation.resume` refuses a FastMACE run directory with `RestartMismatch`.
  Continue such a run in FastMACE, or start a new LoupMD run from its last frame.
- **Caches are separate.** LoupMD exports each model to its own bundle format, so FastMACE's compiled models and
  CUDA-graph caches are not used; LoupMD's first start for a model writes its bundle once
  ([models](models.md)).

## What reads differently

- **The report.** Its lines name LoupMD's plan: the whole system or tiles, the skin, float32 evaluation with float64
  accumulation. FastMACE's tiers, backends and rewrites do not exist here.
- **`RunResult.plan`** is the recorded plan as a dict (`metadata["plan"]`). It has no `describe()`;
  `print(run.summary)` shows it.
- **`loupmd doctor`** checks the GPU and driver, the CUDA run-time libraries, `libloupmd` and its tensor-core
  kernels, the cache root, the model files and whether the optional export extra (torch, MACE, e3nn) is installed.
  LoupMD needs no torch, cuEquivariance, OpenEquivariance or Triton at run time.
