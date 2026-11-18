# Errors

Every refusal LoupMD makes itself is a `LoupError` (`loupmd/errors.py`). Errors from underneath pass through
unchanged: a failed download raises `OSError`, and an unknown keyword to an ASE drop-in class raises `TypeError`.
All classes are importable from the top-level package: `lm.InputError`, `lm.PlanRefused`, and so on. A refused
setting is never replaced by another silently; the error says what was refused and why.

## Message shape

A `LoupError` is made as `LoupError(what, why=None, next_step=None)`. Its message has up to three lines:

```text
<what happened>
Why: <the reason, in a sentence or two>
Next: <what to do next>
```

The `Why:` line is left out when there is no reason, and the `Next:` line when there is no next step. Many classes
have a default next step (`default_next_step`), used when the raise gives none. The parts are also attributes:
`error.what`, `error.why`, `error.next_step`. Errors can be pickled, so they cross process boundaries intact.

Example (a checkpoint that does not match its label; path and hashes shortened):

```text
/data/mace-omat-0-medium.model has SHA-256 3f..., but label 'omat0-medium' is the release with SHA-256 d4b1...
Why: the file is not the released checkpoint of the label, so it was not unpickled
Next: Download the release again (download=True), or pass trusted=True to load this file anyway.
```

## Classes

| Class | Also derives from | Extra attributes | Raised when |
| --- | --- | --- | --- |
| `LoupError` | `Exception` | `what`, `why`, `next_step` | base class; also raised itself where no subclass fits: a bundle that is not in the cache in an environment without the export extra (the message names the missing modules and `pip install 'loupmd[export]'`), a failed export, `loupmd build-kernels` without `nvcc` or the kernel sources, `loupmd bench` on a GPU that is in use |
| `InputError` | `ValueError` | | an argument is refused before the library reads it: a wrong type, shape or value, an unknown label, a missing checkpoint file, an unknown key in a run file, a directory that already holds a run |
| `CheckpointTrustError` | | | a checkpoint would be unpickled without a matching SHA-256 or `trusted=True`; a download whose hash differs (the file is deleted); a URL without a hash |
| `CacheLocationError` | | | the cache root lies inside the Python installation (`sys.prefix`) |
| `BackendError` | `RuntimeError` | `call` (the C function) | a call into `libloupmd` failed or was refused; `what` is the library's own message (`loup_last_error`). Also: the library cannot be loaded, or its C interface version differs from the package's |
| `UnsupportedModel` | | `signature` (the first missing kernel signature) | the library has no compiled kernels for the model's symmetric contraction, and no kernel plugin for it is in the cache |
| `ModelElementError` | | | the structure has an element the model has no embedding for; raised before anything is exported |
| `PlanRefused` | `ValueError` | `reasons` (a tuple) | the run cannot be planned as asked: an unsupported combination (for example a barostat with fixed atoms, `fixcm=True` with fixed atoms, an integrator the tiled mode does not run, a dispersion correction for a model whose data contains dispersion). `reasons` lists every reason at once; the message starts with `plan refused:` |
| `CapacityError` | | | the system does not fit the GPU in the execution mode asked for. Defined and exported; no code path of this version raises it |
| `GateFailed` | | `gate_record` | the accuracy gate on the user's structure failed: LoupMD's energy, forces or stress are further from MACE float64 than three times MACE float32's own deviation (`check="gate"` or `"allow_unavailable"`) |
| `GateUnavailable` | | `gate_record` | the gate's reference could not be computed: the export extra is missing, or no device has the memory for MACE float64 (`check="gate"`; with `"allow_unavailable"` the run goes on and records it) |
| `RestartMismatch` | | | a restart does not belong to this model, element set or atom count; the restart file or `simulation.json` is missing or of another format; the output chunks do not match the restart |
| `NonFiniteError` | | | a non-finite energy, force or thermo value stopped a run or a relaxation; it is never patched over |
| `CellDivergedError` | | `step` | an NPT run's cell left the range its integrator allows; `step` is the step whose cell was checked |

A resume checks the restart, the model and the output chunks before it changes anything, so a `RestartMismatch`
leaves the output directory as it was ([Restart](../guide/restart.md)).

### Default next steps

| Class | `Next:` when the raise gives none |
| --- | --- |
| `CheckpointTrustError` | Pass sha256= with the file's published hash, or trusted=True for a file you trust; a checkpoint is a pickle and runs code when loaded. |
| `UnsupportedModel` | Build a library with this model's kernels: loupmd build-kernels MODEL (needs nvcc), or use a model of a supported architecture. |
| `ModelElementError` | Use a model trained on that element, or remove it from the structure. |
| `PlanRefused` | Change the inputs each reason names; a refused setting is never replaced by another. |
| `CapacityError` | Let LoupMD choose the mode (capacity='auto'), which runs large cells tiled, or use a smaller cell or a GPU with more memory. |
| `GateFailed` | Read the failed quantities in the gate record, and report the model and structure. |
| `GateUnavailable` | Install the export extra (pip install 'loupmd[export]'), or run without the gate (check="skip"), which the run then records. |
| `RestartMismatch` | Resume with the model and structure the restart was written with. |
| `NonFiniteError` | Check the structure and the timestep, and resume from the last restart point. |
| `CellDivergedError` | The structure left its condensed phase or the barostat diverged: raise the pressure, lower the temperature, or couple the cell more slowly. No restart is written at the stop; the last one written before it holds an earlier state. |

`InputError`, `CacheLocationError` and `BackendError` have no default; their raises give the next step where there
is one.

## Catching errors

```python
import loupmd as lm

try:
    result = lm.md("structure.cif", model="omat0-medium", temperature_K=300, steps=10_000)
except lm.PlanRefused as error:
    for reason in error.reasons:
        print("refused:", reason)
except lm.LoupError as error:
    print(error)            # what, Why:, Next:
```

`InputError` and `PlanRefused` are also `ValueError`, and `BackendError` is also a `RuntimeError`, so code that
catches those built-in classes catches them too.

## Command-line exit codes

`loupmd` (`loupmd/cli/__init__.py`) returns:

| Code | Meaning |
| --- | --- |
| 0 | success |
| 1 | a LoupMD error: the message is printed to standard error as `error: <message>`, without a traceback |
| 2 | a usage error: no command, an unknown option or a missing argument (from `argparse`), or a command group without its action (`loupmd cache`, `loupmd init`) |

`--debug` shows the traceback of a LoupMD error instead of the short message. Some commands also return 1 when their
check fails without an exception: `loupmd validate` when the gate fails, `loupmd doctor` when a check fails,
`loupmd relax` when the relaxation did not converge, `loupmd quench` when a frame did not converge, and
`loupmd bench` when a system's temperature is not finite. See [Command-line reference](cli.md).
