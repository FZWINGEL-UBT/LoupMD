# Questions and answers

## Which GPUs and systems does LoupMD run on?

NVIDIA GPUs of compute capability 8.0 or newer, on Linux. The CUDA
runtime, cuBLAS and cuFFT come from NVIDIA's Python wheels, so no CUDA toolkit is needed except for
`loupmd build-kernels`. Tensor-core GEMM kernels ship for compute capabilities 8.x, 9.x and 12.x. `loupmd doctor`
checks the GPU, the driver, the libraries, the cache and the model files, and changes nothing. See
[install](install.md).

## Is float32 accurate enough?

Every model runs in float32, with float64 accumulation. `loupmd validate --model LABEL` compares energy, forces and
stress with MACE in float64. A quantity passes when LoupMD's deviation is at most three times MACE float32's own
deviation, or below 0.01 meV/atom, 1e-5 eV/A and 1e-7 eV/A^3. `check="gate"` (`--check gate`) runs the same rule on
your structure before a run. float64 forces are not offered. See [models](guide/models.md).

## Why do two runs with the same seed differ in the last digits?

The seed fixes the thermostat's noise, but float32 atomic sums on the GPU differ in their last bits from run to
run. Bitwise-reproducible runs are not in LoupMD. A resumed run continues
from its saved state ([restarts](guide/restart.md)).

## Why is the first run of a model slow?

Several things happen once and are then cached or reused:

- **The export.** The checkpoint is converted to a `.loup` bundle for the structure's elements, in a separate process
  with torch. The bundle is kept in `<cache root>/bundles/`; a new element set, head or charge gives a new bundle.
- **GEMM plans.** The engine times its matrix products for each new shape on your GPU and stores the winners in
  `plans_*.bin` files, keyed by GPU, cuBLASLt version and library build.
- **Per-system tuning.** Some kernel variants are timed at the first neighbour-list build of a system. The first
  step of a tiled run tunes the GEMMs for the tile shapes, 1.5 to 5 s ([large cells](guide/large-cells.md)).
- **Setup checks.** `capacity="auto"` probes one tile to plan the memory, and `dispersion="d3bj"` times its candidate
  methods on the starting structure ([dispersion](guide/dispersion.md)).

## Where is the cache, and how do I clean it?

The cache root is `LOUPMD_CACHE`, default `~/.cache/loupmd` (`LOUP_CACHE_DIR` is accepted as an older name). It
holds `models/`, `bundles/`, `plans/`, `gate_refs/` and `lib/`, and must lie outside the Python installation. The
engine writes its GEMM plan files into the root.

```bash
loupmd cache size
loupmd cache list --kind bundles
loupmd cache prune --older-than 30 --dry-run
loupmd cache prune --max-size 5G
```

`prune` removes bundles, plans and gate references by default, which LoupMD makes again when needed. Downloaded
models and built kernel plugins are removed only when named with `--kind models` or `--kind lib`. See
[environment](reference/environment.md).

## The cell does not fit the GPU. What now?

With `capacity="auto"` (the default) a cell that does not fit in 90% of the free GPU memory runs in tiles of up to
65,536 atoms. The tiled mode runs NVE and Langevin; other thermostats and the barostats are refused there.
`capacity="whole"` forces the whole cell and fails with a `CapacityError` when it does not fit. See
[large cells](guide/large-cells.md).

## The run stopped with a non-finite energy. What happened?

A `NonFiniteError` means the energy or forces became NaN or infinite. LoupMD never patches such values: the run
stops, and what was written before stays. Check the structure and the timestep, then resume from the last restart
point. In NPT a `CellDivergedError` means the cell left its limits.
See [errors](reference/errors.md).

## How do I continue a run?

A run writes `restart.npz` to its output directory every 10,000 steps (`--restart-every`) and at the end.

```bash
loupmd run --resume structure.loupmd --steps 50000
```

In Python: `sim = lm.Simulation.resume("structure.loupmd")`, then `sim.run(steps=...)`. A restart of another model,
element set or atom count raises `RestartMismatch` before anything changes. A whole-cell restart cannot be read by
the tiled engine, nor the reverse. See [restarts](guide/restart.md).

## Why is dispersion expensive?

D3 sums over every pair within 40 bohr (21 A), many more pairs than the model's 6 A cutoff. On small cells the term
can cost as much as the model. The three-body term costs many times the pair term at 40 bohr. See
[dispersion](guide/dispersion.md).

## Can I run a molecule or a slab?

Only in a periodic cell. LoupMD needs `pbc=True` in all three directions: put the molecule or slab in a box with
vacuum of at least the model's cutoff on each open side.

## Can I use my own fine-tuned model?

Yes. A file outside the label table needs `sha256=` or `trusted=True`, because a checkpoint is a pickle and runs
code when it is loaded. If its architecture needs kernels the library was not built with, LoupMD says so and
`loupmd build-kernels MODEL` compiles them. See [models](guide/models.md).

## Why is MD through the ASE calculator slow?

ASE calls the calculator once per step and moves the atoms on the host, so each step copies data between host and
GPU. Use `lm.md`, `lm.Simulation` or the drop-in classes of `loupmd.ase` (`Langevin`, `VelocityVerlet`, ...),
which keep ASE's arguments and let the engine run the loop. See [ASE](guide/ase.md).

## How does LoupMD differ from FastMACE?

LoupMD keeps FastMACE's names and arguments wherever a feature exists in both. It refuses float64 forces, the
fallback to the MACE calculator, bitwise-reproducible runs and FastMACE's execution switches (`mode=`, `runner=`,
`engine=`, `neighbours=`, `calibrate=`), each with a message naming what to use instead. In Monte Carlo it refuses
`min_mc_temperature_K=`, the swap velocity treatments other than `"exchange"` and the adaptive displacement step.
See [what LoupMD leaves out](guide/not-in-loupmd.md).
