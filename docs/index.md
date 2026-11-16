# LoupMD

LoupMD runs molecular dynamics with pretrained MACE potentials on NVIDIA GPUs. The model, the neighbour list and
the integrator run in one C++/CUDA engine, and a whole MD step is replayed as a CUDA graph. The Python package is
the front end: it loads a model by label, exports it to the engine's format, checks it against PyTorch MACE, and
writes the trajectory and a report.

```python
import loupmd as lm

result = lm.md("structure.cif", model="omat0-medium", temperature_K=300, steps=10_000, output="run")
print(result.summary)
```

The same run from the command line is `loupmd md structure.cif --model omat0-medium --temperature 300 --steps 10000`;
the [quickstart](quickstart.md) shows it with the CLI, Python and ASE side by side.

## What it does

- NVT, NVE and NPT molecular dynamics with the thermostats and barostats of ASE, on the GPU
  ([integrators](reference/integrators.md)).
- Relaxation of positions and of the cell ([relaxation](guide/relax.md)).
- Monte Carlo swaps, transmutations and displacements between MD steps ([Monte Carlo](guide/monte-carlo.md)).
- Cells too large for the GPU, split into tiles ([large cells](guide/large-cells.md)).
- The D3 dispersion correction ([dispersion](guide/dispersion.md)).
- An ASE calculator and drop-in ASE MD classes ([ASE](guide/ase.md)).

## Accuracy

Every model runs in float32 on the engine. Before a run, LoupMD can compare energies, forces and stress with
PyTorch MACE in float64 on rattled copies of your structure: the accuracy gate allows three times the error of
float32 MACE. See [models](guide/models.md) for the supported families and the gate.

## Coming from FastMACE

LoupMD keeps FastMACE's names and arguments wherever a feature exists in both.
[Coming from FastMACE](guide/from-fastmace.md) lists the few changes a FastMACE script needs, and
[what LoupMD leaves out](guide/not-in-loupmd.md) the FastMACE features that LoupMD refuses.
