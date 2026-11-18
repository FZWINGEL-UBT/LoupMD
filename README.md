# LoupMD

LoupMD runs molecular dynamics with pretrained [MACE](https://github.com/ACEsuit/mace) potentials on NVIDIA GPUs.
The model, the neighbour list and the integrator run in one C++/CUDA engine, and a whole MD step is replayed as a
CUDA graph. The Python package loads a model by label, exports it to the engine's format, checks it against
PyTorch MACE, and writes the trajectory and a report. The runtime needs no PyTorch and no CUDA compiler.

```python
import loupmd as lm

result = lm.md("structure.cif", model="omat0-medium", temperature_K=300, steps=10_000, output="run")
print(result.summary)
```

The same run from the command line:

```bash
loupmd md structure.cif --model omat0-medium --temperature 300 --steps 10000
```

## What it does

- NVT, NVE and NPT molecular dynamics with the thermostats and barostats of ASE, on the GPU.
- Relaxation of positions and of the cell (ASE's LBFGS and FIRE, and its `FrechetCellFilter`).
- Monte Carlo swaps, transmutations and displacements between MD steps.
- Cells too large for the GPU, split into tiles.
- The D3 dispersion correction, with a choice of damping, three-body term and long-range sum.
- An ASE calculator and drop-in ASE MD and optimiser classes.

Every model runs in float32 on the engine. Before a run LoupMD can compare energies, forces and stress with PyTorch
MACE in float64 on rattled copies of your structure (`check="gate"`, `loupmd validate`): the gate allows three
times the error of float32 MACE.

## Requirements

Linux x86_64 (glibc 2.28 or newer) or WSL2, an NVIDIA GPU of compute capability 8.0 or newer with a driver for
CUDA 13, and Python 3.10 or newer. Exporting a model the first time needs the `export` extra (PyTorch and MACE);
see `docs/install.md`.

This pre-release (0.1.0a1) has been verified on an RTX 5060 Ti (compute capability 12.0) only. The wheel also
holds code for compute capability 8.0, 8.6, 8.9, 9.0 and 10.0, which is untested in this release
(`docs/validation/release-0.1.0.md`).

## Documentation

The documentation is in `docs/` and builds with `mkdocs build --strict` (`pip install -e ".[docs]"`): install,
quickstart, guides, the command-line and Python reference, and "Coming from FastMACE". The runnable examples are in
`examples/`.

## Licence

LoupMD is MIT-licensed (`LICENSE`). The tensor-core GEMM kernels compiled from CUTLASS headers are BSD-3-Clause,
and the modules ported from ASE are LGPL-2.1-or-later; `NOTICE.md` lists every file of each, the origins of the
data, and the licences of the run-time dependencies. Each MACE model keeps its own licence, which `loupmd models`
prints.
