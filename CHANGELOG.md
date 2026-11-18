# Changelog

## Unreleased

- The export extra installs torch 2.15.0 and mace-torch 0.3.17 on Python 3.11 and newer. Python 3.10 keeps torch
  2.14.1 and mace-torch 0.3.16, the last versions that support it. With torch 2.15 the CUDA libraries are those of
  cuda-toolkit 13.2.2 (13.0.3 on Python 3.10).
- mace-torch 0.3.17 is the first whose stress for the MACE-POLAR-1 models has the cell dependence of the
  long-range term. LoupMD's stress has always had it: against mace-torch 0.3.16 the two differed by about 0.02 GPa
  on a water box, against 0.3.17 they agree.
- The export extra installs `graph_longrange`, which MACE's POLAR-1 models are built on. With 0.0.1 a `polar1-*`
  model could not be exported after a plain `pip install "loupmd[export]"`. An environment without the package is
  now told what to install.

## 0.0.1 (18 November 2026)

The first public release of LoupMD (`PLAN.md` Stages 0 to 7).

- The C++/CUDA engine with the CUDA-graph MD step, the generic kernels for every supported MACE family and the
  specialised kernels for MACE-OMAT-0 medium; one library for compute capability 8.0 to 12.0 in a
  `manylinux_2_28` wheel that needs no PyTorch at run time.
- `lm.md`, `lm.Simulation`, `lm.load` and the `loupmd` command (`md`, `init`, `run`, `relax`, `quench`, `bench`,
  `validate`, `models`, `doctor`, `cache`, `dispersion`, `build-kernels`); run files and restarts.
- NVT (Langevin, CSVR, Nose-Hoover chain, Berendsen), NVE and NPT (Langevin BAOAB, MTK, Berendsen) as ASE 3.27's
  classes; temperature schedules and profiles.
- Relaxations with ports of ASE's LBFGS (on the GPU at fixed cell), FIRE and `FrechetCellFilter`; quenches of
  annealing runs.
- Monte Carlo swaps, transmutations and displacements.
- Cells larger than the GPU, run in tiles.
- The D3 dispersion correction (BJ, zero, BJM and zerom damping; three-body term; plain, split and automatically
  chosen long-range sums), in MD, NPT, Monte Carlo, the tiled execution, relaxations and the ASE calculator.
- The accuracy gate against PyTorch MACE in float64, and `loupmd validate`.
- An ASE calculator and drop-in ASE MD and optimiser classes.
- Documentation site, examples, and "Coming from FastMACE".
- Licences: MIT for LoupMD's own code; LGPL-2.1-or-later, as ASE, for the code that follows ASE (the optimisers,
  and the engine's thermostats, barostats and minimiser), so the compiled engine library is LGPL-2.1-or-later as a
  whole; BSD-3-Clause for the tensor-core GEMM kernels (`NOTICE.md`).
