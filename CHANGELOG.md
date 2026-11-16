# Changelog

## 0.1.0a1 (not yet released)

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
