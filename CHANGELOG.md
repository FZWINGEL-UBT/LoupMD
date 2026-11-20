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
- Faster kernels for every model family. MD step times on an RTX 5060 Ti against 0.0.1's library (one A B B A
  round each, Langevin at 300 K): MACE-MP-0 medium on 1,080 atoms of SrTiO3 1.76 to 0.67 ms; MACE-POLAR-1-S on
  216 atoms of NaCl 1.30 to 1.12 ms; MACE-OMAT-0 medium on 5,000 atoms of SrTiO3 7.82 to 7.67 ms; MACE-OMAT-0
  small, MACE-OFF23 medium (water, 648 atoms) and MACE-MH-1 (NaCl, 216 atoms) 1.5 to 2.4% faster.
- One set of kernels. The kernels specialised to MACE-OMAT-0 medium are gone, since the generic kernels have
  overtaken them: `LOUP_GENERIC` and the `generic` argument of `loupmd.bundles` are ignored, and
  `loupmd validate --specialised` is refused. A bundle exported for the specialised kernels is refused with a
  message; the bundle cache exports it again by itself.
- LAMMPS. The folder `lammps/` of the repository holds the LAMMPS package ML-LOUPMD: `pair_style loupmd`,
  `pair_style loupmd/kk` for the KOKKOS package, and `run_style loupmd`, which lets the engine integrate between
  LAMMPS's output steps (`lammps/README.md`). It is built into LAMMPS and is GPL-2.0; it is not part of the wheel.
- LAMMPS, `run_style loupmd`: the engine runs its own integrator in place of the input's `fix nve`, `fix nvt`,
  `fix npt ... iso`, `fix langevin`, `fix temp/csvr`, `fix temp/berendsen` and `fix press/berendsen`, with
  `fix setforce 0 0 0` as held atoms and `fix momentum` as a centre of mass at rest. Such an input runs at the
  engine's own step time (measured on an RTX 5060 Ti: within 1% of the Python package's loop; LAMMPS integrating
  itself takes 1.04 to 1.27 times as long with `fix nvt` and 1.16 to 1.65 times with `fix npt`). A fix the engine
  has no integrator for is refused by name.
- An evaluation between two blocks of a run (`loup_evaluate` after `loup_run`, as at every output step) no longer
  makes the engine capture its step graphs again: output every 10 steps cost 0.92 ms per step on a 320-atom cell
  and costs 0.73 now.
- The C interface gains `loup_compute_device` (an evaluation on position and force arrays that stay on the GPU),
  `loup_cutoff`, `loup_species` and `loup_set_evaluate_ahead`; nothing else in it changed (interface version 1).
  `loup_evaluate` on an unchanged cell is one launch on the GPU, with no blocking transfer.
- A system too large for the GPU now fails with "out of memory" under WSL 2 as well, where the driver used to
  accept the allocations and page them out to system memory: a step then took minutes, and the card stayed full
  until the run was stopped. Measured on an RTX 5060 Ti: MACE-MH-1 on 98,415 atoms, whole, used to fill the card;
  now it is refused within about 2 s, and the caller can go on with `capacity="tiled"` (the MACE-POLAR-1 models,
  which are not tiled, with a smaller system). This concerns `capacity="whole"`, the LAMMPS pair style and the
  MACE-POLAR-1 models, which the capacity plan does not cover.
  An allocation is refused when it is larger than the card's free memory as NVML reports it.
- On RTX 50-series GPUs a system whose neighbour list is not rebuilt within 50 steps (a crystal near its lattice, a
  short run, LAMMPS's pair style under `fix npt`) now gets the faster convolution kernel as well: the kernel
  variants are timed then, instead of only at the first rebuild. MACE-OMAT-0 medium on 5,000 and 10,000 atoms of
  SrTiO3 from rest: 2.2% faster per step; MACE-MH-1: 0.8%.
- An engine freed in another thread no longer breaks a run. Python's garbage collector can free an engine in one of
  the package's output threads while another engine starts a run; that run could fail ("operation failed due to a
  previous error during capture"), the next engine could not be created, or the process crashed. An engine may now
  be freed in any thread while others run.
- Every bundle is exported again on its first use (the cache key changed, and a bundle now carries the source of
  its product kernels). A kernel plugin built by 0.0.1's `loupmd build-kernels` is not loaded: build it again.

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
