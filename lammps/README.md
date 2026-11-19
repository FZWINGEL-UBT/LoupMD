# LoupMD in LAMMPS

MACE models in LAMMPS through the LoupMD engine, on one CUDA GPU. The package `ML-LOUPMD` adds:

| style | what it does | when to use it |
|---|---|---|
| `pair_style loupmd` | LAMMPS integrates; positions and forces pass through host memory once per step | the default: any LAMMPS build, any fix |
| `pair_style loupmd/kk` | LAMMPS-Kokkos integrates; positions and forces stay on the GPU | Kokkos CUDA builds that also run other Kokkos styles |
| `run_style loupmd` | the engine integrates (NVE or Langevin) between LAMMPS's output steps | the engine's own step time; no fix may act during the steps |

Design and measurements: `LAMMPS_KOKKOS_PLAN.md` in the repository root.

## Quick start

```
# 1. the engine library and a model bundle (.loup) for your elements
make -j4 CUDA_ARCH=sm_120                      # in the LoupMD checkout: libloupmd.so
python python/export.py --model mace-omat-0-medium.model --elements 8,22,38 --output omat_O_Ti_Sr.loup

# 2. LAMMPS with the package
lammps/install.sh /path/to/lammps
cmake -S /path/to/lammps/cmake -B build -D PKG_ML-LOUPMD=ON -D LOUPMD_ROOT=/path/to/LoupMD
cmake --build build -j 8
```

```
# 3. an input deck
units           metal
atom_style      atomic
atom_modify     map yes
boundary        p p p
read_data       srtio3.data

pair_style      loupmd
pair_coeff      * * omat_O_Ti_Sr.loup Sr Ti O        # one element per atom type, in type order

timestep        0.001
run_style       loupmd langevin 300.0 0.1 48279      # the engine integrates: T [K], damp [ps], seed
thermo          100
run             10000
```

With LAMMPS's own integrators instead, drop the `run_style` line and add fixes as usual:

```
fix             1 all npt temp 300.0 300.0 0.1 iso 1.0 1.0 1.0
```

For `loupmd/kk`, build LAMMPS with `-D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_<GPU>=ON` as well and run
`lmp -k on g 1 -sf kk -in in.lammps`.

## Syntax

```
pair_style loupmd [device N] [skin S] [ghost yes|no]
pair_coeff * * model.loup El1 El2 ...
```

- `device N`: the CUDA device (default 0; `loupmd/kk`: the device Kokkos uses).
- `skin S`: the skin of the engine's own neighbour list in Å (default: LAMMPS's `neighbor` skin).
- `ghost no`: tells LAMMPS a pair cutoff of zero, so it creates no ghost atoms. The engine does not need them; LAMMPS
  then warns that the communication cutoff is 0 and switches atom sorting off. Default `yes` (the model's cutoff).
- `pair_coeff`: the bundle and one element symbol per atom type. Every element must be in the bundle. `NULL` is not
  supported, since the engine evaluates every atom.

```
run_style loupmd nve
run_style loupmd langevin T damp seed
```

- The engine's integrators: velocity Verlet, and the Langevin scheme of ASE (Vanden-Eijnden and Ciccotti) with
  friction 1 / `damp`. `timestep` sets the step.
- The engine runs blocks of steps up to the next step with thermo, dump or restart output. There LAMMPS gets
  positions and velocities back and the pair style evaluates energy, forces and pressure for the output. Each block
  costs about two extra force evaluations, so output every 100 steps or more keeps the engine's step time.
- No fix may act during the steps (no `fix nve`, `fix langevin`, ...): the run stops with an explanation. Use the
  pair style with LAMMPS's default `run_style` for those.
- Run it without `-k on` / `-sf kk`.

## Requirements and limits

- `units metal`, three periodic dimensions (orthogonal or triclinic box), one MPI process, one GPU.
- Atom IDs must be consecutive (1 … N): the engine keeps atoms by ID, so LAMMPS may sort and reorder them freely.
  Use `reset_atoms id` after deleting atoms.
- Energy, forces and the global virial (pressure, `fix npt`, `fix box/relax`). No per-atom energies or virials
  (`compute pe/atom`, `compute stress/atom`).
- A changing box (`fix npt`) rebuilds the engine's neighbour list in every step: 1.7 to 1.9× the step time on a
  648-atom cell.
- An isolated molecule needs a periodic box with more than 2 (cutoff + skin) of vacuum.
- MACE-POLAR-1 models run as in the engine: neutral, singlet, zero field.
- `hybrid/overlay` with other pair styles works with the pair styles (forces are added), not with `run_style loupmd`.

## Speed (RTX 5060 Ti under WSL 2, MACE-OMAT-0 medium, Langevin 300 K, 19 November 2026)

SrTiO3 from the benchmark inputs, 100 warm-up and 256 measured steps, two runs each:

| | 5,000 atoms, ms/step | 320 atoms, ms/step |
|---|---|---|
| `run_style loupmd` | 7.84 - 7.86 | 0.61 - 0.63 |
| `pair_style loupmd` | 8.20 | 0.86 - 0.90 |
| `pair_style loupmd/kk` | 8.38 - 8.45 | 1.29 - 1.38 |
| Symmetrix `symmetrix/mace/foundation/float32/kk` (same LAMMPS version, same job) | 13.47 - 13.54 | 1.55 - 1.59 |

Under WSL 2 every blocking exchange between host and GPU costs 0.1 - 0.2 ms, and LAMMPS-Kokkos makes several per step:
with a zero-strength Lennard-Jones potential on 648 atoms it takes 2.4 ms per step by itself. That is why
`loupmd/kk` trails the plain style there. On native Linux the Kokkos style has not been measured.

## Files

```
lammps/install.sh                          copies the sources into a LAMMPS tree and registers the package
lammps/cmake/ML-LOUPMD.cmake               finds loupmd.h and libloupmd.so (LOUPMD_ROOT)
lammps/ML-LOUPMD/pair_loupmd.{h,cpp}       pair_style loupmd
lammps/ML-LOUPMD/integrate_loupmd.{h,cpp}  run_style loupmd
lammps/KOKKOS/pair_loupmd_kokkos.{h,cpp}   pair_style loupmd/kk
lammps/examples/                           input decks
```

The files in `lammps/` are built into LAMMPS and are distributed under the GNU General Public License version 2,
as LAMMPS is. `libloupmd.so` keeps LoupMD's licence and is linked through its C interface (`include/loupmd.h`).

Written against LAMMPS `stable_30Sep2026_update1` (Kokkos 5.2.1).
