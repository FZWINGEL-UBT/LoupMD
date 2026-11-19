# LoupMD in LAMMPS

MACE models in LAMMPS through the LoupMD engine, on one CUDA GPU. The package `ML-LOUPMD` adds:

| style | what it does | when to use it |
|---|---|---|
| `pair_style loupmd` | LAMMPS integrates; positions and forces pass through host memory once per step | the default: any LAMMPS build, any fix |
| `pair_style loupmd/kk` | LAMMPS-Kokkos integrates; positions and forces stay on the GPU | Kokkos CUDA builds that also run other Kokkos styles |
| `run_style loupmd` | the engine integrates between LAMMPS's output steps, in the ensemble of the input's `fix nve`, `nvt`, `npt`, `langevin`, ... | the engine's own step time for NVE, NVT and NPT; a fix the engine has no integrator for is refused by name |

Design and measurements: `LAMMPS_KOKKOS_PLAN.md` in the repository root. The plan for the options of Symmetrix's
pair styles (run modes, several processes, per-atom energies, open boundaries): `LAMMPS_SYMMETRIX_PARITY_PLAN.md`.

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
fix             1 all nvt temp 300.0 300.0 0.1
run_style       loupmd                               # the engine integrates fix nvt's ensemble
thermo          100
run             10000
```

Without the `run_style` line LAMMPS integrates itself, with any of its fixes, at a slower step (see Speed).

For `loupmd/kk`, build LAMMPS with `-D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_<GPU>=ON` as well and run
`lmp -k on g 1 -sf kk -in in.lammps`.

## Syntax

```
pair_style loupmd [device N] [skin S] [ghost yes|no] [ahead yes|no]
pair_coeff * * model.loup El1 El2 ...
```

- `device N`: the CUDA device (default 0; `loupmd/kk`: the device Kokkos uses).
- `skin S`: the skin of the engine's own neighbour list in Å (default: LAMMPS's `neighbor` skin).
- `ghost no`: tells LAMMPS a pair cutoff of zero, so it creates no ghost atoms. The engine does not need them; LAMMPS
  then warns that the communication cutoff is 0 and switches atom sorting off. Default `yes` (the model's cutoff).
- `ahead yes`: the engine queues each next force pass ahead of the call that needs it. Measured: 2 - 7% off the step
  on cells of a few hundred atoms, nothing at 5,000 atoms. Default `no`, because between two steps the engine's GPU
  stream then holds work that cannot run, and any other GPU library in the same process could wait for it forever.
  Use it only when LoupMD is the process's one GPU user. It is ignored with the KOKKOS or GPU package active, and
  has no meaning for `loupmd/kk` and `run_style loupmd`.
- `pair_coeff`: the bundle and one element symbol per atom type. Every element must be in the bundle. `NULL` is not
  supported, since the engine evaluates every atom.

```
run_style loupmd
run_style loupmd nve
run_style loupmd langevin T damp seed
```

`run_style loupmd` reads the input's fixes and runs the engine's integrator of that ensemble in their place:

| in the input | the engine runs | limits |
|---|---|---|
| `fix nve` | velocity Verlet | |
| `fix nve` + `fix langevin Tstart Tstop damp seed` | Langevin (the scheme of Vanden-Eijnden and Ciccotti), friction 1 / `damp` | `zero yes` keeps the centre of mass at rest; no `omega`, `angmom` or `scale` |
| `fix nve` + `fix temp/csvr Tstart Tstop Tdamp seed` | velocity rescaling of Bussi, Donadio and Parrinello | the noise is seeded from the fix's generator (package EXTRA-FIX) |
| `fix nve` + `fix temp/berendsen Tstart Tstop Tdamp` | Berendsen thermostat | |
| the same + `fix press/berendsen iso P P Pdamp modulus M` | Berendsen thermostat and barostat, compressibility 1 / `M` | orthogonal box (the fix's own rule); `iso`, or `aniso` / single directions with one pressure and one `Pdamp`; no ramp |
| `fix nvt temp Tstart Tstop Tdamp` | Nosé-Hoover chain (`tchain`, `tloop`) | no `drag` |
| `fix npt temp Tstart Tstop Tdamp iso P P Pdamp` | Martyna-Tobias-Klein, isotropic (`tchain`, `pchain`, `tloop`, `ploop`) | `iso` only (orthogonal or triclinic box, the shape is kept); no pressure ramp, `drag`, `mtk no` or `dilate` |
| `fix setforce 0.0 0.0 0.0` on a group at rest; atoms outside the group of `fix nve` / `nvt` | those atoms are held in place | not with a barostat |
| `fix momentum N linear 1 1 1` on `all` | the centre of mass is kept at rest | in every step |

- The log names what runs (`Ensemble : fix 1 (nvt 300 -> 300 K, Tdamp 0.1 ps, tchain 3): Nose-Hoover chain`).
- Any other fix that acts during the steps stops the run with its name and the reason (`fix nph`, `fix npt aniso`,
  `fix addforce`, `fix shake`, `fix ave/time`, a thermostat with `fix_modify temp`, a temperature given as a
  variable, ...). Nothing is dropped silently. Use LAMMPS's default run style with the pair style for such inputs.
  Fixes that do not act during the steps are untouched.
- The settings carry over as written: temperatures in K with their ramp over the run, `Tdamp` and `Pdamp` in ps,
  pressures in bar, the chain lengths. The integrators are the engine's own (those of ASE 3.27): they sample the
  same ensembles as LAMMPS's fixes with the same parameters, but a trajectory is not LAMMPS's step for step.
- Degrees of freedom. LAMMPS's thermostats count 3N - 3. The engine counts the same when it keeps the centre of mass
  at rest: with `fix momentum` or `zero yes`, and for `nvt`, `npt`, `temp/berendsen` and `temp/csvr` whenever the
  starting velocities have no total momentum (`velocity ... mom yes`) and no atom is held. Otherwise it counts
  three per moving atom.
- The engine runs blocks of steps up to the next step with thermo, dump or restart output. There LAMMPS gets
  positions, velocities and the box back and the pair style evaluates energy, forces and pressure for the output.
  Each output costs about two force evaluations: measured on 320 atoms, +2% on the step with output every 100
  steps and +26 to 34% with output every 10.
- From one `run` command to the next the engine keeps its state, with the thermostat's and the barostat's variables,
  when nothing changed in between (positions, velocities, box, masses, the fixes' settings). After a change those
  variables start from zero, as under a newly defined fix. The noise of `langevin` and `temp/csvr` never repeats
  from run to run.
- The fixes themselves do not run, so their own outputs (`f_ID`, and with them `ecouple` and `econserve`) stay as
  they were. `temp`, `pe`, `ke`, `press` and the box in the thermo output are LAMMPS's own, computed on the state
  the engine returned. The forces LAMMPS writes for atoms under `fix setforce 0 0 0` are zero.
- `run_style loupmd nve` and `run_style loupmd langevin T damp seed` name the ensemble on the line itself; then no
  fix may act during the steps.
- Run it without `-k on` / `-sf kk`.

## Requirements and limits

- `units metal`, three periodic dimensions (orthogonal or triclinic box), one MPI process, one GPU.
- Atom IDs must be consecutive (1 … N): the engine keeps atoms by ID, so LAMMPS may sort and reorder them freely.
  Use `reset_atoms id` after deleting atoms.
- Energy, forces and the global virial (pressure, `fix npt`, `fix box/relax`). No per-atom energies or virials
  (`compute pe/atom`, `compute stress/atom`).
- LAMMPS's fixes work with the pair styles as with any pair style. Run on 19 November 2026: `nve`, `nvt`, `npt`,
  `nph`, `langevin`, `temp/csvr`, `temp/berendsen`, `press/berendsen`, `setforce`, `spring/self`, `momentum`,
  `recenter`, `addforce`, `deform`, `nve/limit`, `viscous`, `shake`, `rattle`, `rigid/nve/small`,
  `rigid/nvt/small`, `box/relax` with `minimize`, `hybrid/overlay`. `fix press/berendsen` needs the material's
  `modulus`: LAMMPS's default of 10 bar makes a solid's box explode.
- With LAMMPS's own integrator a changing box (`fix npt`) rebuilds the engine's neighbour list in every step.
  `run_style loupmd` runs `fix npt ... iso` at the engine's step time instead (see Speed).
- An isolated molecule needs a periodic box with more than 2 (cutoff + skin) of vacuum.
- MACE-POLAR-1 models run as in the engine: neutral, singlet, zero field.
- `hybrid/overlay` with other pair styles works with the pair styles (forces are added), not with `run_style loupmd`.

## Speed (RTX 5060 Ti under WSL 2, MACE-OMAT-0 medium, Langevin 300 K, 19 November 2026)

SrTiO3 from the benchmark inputs, 100 warm-up and 256 measured steps, two runs each:

| | 5,000 atoms, ms/step | 320 atoms, ms/step |
|---|---|---|
| `run_style loupmd` | 7.84 - 7.86 | 0.61 - 0.63 |
| `pair_style loupmd` | 8.20 | 0.86 - 0.90 |
| `pair_style loupmd/kk` | 8.14 - 8.19 | 0.97 - 1.03 |
| Symmetrix `symmetrix/mace/foundation/float32/kk` (same LAMMPS version, same job) | 13.47 - 13.54 | 1.55 - 1.59 |

With `pair_style loupmd` a step is the engine's force pass plus about 0.2 ms: the start of the engine's step on an
idle GPU, which the engine's own loop hides by queueing ahead, and LAMMPS's own CPU work (another 0.2 ms at 5,000
atoms). `run_style loupmd` avoids both.

The ensembles of the fixes (rattled SrTiO3 cells, 300 K, 1 fs, 2,000 steps after 200, thermo every 1,000, two runs
each with fresh GEMM plans; ms/step):

| atoms | input | the Python package's own loop | `run_style loupmd` | LAMMPS integrates (`pair_style loupmd`) |
|---|---|---|---|---|
| 320 | `fix nvt` | 0.647, 0.647 | 0.651, 0.647 | 0.810, 0.837 |
| 320 | `fix npt iso` | 0.795, 0.780 | 0.783, 0.792 | 1.309, 1.287 |
| 320 | `fix nve` + `fix langevin` | 0.580, 0.573 | 0.594, 0.578 | 0.795, 0.801 |
| 1,080 | `fix nvt` | 1.748, 1.751 | 1.775, 1.761 | 1.969, 2.007 |
| 1,080 | `fix npt iso` | 1.847, 1.863 | 1.844, 1.847 | 2.630, 2.643 |
| 1,080 | `fix nve` + `fix langevin` | 1.718, 1.711 | 1.718, 1.716 | 1.986, 2.032 |
| 5,000 | `fix nvt` | 7.689, 7.678 | 7.743, 7.729 | 8.069, 8.083 |
| 5,000 | `fix npt iso` | 7.687, 7.666 | 7.674, 7.663 | 8.891, 8.861 |
| 5,000 | `fix nve` + `fix langevin` | 7.659, 7.676 | 7.663, 7.677 | 8.192, 8.166 |

The run style is within 1% of the package's own loop in every row. LAMMPS integrating itself takes 1.04 to 1.27
times as long with `fix nvt` and 1.16 to 1.65 times with `fix npt`.

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
as LAMMPS is (`lammps/LICENSE`; the owner's decision of 19 November 2026). `libloupmd.so` keeps LoupMD's licence and
is linked through its C interface (`include/loupmd.h`).

Written against LAMMPS `stable_30Sep2026_update1` (Kokkos 5.2.1).
