# Integrators

An integrator record of `lm.integrators` (`loupmd/integrators.py`) is what `lm.Simulation` and `lm.md` take as
`integrator=`. Each record is frozen and takes keyword arguments only. It checks its values when it is made and maps
onto an `IntegratorConfig` (`loupmd/config.py`), which is what the engine runs (`src/integrator.cu`, `src/cell.cu`).
How to run MD is in [Molecular dynamics](../guide/md.md); constant pressure is in [NPT](../guide/npt.md).

Units: time in fs, temperature in K, friction per fs, pressure in GPa, compressibility per GPa. A thermostat's
temperature, friction and time constant have no defaults: physics is not chosen silently.

`lm.integrators` exports `NVE`, `Langevin`, `CSVR`, `NoseHooverChain` and `Berendsen` (`__all__`). `LangevinBAOAB`
and `MTK` are records of the same module (`lm.integrators.LangevinBAOAB`, `lm.integrators.MTK`); both are
experimental.

## Overview

| Record | Ensemble | Scheme (`IntegratorConfig.scheme`) | Follows (ASE 3.27) | Conserved quantity |
| --- | --- | --- | --- | --- |
| `NVE` | NVE | `nve` | `VelocityVerlet` | E_kin + E_pot |
| `Langevin` | NVT | `langevin` | `Langevin` (Vanden-Eijnden-Ciccotti scheme) | - |
| `CSVR` | NVT | `csvr` | `Bussi` | E_kin + E_pot minus the energy the thermostat transferred |
| `NoseHooverChain` | NVT | `nose_hoover` | `NoseHooverChainNVT` | E_kin + E_pot plus the chain's energy |
| `Berendsen` | NVT | `berendsen` | `NVTBerendsen` | - (not canonical) |
| `Berendsen` with barostat fields | NPT (experimental) | `berendsen` | `NPTBerendsen`; with `mask`, `Inhomogeneous_NPTBerendsen` | - (not canonical) |
| `LangevinBAOAB` | NPT (experimental) | `baoab` | `LangevinBAOAB` with a barostat | - |
| `MTK` | NPT (experimental) | `mtk` | `IsotropicMTKNPT` (`cell_mode="hydrostatic"`), `MTKNPT` (`"full"`) | E_pot + E_kin plus both chains' energies, the cell's kinetic energy and P V |

The conserved quantity is listed where the record documents one.

The tiled execution of cells that do not fit the GPU integrates `NVE` and `Langevin` only, on the host; other
records are refused there ([Large cells](../guide/large-cells.md)). A long-range model (MACE-POLAR-1) with a
barostat is refused.

## Selecting a record

| How | Record |
| --- | --- |
| `loupmd md --ensemble NVE` | `NVE(timestep_fs=--timestep)` |
| `loupmd md` (default `--ensemble NVT`) | `Langevin` at `--temperature` or `--schedule`, friction `--friction` (default below) |
| `loupmd md --thermostat csvr --tau FS` | `CSVR` with `tau_fs=FS`; `--friction` is refused with it |
| `loupmd md --ensemble NPT [--pressure GPA] [--cell-mode MODE]` | `LangevinBAOAB` at that pressure (default 0) and cell mode (default `hydrostatic`), friction `--friction` (default below) |
| a run file's `[integrator]` table with `type = "..."` (`loupmd run`) | any record of `RECORDS`: `NVE`, `Langevin`, `CSVR`, `NoseHooverChain`, `Berendsen`, `LangevinBAOAB`, `MTK` |
| `lm.md(..., integrator=...)`, `lm.Simulation(..., integrator=...)` | the record given |
| `lm.md(..., ensemble="NPT")` without `integrator=` | `LangevinBAOAB` with `pressure_GPa=` and `cell_mode=` |

The default friction of `lm.md` and `loupmd md` is 0.02 per ASE time unit, about 0.001965 per fs
(`default_friction_per_fs`). The default timestep of `lm.md` and `loupmd md` is 0.5 fs.

In a run file, the `[integrator]` table holds `type` and the record's fields (`to_data`, `from_data`). The run file's
`timestep_fs` and the starting temperature fill those fields where the table leaves them out. Unknown keys and
missing fields without a default are refused by name. `loupmd init md` writes a run file
([Command-line reference](cli.md)).

## Centre of mass: `fixcm`

`fixcm` keeps the centre of mass at rest. `"auto"` (the default where the record has the field) is on without
fixed atoms and off with them, since fixed atoms pin the frame already. `True` with fixed atoms is refused
(`PlanRefused`). With `fixcm` on:

- `Langevin` removes the mass-weighted mean of its noise every step;
- `NVE`, `CSVR` and `NoseHooverChain` remove the total momentum at the start of every run call;
- the temperature counts three degrees of freedom fewer, and so do the thermostats that count them (CSVR, the
  Nose-Hoover chain, MTK).

Atoms under ASE's `FixAtoms` stay where they are and count no degrees of freedom. The barostats (`Berendsen` with
barostat fields, `LangevinBAOAB`, `MTK`) refuse fixed atoms.

## `NVE`

Velocity Verlet. Example: `lm.integrators.NVE(timestep_fs=1.0)`.

| Field | Type | Default | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `timestep_fs` | float | required | fs | timestep |
| `fixcm` | bool or `"auto"` | `"auto"` | | centre of mass at rest (above) |

## `Langevin`

Langevin dynamics at constant volume, as ASE's `Langevin`. Example:
`lm.integrators.Langevin(timestep_fs=0.5, temperature_K=300, friction_per_fs=0.002)`.

| Field | Type | Default | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `timestep_fs` | float | 0.5 | fs | timestep |
| `temperature_K` | float | required | K | target temperature |
| `friction_per_fs` | float | required | 1/fs | friction |
| `fixcm` | bool or `"auto"` | `"auto"` | | centre of mass at rest (above) |

## `CSVR`

Canonical sampling through velocity rescaling (Bussi, Donadio, Parrinello), as ASE's `Bussi`. At the start of a step
the velocities are scaled; the step is then velocity Verlet. The structure must move at the start, as for ASE's
`Bussi`. Example: `lm.integrators.CSVR(timestep_fs=1.0, temperature_K=300, tau_fs=100)`.

| Field | Type | Default | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `timestep_fs` | float | required | fs | timestep |
| `temperature_K` | float | required | K | target temperature |
| `tau_fs` | float | required | fs | the thermostat's time constant (ASE's `taut`) |
| `fixcm` | bool or `"auto"` | `"auto"` | | centre of mass at rest (above) |

## `NoseHooverChain`

Nose-Hoover chains, as ASE's `NoseHooverChainNVT`: the chain for dt / 2, a half kick, the drift, a half kick, the
chain for dt / 2. The thermostat masses are made for `temperature_K` and stay when a schedule moves the target, as
in ASE. The degrees of freedom stand where ASE counts 3 N: 3 N - 3 with the centre of mass at rest, 3 (N - fixed)
with fixed atoms. Example: `lm.integrators.NoseHooverChain(timestep_fs=1.0, temperature_K=300, tdamp_fs=100)`.

| Field | Type | Default | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `timestep_fs` | float | required | fs | timestep |
| `temperature_K` | float | required, above 0 | K | target temperature; the thermostat masses are made for it |
| `tdamp_fs` | float | required | fs | the chain's time constant (ASE's `tdamp`) |
| `tchain` | int | 3 | | thermostats in the chain (ASE's `tchain`), 1 to 16 |
| `tloop` | int | 1 | | sub-steps per half step (ASE's `tloop`), 1 to 100 |
| `fixcm` | bool or `"auto"` | `"auto"` | | centre of mass at rest (above) |

For a run that starts at 0 K, give the temperature the masses should be made for here and the targets as a schedule.

## `Berendsen`

Berendsen's thermostat, as ASE's `NVTBerendsen`. With `pressure_GPa`, `taup_fs` and `compressibility_per_GPa` it is
also Berendsen's barostat, as ASE's `NPTBerendsen` (experimental). Not canonical: for equilibration only. Example:
`lm.integrators.Berendsen(timestep_fs=1.0, temperature_K=300, taut_fs=100)`; with the barostat
`lm.integrators.Berendsen(timestep_fs=1.0, temperature_K=300, taut_fs=100, pressure_GPa=0.0, taup_fs=1000,
compressibility_per_GPa=0.006)`.

| Field | Type | Default | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `timestep_fs` | float | required | fs | timestep |
| `temperature_K` | float | required | K | target temperature |
| `taut_fs` | float | required, at least the timestep | fs | the thermostat's time constant (ASE's `taut`) |
| `pressure_GPa` | float or None | None | GPa | barostat target |
| `taup_fs` | float or None | None | fs | the barostat's time constant (ASE's `taup`) |
| `compressibility_per_GPa` | float or None | None | 1/GPa | the material's compressibility (ASE's `compressibility_au` times `units.GPa`) |
| `mask` | tuple or None | None | | three entries of 0 or 1 |
| `fixcm` | bool or `"auto"` | `"auto"` | | centre of mass at rest (above) |

Any of `pressure_GPa`, `taup_fs`, `compressibility_per_GPa` or `mask` turns the barostat on (`Berendsen.barostat`);
then the first three are required. Without `mask` the cell keeps its shape. `mask=(1, 1, 0)` lets the first two cell
vectors follow their own stress and leaves the third alone, as ASE's `Inhomogeneous_NPTBerendsen`; `(0, 0, 0)` is
refused. The barostat makes one force call per step: the first kick of a step uses the forces of the cell before its
scaling, where ASE's classes evaluate again.

## `LangevinBAOAB`

Langevin BAOAB at constant pressure, as ASE's `LangevinBAOAB` with a barostat (experimental): BAOAB Langevin dynamics
for the atoms (Leimkuhler and Matthews) and a Langevin-Hoover barostat for the cell (Quigley and Probert). Example:
`lm.integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=300, friction_per_fs=0.002, pressure_GPa=0.0)`.

| Field | Type | Default | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `timestep_fs` | float | required | fs | timestep |
| `temperature_K` | float | required | K | target temperature |
| `friction_per_fs` | float | required | 1/fs | the thermostat's friction (ASE's `T_tau` is its inverse) |
| `pressure_GPa` | float | 0.0 | GPa | target pressure |
| `cell_mode` | `"hydrostatic"` or `"full"` | `"hydrostatic"` | | `hydrostatic`: the cell keeps its shape (ASE's `hydrostatic=True`); `full`: the full cell |
| `barostat_tau_fs` | float or None | None | fs | ASE's `P_tau`; None is ASE's default, 20 `T_tau` |
| `fixcm` | bool | False | | must be False, as for ASE's class |
| `external_stress_GPa` | | None | GPa | only None |
| `barostat_mass_factor` | float | 1.0 | | only 1.0; the barostat's mass is ASE's heuristic |
| `disable_cell_langevin` | bool | False | | only False; the cell's thermostat is always on |
| `max_volume_ratio` | float or None | 20.0 | | only 20.0; the run stops once the volume leaves a twentieth to twenty times its start |
| `max_cell_length_A` | float or None | 1000.0 | A | only 1000.0 |

The last four fields are FastMACE's; other values are refused by name (`PlanRefused`). In the full-cell mode the
barostat's drag on the velocities is Quigley and Probert's; ASE 3.27 adds its trace term to all nine elements. A
cell that leaves its limits stops the run with `CellDivergedError` ([Errors](errors.md)).

## `MTK`

Martyna-Tobias-Klein NPT with Nose-Hoover chains on the atoms and on the cell, as ASE's `IsotropicMTKNPT`
(`cell_mode="hydrostatic"`) and `MTKNPT` (`"full"`); experimental and deterministic. The masses of both chains and of
the cell are made for `temperature_K` and stay when a schedule moves the target. The degrees of freedom stand where
ASE counts 3 N: 3 N - 3 with the centre of mass at rest. Example:
`lm.integrators.MTK(timestep_fs=1.0, temperature_K=300, pressure_GPa=0.0, tdamp_fs=100, pdamp_fs=1000)`.

| Field | Type | Default | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `timestep_fs` | float | required | fs | timestep |
| `temperature_K` | float | required | K | target temperature |
| `pressure_GPa` | float | required | GPa | target pressure |
| `tdamp_fs` | float | required | fs | the atoms' chain's time constant (ASE's `tdamp`) |
| `pdamp_fs` | float | required | fs | the cell's time constant (ASE's `pdamp`) |
| `tchain` | int | 3 | | links of the atoms' chain (ASE's `tchain`), 1 to 8 |
| `pchain` | int | 3 | | links of the cell's chain (ASE's `pchain`), 1 to 8 |
| `tloop` | int | 1 | | sub-steps of the atoms' chain (ASE's `tloop`), 1 to 100 |
| `ploop` | int | 1 | | sub-steps of the cell's chain (ASE's `ploop`), 1 to 100 |
| `cell_mode` | `"hydrostatic"` or `"full"` | `"hydrostatic"` | | as above |
| `fixcm` | bool or `"auto"` | `"auto"` | | centre of mass at rest (above) |

FastMACE's record has a `dof` field beside `fixcm`; here the count follows `fixcm` alone.

## The engine's record: `IntegratorConfig`

`to_config(seed=..., fixed_atoms=...)` turns a record into an `IntegratorConfig`. The run's `metadata.json` and
`simulation.json` store this record under `integrator` ([Output](output.md)). Its fields: `ensemble` (`nve`, `nvt`,
`npt`), `scheme`, `timestep_fs`, `temperature_K`, `friction_per_fs`, `tau_fs`, `tchain`, `tloop`,
`mass_temperature_K`, `pressure_GPa`, `taup_fs`, `compressibility_per_GPa`, `mask`, `cell_mode`, `pchain`, `ploop`,
`seed`, `fixcm`.

| Record field | `IntegratorConfig` field |
| --- | --- |
| `CSVR.tau_fs`, `NoseHooverChain.tdamp_fs`, `Berendsen.taut_fs`, `MTK.tdamp_fs` | `tau_fs` |
| `Berendsen.taup_fs`, `LangevinBAOAB.barostat_tau_fs`, `MTK.pdamp_fs` | `taup_fs` |

`mass_temperature_K` is the temperature the Nose-Hoover and MTK masses are made for; it defaults to `temperature_K`.
The thermostat's noise seed comes from the simulation (`seed=`), not from the record.

## Drop-in ASE classes

`loupmd.ase.md` (also importable from `loupmd.ase`) has classes with ASE's names, arguments and units that run on
LoupMD's engine: `Langevin`, `VelocityVerlet`, `Bussi`, `NoseHooverChainNVT`, `NVTBerendsen`, `NPTBerendsen`,
`Inhomogeneous_NPTBerendsen`, `LangevinBAOAB`, `IsotropicMTKNPT` and `MTKNPT`. See [ASE](../guide/ase.md).
