# Constant pressure (NPT)

Every barostat in LoupMD is experimental in this version, and the run's report says so. The cell is integrated on
the GPU with the atoms; the stress is computed in every step.

```python
import loupmd as lm

result = lm.md("structure.cif", model="omat0-medium", ensemble="NPT", temperature_K=300, pressure_GPa=1.0,
               steps=20_000, timestep_fs=1.0)
```

```bash
loupmd md structure.cif --model omat0-medium --ensemble NPT --temperature 300 --pressure 1.0 --steps 20000 \
    --timestep 1
```

`ensemble="NPT"` runs Langevin BAOAB with a Langevin-Hoover barostat (`lm.integrators.LangevinBAOAB`):

- `pressure_GPa` / `--pressure`: the target pressure in GPa, default 0.
- `cell_mode` / `--cell-mode`: `hydrostatic` (default: the cell keeps its shape and only its size changes) or `full`
  (all nine components move).
- The friction is the NVT default, about 0.002/fs; `--friction PER_FS` changes it and keeps the run NPT.
- The barostat's time constant is 20 / friction (about 10 ps at the default friction).

`pressure_GPa=` and `cell_mode=` belong to NPT; with another ensemble they are an error. `--thermostat csvr` is an
NVT thermostat and conflicts with `--ensemble NPT`.

## What an NPT run writes

The thermo rows gain two columns, `volume_A3_atom` (A^3 per atom) and `pressure_GPa`. The pressure is minus a third
of the stress trace plus the ideal-gas part 2 E_kin / (3 V). Every trajectory frame carries its cell, and
`Simulation.atoms` returns the current cell. The restart holds the cell, so a resumed run goes on from it.

## The barostats

`integrator=` (Python) or an `[integrator]` table (run files) chooses one of three records. The time constants are
in fs, the pressure in GPa, the compressibility per GPa.

| record | as in ASE | fields without a default |
|---|---|---|
| `LangevinBAOAB` | `LangevinBAOAB` with a barostat | `timestep_fs`, `temperature_K`, `friction_per_fs` |
| `MTK` | `IsotropicMTKNPT` (hydrostatic), `MTKNPT` (full) | `timestep_fs`, `temperature_K`, `pressure_GPa`, `tdamp_fs`, `pdamp_fs` |
| `Berendsen` with a pressure | `NPTBerendsen` | `timestep_fs`, `temperature_K`, `taut_fs`, `pressure_GPa`, `taup_fs`, `compressibility_per_GPa` |

`LangevinBAOAB` is BAOAB Langevin dynamics for the atoms (Leimkuhler and Matthews) with a Langevin-Hoover barostat for
the cell (Quigley and Probert). `MTK` is Martyna-Tobias-Klein with Nose-Hoover chains on the atoms and on the cell.
`Berendsen` is Berendsen's thermostat and barostat: not canonical, for equilibration only.

```python
mtk = lm.integrators.MTK(timestep_fs=1.0, temperature_K=300, pressure_GPa=0.0, tdamp_fs=100, pdamp_fs=1000,
                         cell_mode="full")
result = lm.md("structure.cif", model="omat0-medium", integrator=mtk, steps=20_000)

ber = lm.integrators.Berendsen(timestep_fs=1.0, temperature_K=300, taut_fs=100, pressure_GPa=0.0, taup_fs=1000,
                               compressibility_per_GPa=0.006)
```

- **LangevinBAOAB:** `pressure_GPa` (default 0), `cell_mode`, and `barostat_tau_fs` (ASE's `P_tau`; `None` gives
  ASE's default, 20 times the inverse friction). The barostat's mass is ASE's heuristic. In the full-cell mode the
  barostat's drag on the velocities is Quigley and Probert's; ASE 3.27 adds its trace term to all nine elements.
  `fixcm` must be `False`, as in ASE. FastMACE's fields `external_stress_GPa`, `barostat_mass_factor`,
  `disable_cell_langevin`, `max_volume_ratio` and `max_cell_length_A` are accepted only at their defaults and
  refused by name otherwise.
- **MTK:** `tchain`, `pchain` (chains of 1 to 8 links, default 3), `tloop`, `ploop` (default 1) and `cell_mode`.
  It is deterministic: no random noise. The conserved quantity is E_pot + E_kin plus both chains' energies, the
  cell's kinetic energy and P V. The masses of both chains and of the cell are made for `temperature_K` and stay
  when a schedule moves the target. `fixcm="auto"` keeps the centre of mass at rest.
- **Berendsen:** without `mask` the cell keeps its shape; `mask=(1, 1, 0)` lets the first two cell vectors follow
  their own stress and leaves the third alone (ASE's `Inhomogeneous_NPTBerendsen`). It makes one force call per step:
  the first kick of a step uses the forces of the cell before its scaling, where ASE evaluates again. `taut_fs` must
  be at least the timestep.

[The integrator reference](../reference/integrators.md) lists every field. The ASE drop-ins in `loupmd.ase.md`
(`LangevinBAOAB`, `IsotropicMTKNPT`, `MTKNPT`, `NPTBerendsen`, `Inhomogeneous_NPTBerendsen`) take ASE's arguments
and run the same integrators ([ASE](ase.md)).

## Run files and profiles

In a run file `ensemble = "NPT"`, `pressure_GPa` and `cell_mode` select Langevin BAOAB; an `[integrator]` table with
`type = "MTK"`, `"LangevinBAOAB"` or `"Berendsen"` selects another. A profile with `ensemble = "NPT"` and
`pressure_GPa` (default 0 GPa) runs at that constant pressure: `--profile` then means Langevin BAOAB at the
profile's pressure, `--pressure` overrides it, and an `[integrator]` table's barostat runs at the profile's pressure
unless the table names one. A profile at constant pressure refuses any other `ensemble` and an integrator without
a barostat. Pressure schedules are refused; temperature schedules work with every barostat.

```toml
structure = "structure.cif"
model = "omat0-medium"
timestep_fs = 1.0
temperature_K = 300.0
steps = 20000

[integrator]
type = "MTK"
pressure_GPa = 0.0
tdamp_fs = 100.0
pdamp_fs = 1000.0
```

## Restrictions

- **Fixed atoms** (`FixAtoms`) are refused with every barostat: the cell's scaling would move them.
- **Long-range models** (MACE-POLAR-1) are refused with a barostat: their k-space terms are set up for one cell in
  this version.
- **Tiled execution** keeps the cell fixed: it runs NVE and the NVT thermostats (Langevin, CSVR, Nose-Hoover chain,
  Berendsen) and refuses every barostat, so a barostat needs a cell that fits the GPU whole. With
  `capacity="auto"` a cell that does not fit is refused at the start of an NPT run ([large cells](large-cells.md)).
- **Volume limits.** The engine stops a run whose volume leaves V0 / 20 to 20 V0, with V0 the starting volume (kept
  across restarts). The run raises `CellDivergedError` naming the steps; the structure has usually left its
  condensed phase or the barostat diverged: raise the pressure or lower the temperature.
- **Events.** Every `Simulation.update` that sets the state anew (positions, momenta, masses, species, charge)
  starts the barostat's own momentum from zero.

## Dispersion in NPT

`dispersion=` adds the D3 term's stress to the model's, so the barostat sees it ([dispersion](dispersion.md)). The
plain sum, the three-body term and the split sum all run under the barostat. `long_range="auto"` chooses its method
once, at setup, on the starting cell; a change of cell does not repeat that check. `Simulation.recheck_dispersion()`
repeats it on the current cell.

## Relaxing the cell

To bring a structure near its equilibrium cell before MD, relax positions and cell together with
`lm.optimize.FrechetCell` ([relaxation](relax.md)); it is refused with fixed atoms and for long-range models, as the
barostats are.
