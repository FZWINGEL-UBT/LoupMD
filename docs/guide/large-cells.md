# Cells that do not fit the GPU

LoupMD runs the whole cell in one pass when it fits the GPU. A larger cell is split into tiles. Each tile is a block
of at most `receivers` atoms plus the atoms within the model's reach around it. The forces are the same as the
whole-cell call to float32 rounding: the plan's bounds are 1e-7 eV/atom and 1e-5 eV/A with the hybrid GEMMs off
(PLAN.md D60).

## Choosing the mode

`capacity="auto"` is the default of `lm.md`, `lm.Simulation` and `loupmd md`. Before the first step the engine
plans the run:

1. It sets up one probe tile of at most 4,096 atoms and measures the memory per atom and per neighbour edge.
2. It uses the whole-cell call when that fits in 90% of the free GPU memory and the neighbour list stays under
   536,870,911 edges.
3. Otherwise it uses the largest tile size from 65,536 down to 4,096 atoms that fits.

```python
import loupmd as lm

result = lm.md("big.cif", model="omat0-medium", ensemble="NVT", temperature_K=300, steps=10_000)
print(result.plan["execution"], result.plan.get("receivers"))      # "tiled", 65536
```

```bash
loupmd md big.cif --model omat0-medium --temperature 300 --steps 10000 --capacity auto
```

The report says which mode the run took (here 320,000 atoms on a 16 GB card):

```
Plan
  8 tiles of at most 65,536 atoms on GPU 0 (NVIDIA GeForce RTX 5060 Ti, ...) (the cell does not fit the card
  whole); the integrator runs on the host
  memory: the whole system would need 32.12 GiB; tiled 10.45 GiB planned, 14.00 GiB free on the card
```

`capacity="whole"` forces the whole-cell call; a cell that does not fit then fails with an out-of-memory message.
`capacity="tiled"` forces tiles, and `lm.config.CapacityConfig(mode="tiled", receivers=8192)` also fixes their
size. `memory_fraction` (default 0.9) is the share of the free memory the plan may use.

## What the tiled mode runs

- **Integrators:** `NVE` and `Langevin` (ASE's formulas). Every other thermostat and every barostat is refused in
  tiled mode, with a message. In this version the integrator runs on the host in numpy, and only the forces come
  from the GPU (PLAN.md D65). Moving the integrators onto the device is a later step.
- **Random stream:** the Langevin noise of a tiled run comes from its own stream, seeded by the run's seed and the
  step. A tiled run does not repeat a whole-cell run's trajectory step for step, but a resumed tiled run continues
  its own stream exactly.
- **Fixed atoms and centre of mass:** both are handled as in the whole-cell run (`FixAtoms`, `fixcm`).
- **Neighbour lists:** each tile keeps its list across steps. All lists are rebuilt once any atom has moved half the
  skin.
- **Models:** generic-path models with two or more interaction layers and no long-range terms. `MACE-POLAR-1` always
  runs whole.
- **Restarts:** a restart records the mode, and a resume uses the same one. A whole-cell restart cannot be read by
  the tiled engine, and a tiled restart cannot be read by the whole-cell engine.

## Speed

Measured on an RTX 5060 Ti (16 GB) with OMAT-0 medium; µs per atom per force call:

| Atoms | Whole cell | 2 tiles | 8 tiles |
|---|---|---|---|
| 40,000 | 1.65 | 2.97 | 4.33 |
| 320,000 (`SrTiO3`, NVE with the host integrator) | does not fit | | 3.50 (8 tiles of 65,536; 1.12 s per step) |

The tiled mode costs about two to three times the whole-cell call per atom. The first step of a tiled run tunes the
GEMMs for the tile shapes, which takes 1.5 to 5 s. STATUS_AND_PROFILE.md §6.16 to §6.18 has the measurements.
