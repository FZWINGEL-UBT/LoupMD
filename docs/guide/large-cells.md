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
4. When no tile size fits, it keeps the tiles' neighbour lists in the computer's memory (pinned, at most half of
   what is free) and the GPU reads them over the bus. The run is then 15-20% slower, and larger cells fit: the
   lists are about a third of what the tiles keep on the GPU (an estimate; most of it with the wavefront). The
   report says so on its memory line.

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
  whole); the integrator runs on the device
  memory: the whole system would need 32.12 GiB; tiled 10.45 GiB planned, 14.00 GiB free on the card
```

`capacity="whole"` forces the whole-cell call; a cell that does not fit then fails with an out-of-memory message.
`capacity="tiled"` forces tiles, and `lm.config.CapacityConfig(mode="tiled", receivers=8192)` also fixes their
size. `memory_fraction` (default 0.9) is the share of the free memory the plan may use.

## The wavefront

The plain tiled call keeps every layer's output for the whole cell on the GPU. At a million atoms that alone needs
about 30 GB for OMAT-0 medium. For two-layer models the tiles can run in the wavefront order instead. The tiles lie
in layers along the first cell axis, and each tile's first pass runs just before the first tile that needs its
output. Its last pass runs just after the last tile that adds to it. Only a moving band of tiles is held, in a pool
of rows.

`capacity="auto"` uses the wavefront from 2^20 atoms when the plan has 12 or more layers of tiles along the first
axis. `capacity="wavefront"` forces it. A model with three layers (OMOL-0) is refused in that mode and uses the
plain tiled call. The forces are those of the plain tiled call to float32 rounding. The report then adds "the tiles
run in the wavefront order".

Measured on an RTX 5060 Ti with OMAT-0 medium (STATUS_AND_PROFILE.md §6.22):

| Atoms | Mode | Tiles | GPU peak | Seconds per force call |
|---|---|---|---|---|
| 368,640 | tiled | 32 | 4.91 GiB | 0.80 |
| 368,640 | wavefront | 40 | 4.05 GiB | 0.80 |
| 1,081,600 | wavefront (`auto`; 11% of the atoms' rows held) | 34 | 10.84 GiB | 2.35 |

## What the tiled mode runs

- **Integrators:** the fixed-cell ones: `NVE`, `Langevin`, `CSVR`, `NoseHooverChain` and `Berendsen` without a
  pressure. They run on the GPU with the whole-cell run's kernels; positions, velocities and forces stay on the card
  between steps. Barostats are refused in tiled mode, with a message: the cell stays fixed (PLAN.md D65).
- **Random stream:** the noise is the whole-cell run's, so a tiled Langevin or CSVR run follows the whole-cell
  trajectory to float32 force rounding, and a resumed run continues the stream and the thermostat's state.
- **With a dispersion term** the D3 energy is added on the host each step, so the integrator runs on the host in
  numpy, `NVE` and `Langevin` only, with its own Langevin noise stream (`LOUP_TILED_HOST_MD=1` forces this loop).
  The report says where the integrator ran.
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
| 320,000 (`SrTiO3`, MD step, NVE / Langevin) | does not fit | | 1.95 / 1.97 (8 tiles of 65,536; 624 / 630 ms per step) |
| 1,081,600 (`SrTiO3`, MD step, Langevin, wavefront) | does not fit | | 2.11 (32 tiles of 49,152; 2.28 s per step) |

The 40,000-atom row is older (before the November 2026 tiled-call work); at 320,000 atoms and more the tiled MD
step costs about 1.2 to 1.3 times the whole-cell call per atom (1.65 us). The first step of a tiled run tunes the
GEMMs for the tile shapes, which takes 1.5 to 5 s. STATUS_AND_PROFILE.md §6.16 to §6.18 has the measurements.
