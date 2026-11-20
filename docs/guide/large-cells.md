# Cells that do not fit the GPU

LoupMD runs the whole cell in one pass when it fits the GPU. A larger cell is split into tiles. Each tile is a block
of at most `receivers` atoms plus the atoms within the model's reach around it. The forces are the same as the
whole-cell call to float32 rounding: the plan's bounds are 1e-7 eV/atom and 1e-5 eV/A with the hybrid GEMMs off
(PLAN.md D60). A whole-cell run folds the chain of matrix products between two layers into one and a tiled run
does not, which alone moves forces by about 2e-6 eV/A.

## Choosing the mode

`capacity="auto"` is the default of `lm.md`, `lm.Simulation` and `loupmd md`. Before the first step the engine
plans the run:

1. It sets up one probe tile of at most 4,096 atoms and measures the memory per atom and per neighbour edge.
2. It uses the whole-cell call when that fits in 90% of the free GPU memory, with room for the scratch buffers of a
   first call, and the neighbour list stays under 536,870,911 edges.
3. Otherwise it uses the largest tile size of 65,536, 49,152, 32,768, 24,576, 16,384, 12,288, 8,192, 6,144 and
   4,096 atoms whose state and workspace fit.
4. When the tiles' state does not fit the GPU, or fits only with small tiles, parts of it move to the computer's
   memory, pinned so that the GPU reads and writes them directly: first the tiles' neighbour lists (at most half of
   the available host memory), then also the layers' outputs and their gradients for the whole cell (with the lists
   at most two thirds of it). The report names what was moved on its memory line.
5. From 2^20 atoms, when the layers' state fits neither the GPU nor that share of the host, two-layer models run the
   tiles in [the wavefront order](#the-wavefront), which holds the state of a moving band of tiles only.
6. A cell that fits in none of these ways at 4,096-atom tiles is refused before anything runs, with the memory it
   would need.

The free GPU memory is what other programs leave at that moment. Under WSL 2 CUDA's own figure leaves out most of
what Windows programs hold, so the engine also asks NVML and takes the smaller of the two.

```python
import loupmd as lm

result = lm.md("big.cif", model="omat0-medium", ensemble="NVT", temperature_K=300, steps=10_000)
print(result.plan["execution"], result.plan.get("receivers"))      # "tiled", 65536
```

```bash
loupmd md big.cif --model omat0-medium --temperature 300 --steps 10000 --capacity auto
```

The report says which mode the run took. 320,000 atoms on a 16 GB card:

```
Plan
  8 tiles of at most 65,536 atoms on GPU 0 (NVIDIA GeForce RTX 5060 Ti, ...) (the cell does not fit the card
  whole); the integrator runs on the device
  memory: the whole system would need 29.27 GiB; tiled 9.10 GiB planned, 11.80 GiB free on the card
```

and 1,080,000 atoms on the same card:

```
Plan
  27 tiles of at most 65,536 atoms on GPU 0 (NVIDIA GeForce RTX 5060 Ti, ...) (the cell does not fit the card
  whole); the integrator runs on the device
  memory: the whole system would need 98.63 GiB; tiled 8.43 GiB planned, 11.73 GiB free on the card; the tile
  lists' edges and the layer state in pinned host memory (5.70 GiB)
```

`capacity="whole"` forces the whole-cell call; a cell that does not fit then fails with the engine's out-of-memory
error. `capacity="tiled"` forces tiles. `lm.md` and the command line take the mode only; `lm.Simulation` also takes
`capacity=lm.config.CapacityConfig(mode="tiled", receivers=8192)`, which fixes their size. `memory_fraction` (default 0.9) is the share of the free GPU memory the plan may use. The planned figure
is what the call will hold, within a few percent (`docs/validation/capacity.md` in the repository).

## The wavefront

The plain tiled call needs every layer's output and its gradient for the whole cell: on the GPU while they fit,
else in pinned host memory. With OMAT-0 medium the host then holds 5.6 GiB at 1,080,000 atoms and 22 GiB at
5,000,000, lists included. When the host's share does not hold them either, two-layer models run the tiles in the
wavefront order instead. The tiles lie in layers along the first cell axis, and each tile's first pass runs just
before the first tile that needs its output. Its last pass runs just after the last tile that adds to it. Only a
moving band of tiles is held, in a pool of rows on the GPU.

`capacity="auto"` uses the wavefront from 2^20 atoms when the plan has 12 or more layers of tiles along the first
axis and the layers' state is not kept in host memory. `capacity="wavefront"` forces it. A model with three layers
(OMOL-0) is refused in that mode and uses the plain tiled call. The forces are those of the plain tiled call to
float32 rounding. The report then adds "the tiles run in the wavefront order".

## What fits a 16 GB card

Measured on an RTX 5060 Ti (16 GB, about 13 GiB free) in a machine with 39 GiB of host memory, with OMAT-0 medium
on cubic `SrTiO3` cells and `capacity="auto"` at the default memory fraction (`docs/validation/capacity.md` in the
repository has the full record):

| Atoms | What `auto` chose | Tiles (largest tile) | GPU peak | Pinned host memory | Seconds per force call |
|---|---|---|---|---|---|
| 121,945 | whole cell | 1 | 11.2 GiB (planned) | none | 0.197 |
| 135,000 | tiles | 8 (65,536) | 4.05 GiB | none | 0.246 |
| 320,000 | tiles | 8 (65,536) | 9.03 GiB | none | 0.580 |
| 1,080,000 | tiles, lists and layer state in host memory | 27 (65,536) | 8.34 GiB | 5.58 GiB | 2.164 |
| 2,560,000 | tiles, lists and layer state in host memory | 64 (49,152) | 11.65 GiB | 12.33 GiB | 5.142 |
| 5,000,000 | tiles, lists and layer state in host memory | 216 (32,768) | 11.33 GiB | 22.07 GiB | 10.105 |
| 8,008,065 | wavefront, lists in host memory | 2,016 (6,144) | 11.62 GiB | 5.94 GiB | 16.906 |
| 10,985,000 | wavefront, lists in host memory | 3,825 (4,096) | 11.58 GiB | 8.15 GiB | 24.206 |

121,945 atoms is the largest cube that runs whole there, and 10,985,000 the largest the plan accepts; 12,301,875
atoms are refused. With OMAT-0 small, 20,480,000 atoms ran (24.5 s per force call). Other models, cards and host
memory give other limits: the plan decides from what it measures on the machine.

## What the tiled mode runs

- **Integrators:** all of them: `NVE`, `Langevin`, `CSVR`, `NoseHooverChain`, `Berendsen` with or without a
  pressure, `LangevinBAOAB` and `MTK`. They run on the GPU with the whole-cell run's kernels; positions, velocities
  and forces stay on the card between steps, and under a barostat the cell, its momentum and the virial too.
- **Random stream:** the noise is the whole-cell run's, so a tiled Langevin or CSVR run follows the whole-cell
  trajectory to float32 force rounding, and a resumed run continues the stream and the thermostat's state.
- **With a dispersion term** the D3 energy is added on the host each step, so the integrator runs on the host in
  numpy, `NVE` and `Langevin` only, with its own Langevin noise stream (`LOUP_TILED_HOST_MD=1` forces this loop).
  That loop keeps the cell fixed, so a barostat is refused with a dispersion term. The report says where the
  integrator ran.
- **Fixed atoms and centre of mass:** both are handled as in the whole-cell run (`FixAtoms`, `fixcm`).
- **Neighbour lists:** each tile keeps its list across steps. All lists are rebuilt once any atom has moved half the
  skin. When the cell changes (a barostat), the strain since the lists were built takes part of the skin: they are
  rebuilt once an atom has moved, beyond what the strain and a uniform shift explain, more than
  (r<sub>list</sub> (1 − |F − I|) − r<sub>max</sub>) / 2, with F the cell's deformation since the build,
  r<sub>max</sub> the model's cutoff and r<sub>list</sub> = r<sub>max</sub> + skin.
- **Models:** models with two or more interaction layers and no long-range terms. `MACE-POLAR-1` always
  runs whole.
- **Restarts:** a restart records the mode, and a resume uses the same one. A whole-cell restart cannot be read by
  the tiled engine, and a tiled restart cannot be read by the whole-cell engine.

## In LAMMPS

The LAMMPS package ([`lammps/README.md`](https://github.com/FZWINGEL-UBT/LoupMD/blob/release/v1/lammps/README.md),
"Cells larger than the GPU") runs the same tiled mode, and `capacity auto` is
its default as well. Its pair style takes the keywords `capacity auto|whole|tiled|wavefront`, `receivers N` and
`memory F`, the counterparts of `capacity=`, `receivers=` and `memory_fraction=`:

```
pair_style loupmd capacity auto memory 0.9
```

- `pair_style loupmd` and `pair_style loupmd/kk` make one tiled call per step, with LAMMPS's integrator and fixes,
  barostats included: under `fix npt` the tile lists stay across the box's changes by the rule above.
- `run_style loupmd` runs the integrators listed above as the engine's tiled MD, `fix npt ... iso` (MTK) and
  `fix press/berendsen` included, with `fix setforce 0 0 0` as fixed atoms and `fix momentum` as `fixcm`.
- The log says at the first step whether the cell runs whole or in tiles, and why; at the end of a run it counts the
  tiled calls, the tiled MD steps and the list builds.
- Speed (RTX 5060 Ti, MACE-MH-1 on 50,820 and 98,415 atoms of SrTiO3, NVE): the pair style and the run style take
  the Python package's step time to within 2%. In a tiled run each output step costs about one more force call.
  The README has the table.

## Speed

From the table above, per atom and force call (RTX 5060 Ti, OMAT-0 medium): 1.6 us for the whole cell, 1.8 us in
tiles on the GPU, 2.0 us with the lists and the layer state in host memory, and 2.1 to 2.2 us in the wavefront
order. A tiled step therefore costs 1.1 to 1.4 times the whole-cell call per atom. An MD step is the force call plus
the integrator on the GPU: a short Langevin run took 585 ms per step at 320,000 atoms and 2.16 s at 1,080,000.

The first call of a run builds the tiles' neighbour lists and tunes the GEMMs for the tile shapes: 1.8 s at 135,000
atoms, 7.9 s at 1,080,000 and 79 s at 10,985,000. At 10,985,000 atoms the plan itself took 21 s.
