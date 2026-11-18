# Capacity records: RTX 5060 Ti

The largest `SrTiO3` cells that ran on one machine (`PLAN.md` S5-T11, the owner's go of 18 November 2026 for this
machine), and how the memory plan compared with what each call took. Nothing here was measured on another GPU.

- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB (16,311 MiB), compute capability 12.0, in WSL 2 (Ubuntu) on Windows 11.
  Windows programs held 2.3 to 2.8 GiB of the card during the runs, so 13.1 to 13.6 GiB were free.
- Host: the WSL virtual machine has 39 GiB of memory (38 GiB available).
- Structure: cubic `SrTiO3`, `n x n x n` cells of 5 atoms, every atom displaced by a normal deviate of 0.04 A. Skin
  0.5 A: 99.5 neighbours per atom within the list's 6.5 A.
- Settings: capacity `auto` at the default memory fraction 0.9, no stress. One building call, then kept calls
  (same positions; two, one for the two largest cells). Times are wall-clock times of `evaluate` /
  `evaluate_tiled` from Python.
- Date and code: 18 November 2026, `release/v1` with the memory plan of `PLAN.md` D87 and D88.

"Plan" is what `loup_plan_execution` predicted for the device before the call (`whole_bytes`, or
`tiled_global_bytes + tile_workspace_bytes`). "Call" is the call's own peak (`loup_get_execution`: the free memory
when it began less the least free memory during it). "NVML" is the highest use NVML showed during the run above
the use before the engine was made: it adds the engine's weights (0.15 GiB), and in a first call the scratch of the
GEMM timing. Host memory: the plan's `host_bytes` against the bytes the call pinned.

## MACE-OMAT-0 medium (generic kernels)

| Atoms | Mode the plan chose | Tiles (cap) | Device: plan / call / NVML (GiB) | Host: plan / pinned (GiB) | Kept call | Building call | Energy (eV/atom) | Largest force (eV/A) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40,000 | whole | 1 | 3.72 / - / 4.21 | 0 | 0.063 s (1.58 us/atom) | 2.1 s | -7.961154 | 6.219 |
| 121,945 | whole | 1 | 11.20 / - / 12.27 | 0 | 0.197 s (1.61 us/atom) | 4.7 s | -7.961221 | 5.902 |
| 135,000 | tiled | 8 (65,536) | 4.12 / 4.05 / 4.44 | 0 | 0.246 s (1.82 us/atom) | 1.8 s | -7.961276 | 6.088 |
| 320,000 | tiled | 8 (65,536) | 9.11 / 9.03 / 9.70 | 0 | 0.580 s (1.81 us/atom) | 3.1 s | -7.961251 | 5.906 |
| 1,080,000 | tiled, lists and layer state in host memory | 27 (65,536) | 8.43 / 8.34 / 8.91 | 5.76 / 5.58 | 2.164 s (2.00 us/atom) | 7.9 s | -7.961251 | 7.008 |
| 2,560,000 | tiled, lists and layer state in host memory | 64 (49,152) | 11.75 / 11.65 / 12.22 | 12.62 / 12.33 | 5.142 s (2.01 us/atom) | 18.3 s | -7.961216 | 7.719 |
| 5,000,000 | tiled, lists and layer state in host memory | 216 (32,768) | 11.42 / 11.33 / 11.79 | 21.87 / 22.07 | 10.105 s (2.02 us/atom) | 34.9 s | -7.961214 | 7.280 |
| 8,008,065 | wavefront, lists in host memory | 2,016 (6,144) | 11.65 / 11.62 / 11.92 | 6.54 / 5.94 | 16.906 s (2.11 us/atom) | 50.9 s | -7.961207 | 7.681 |
| 10,985,000 | wavefront, lists in host memory | 3,825 (4,096) | 11.67 / 11.58 / 11.92 | 8.41 / 8.15 | 24.206 s (2.20 us/atom) | 79.1 s | -7.961204 | 7.937 |

121,945 atoms (29 cells a side) is the largest cube the plan runs whole; 135,000 (30 a side) runs in tiles.
10,985,000 atoms (130 a side) is the largest cube the plan accepts at fraction 0.9; for 12,301,875 (135 a side) it
refuses: 11.9 GiB of state and workspace against 11.8 GiB allowed. The 10,985,000-atom plan itself took 21.5 s
and the run lowered the host's available memory by 15.2 GiB (pinned lists and the tile plan's own arrays).

## MACE-OMAT-0 small

| Atoms | Mode the plan chose | Tiles (cap) | Device: plan / call / NVML (GiB) | Host: plan / pinned (GiB) | Kept call | Building call | Energy (eV/atom) | Largest force (eV/A) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 320,000 | tiled | 8 (65,536) | 6.97 / 6.90 / 7.38 | 0 | 0.303 s (0.95 us/atom) | 2.1 s | -7.957742 | 5.917 |
| 1,080,000 | tiled | 45 (65,536) | 10.89 / 10.78 / 11.35 | 0 | 1.180 s (1.09 us/atom) | 5.1 s | -7.957740 | 7.049 |
| 2,560,000 | tiled | 96 (65,536) | 11.43 / 10.99 / 11.72 | 0 | 2.917 s (1.14 us/atom) | 7.9 s | -7.957704 | 7.849 |
| 5,000,000 | tiled, lists and layer state in host memory | 125 (49,152) | 11.43 / 11.33 / 11.77 | 9.02 / 8.66 | 5.630 s (1.13 us/atom) | 17.0 s | -7.957704 | 7.377 |
| 8,008,065 | tiled, lists and layer state in host memory | 133 (65,536) | 11.25 / 11.13 / 11.70 | 14.56 / 13.99 | 9.275 s (1.16 us/atom) | 31.1 s | -7.957698 | 7.812 |
| 20,480,000 | wavefront, lists in host memory | 2,080 (12,288) | 11.64 / 11.57 / 11.88 | 15.66 / 15.19 | 24.459 s (1.19 us/atom) | 88.1 s | -7.957699 | 9.194 |

20,480,000 atoms (160 cells a side) is the largest cell that was run, not the largest the plan would accept: larger
ones were not tried. Its plan took 34.0 s and the run lowered the host's available memory by 20.0 GiB.

## What limits the size

- **Whole systems:** the device. The plan counts the system's buffers and leaves room for the scratch of a first
  call; at fraction 0.9 that is 11.8 GiB of the 13.1 GiB free here.
- **Tiles with the layer state in host memory:** the host. That state is pinned (it cannot be swapped) and may take
  two thirds of the host memory available when the plan is made: 22 GiB at 5,000,000 atoms of OMAT-0 medium. Above
  that the plan keeps the layer state on the device in the wavefront's row pool.
- **The wavefront:** the device again: positions, forces and each tile's rows stay there (1.0 KiB per atom at these
  tile sizes), and at 12,301,875 atoms of OMAT-0 medium they no longer fit.

## How close the plan was

In the 13 tiled rows above the plan is 0.3% to 4.0% above the call's peak (0.3% to 1.1% in 11 of them). Of the
eight rows that pin host memory, seven have the plan within -0.9% to +4.2% of the pinned bytes. The eighth,
8,008,065 atoms of OMAT-0 medium, is 10% above: only the lists are pinned there, and the plan then counted them
with 10% of room; it now counts 3% (the room a pinned list buffer gets to grow), as in the last row of each table.
For whole systems `STATUS_AND_PROFILE.md` §6.35 compares the count with measured memory for five model families.
