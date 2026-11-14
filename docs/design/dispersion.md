# Dispersion in LoupMD: what to build

Research note for plan task S6-D01. Written 14 November 2026. **Status: approved by the owner on 14 November 2026
with the four decisions of §6.2.** No dispersion code exists in LoupMD yet.

## Summary

Build one real-space D3 engine that offers every standard damping, the three-body term, configurable cutoffs, a
smooth switch and a long-range tail correction, with named presets that reproduce what other packages compute. Add
the reciprocal-space sum of FourierD3 as a second variant for mid-size cells. Leave D4 and the density-based methods
out of 1.0. By the owner's decision the default is not a fixed cutoff: it is the fastest evaluation that is as
accurate as the conventional 40 bohr sum (§6.2).

The finding that shapes the design is cost. LoupMD's step is fast enough that a converged D3 sum with any existing
implementation would cost more than the model itself: FastMACE's recorded D3 speed puts a 40 bohr sum at two to
three times LoupMD's OMAT-0 medium step. Speed and the choice of cutoff are therefore the user-experience problem,
more than the list of dampings.

What was done: five web surveys (D3 reference definitions, D4, GPU implementations, the FourierD3 paper, methods
beyond pairwise sums and the models' training functionals), a read of FastMACE's own D3 and its recorded cost, and
CPU measurements with the reference code on six of our structures. What was not done: no GPU timing of any
dispersion code (the GPU was in use), no molecular-dynamics run, and FourierD3's code was not run.

## 1. Inventory

### 1.1 Methods

| Method | What it is | Needs beyond geometry | Verdict for LoupMD |
| --- | --- | --- | --- |
| D2 | pairwise C6/r^6 with fixed coefficients | nothing | superseded by D3; not planned |
| D3, zero damping | pairwise C6 and C8 terms, C6 from the coordination number | nothing | build |
| D3, Becke-Johnson damping | same sum, rational damping; the Grimme group's recommendation ("does not lead to artificial repulsive forces") | nothing | build; the default |
| D3M(0), D3M(BJ) | refitted parameters; D3M(0) adds one parameter | nothing | build: same kernel, other numbers |
| D3(op), D3(CSO), D3(Z) | further dampings in the reference code; few parametrisations (CSO: eight functionals; Z: none) | nothing | not at 1.0 |
| ATM three-body term | Axilrod-Teller-Muto triple sum, zero-damped, C9 from the pair C6 | nothing | build as an option |
| D4 | D3's successor with charge-dependent polarisabilities; BJ damping and ATM on by default | atomic charges from an electronegativity-equilibration solve with Ewald summation | not at 1.0 (§6.3) |
| Reciprocal-space D3 (FourierD3) | the damped pair sum on a particle mesh, no real-space cutoff | nothing | build second |
| MBD, TS, XDM, dDsC | many-body or density-derived coefficients | Hirshfeld volumes or the density; machine-learned substitutes exist only as research models; MBD scales cubically | out of scope |
| VV10, rVV10, vdW-DF | non-local density functionals | the electron density | not applicable |

### 1.2 Implementations

| Code | Damping | Three-body | Default cutoffs (pair / coordination number) | Smoothing | Periodic, stress | Licence |
| --- | --- | --- | --- | --- | --- | --- |
| simple-dftd3 1.7.0 (reference, CPU) | zero, BJ, both modified, op, CSO, Z | yes; on in the Python and ASE interfaces, off in the CLI | 60 / 40 bohr; three-body 40 bohr | optional quintic switch (off by default); optional Ewald sum for the pair term since 1.6.0 | yes, yes | LGPL-3.0-or-later |
| torch-dftd 0.5.3 ("not actively maintained") | zero (default), BJ, both modified | yes, needs cupy on the GPU | 95 / 40 bohr | optional polynomial over the last bohr; not for the three-body term | yes, yes | MIT |
| `mace_mp(dispersion=True)` (mace-torch, through torch-dftd) | BJ (default), others | no | 40 / 40 bohr | none | yes, yes | MIT |
| FastMACE `fm.D3` | BJ only | refused | 40 / 40 bohr | none | yes, yes | MIT |
| SevenNet D3 (CUDA, LAMMPS and Python) | zero, BJ, both modified | "not implemented" | 9000 / 1600 bohr^2 (95 / 40 bohr) | none documented | yes, yes | file carries LAMMPS's GPL header |
| LAMMPS `dispersion/d3` (since Feb 2025) | zero, BJ, both modified | "currently not possible" | none; the user gives both | not confirmed | yes | GPL |
| NVIDIA ALCHEMI Toolkit-Ops 0.5.0 | BJ only | "not included" | the caller's neighbour list; its model wrapper defaults to 15 A | S5 switch window | yes, virial | Apache-2.0 |
| fairchem / UMA, ORB v3, TorchSim (all through ALCHEMI) | BJ | no | 15 A with smoothing fraction 0.2; 22 A; 95 bohr | per package | yes | per package |
| tad-dftd3 0.7.0 | not recorded | molecules only | 60 / 40 bohr | none | release: molecules only | Apache-2.0 |
| FourierD3 0.1.0 (PyTorch) | BJ only | no | no pair cutoff; modified coordination number that vanishes at 6 A | not needed | 3D periodic only; stress by autograd | MIT |
| FourierD3, NVIDIA-BioNeMo (JAX front end, CUDA engine) | BJ only | no | as above | not needed | strain gradient | Apache-2.0; "not actively maintained" |
| LASP-D3 (CUDA) | not read | not read | 12-40 A in the FourierD3 paper's comparison | not read | yes, yes | MIT |
| `CheukHinHoJerry/torch-dftd` fork | as torch-dftd | as torch-dftd | as torch-dftd | as torch-dftd | adds a skin neighbour list | skin-list code under the Academic Software License |

What this shows:

- Every GPU implementation that is paired with machine-learned potentials today is two-body D3. Becke-Johnson
  damping is universal; the other three dampings exist in torch-dftd, SevenNet and LAMMPS; the three-body term on a
  GPU exists only in torch-dftd.
- **The defaults disagree widely.** The pair cutoff is 95 bohr in torch-dftd, SevenNet and TorchSim, 60 bohr in the
  reference code, 40 bohr in `mace_mp` and FastMACE, 22 A in ORB v3 and 15 A with smoothing in fairchem. A user who
  switches package gets different energies and pressures without changing any "setting". LoupMD should name its
  presets after what they reproduce.
- Code that cannot be copied into an MIT project: the reference code (LGPL), SevenNet's and LAMMPS's pair styles
  (GPL), the fork's skin list (ASL). MIT or Apache code may be reused with attribution: torch-dftd (whose reference
  table FastMACE already ships), ALCHEMI, both FourierD3 codes, LASP-D3.

## 2. What the models need

Functional and dispersion content of the training data, from FastMACE's label table
(`fastmace/model/labels.py`) and the model papers and cards.

| Models and heads | Training functional | Dispersion already in the data | D3 on top |
| --- | --- | --- | --- |
| MP-0 medium and large, MP-0b3, MPA-0, OMAT-0 small and medium, MatPES-PBE; MH-1 heads `omat_pbe`, `mp_pbe_refit_add`, `oc20_usemppbe` | PBE (with +U where the dataset has it) | no | allowed; PBE parameter set |
| MatPES-r2SCAN; MH-1 head `matpes_r2scan` | r2SCAN | no | allowed; r2SCAN parameter set, which exists for BJ only |
| OFF23 small, medium, large, OFF23b, OFF24; MH-1 head `spice_wB97M` | wB97M-D3(BJ) | yes, D3(BJ) | refuse: it would count dispersion twice |
| OMOL-0; MH-1 head `omol` | wB97M-V | yes, VV10 | refuse |
| POLAR-1 S, M, L | wB97M-V according to the MACE documentation | yes, VV10 | refuse. FastMACE's table lists the functional as unknown; settle this when the table is ported (S2-T04) |

Practice elsewhere: `mace_mp` applies D3 only on request, with BJ damping, the PBE set and 40 bohr for both
cutoffs. The MH-1 model card says to "use D3 corrections for systems with dispersions" on the `omat_pbe` head. The
MACE documentation gives no guidance for the other families. Note that `mace_mp` would apply the PBE set to an
r2SCAN model unless the user overrides it; a default that follows the label avoids that.

Reference parameters (s6 = 1): PBE BJ s8 0.7875, a1 0.4289, a2 4.4407; PBE zero s8 0.722, rs6 1.217; r2SCAN BJ
s8 0.7898, a1 0.4948, a2 5.7308; no zero-damping set for r2SCAN in the reference table.

## 3. Measurements

Reference code simple-dftd3 (Python package `dftd3` 1.6.0), float64, CPU. Six cells, each rattled by 0.05 A so that
forces do not vanish by symmetry: NbCl5 (72 atoms), naphthalene (36), urea 2x2x2 (128), the 216-molecule water box
(648), graphite 4x4x2 (128), SrTiO3 3x3x3 (135). PBE parameters. Script, raw output and JSON:
`docs/design/dispersion-measurements/`. Command:
`python measure_d3_reference.py results.json <folder holding FastMACE and ResearchModule>`.

### 3.1 How far the pair sum must reach

D3(BJ) two-body term; error against the same sum at 150 bohr.

| Structure | Dispersion energy, meV/atom | Dispersion pressure, GPa | Error at 12 bohr (6.4 A) | at 30 bohr | at 40 bohr | at 60 bohr | at 95 bohr | Neighbours per atom at 40 bohr |
| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| NbCl5 | -191.8 | -1.20 | 26.0 meV, 0.30 GPa | 1.40, 0.016 | 0.58, 0.007 | 0.16, 0.002 | 0.03, 0.000 | 1,381 |
| naphthalene | -80.4 | -1.24 | 7.0, 0.22 | 0.42, 0.013 | 0.17, 0.005 | 0.05, 0.002 | 0.01, 0.000 | 4,180 |
| urea | -59.8 | -1.13 | 4.8, 0.17 | 0.30, 0.011 | 0.12, 0.004 | 0.04, 0.001 | 0.01, 0.000 | 4,369 |
| water | -33.4 | -0.59 | 2.6, 0.08 | 0.16, 0.005 | 0.06, 0.002 | 0.02, 0.001 | 0.00, 0.000 | 3,972 |
| graphite | -155.2 | -1.04 | 14.4, 0.35 | 0.86, 0.020 | 0.35, 0.008 | 0.10, 0.002 | 0.02, 0.001 | 4,516 |
| SrTiO3 | -265.6 | -3.66 | 61.5, 1.66 | 2.72, 0.075 | 1.06, 0.029 | 0.30, 0.008 | 0.06, 0.002 | 3,336 |

Errors are meV per atom and GPa. Force errors at 40 bohr are 0.005 to 0.41 meV/A (largest in graphite).

- A cutoff at the model's own 6 A is unusable: up to 61 meV/atom and 1.7 GPa.
- 40 bohr, the `mace_mp` default, leaves at most 1.1 meV/atom and 0.03 GPa on these cells.
- The work grows with the cube of the cutoff: 40 bohr means 1,400 to 4,500 neighbours per atom, 95 bohr means
  18,000 to 60,000.

### 3.2 The three-body term

| Structure | Three-body energy, meV/atom | Share of the two-body energy | Pressure, GPa | Largest force, meV/A | Reached with a 20 bohr cutoff | with 30 bohr | Reference CPU time at 40 bohr (pair sum: at most 0.02 s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NbCl5 | +7.6 | 4.0% | +0.09 | 0.8 | 93% | 99% | 1.0 s |
| naphthalene | +4.0 | 4.9% | +0.12 | 2.8 | 97% | 100% | 3.6 s |
| urea | +3.2 | 5.4% | +0.12 | 1.6 | 98% | 100% | 14.4 s |
| water (20 bohr) | +1.5 | 4.6% | +0.05 | 3.5 | n/a | n/a | n/a |
| graphite | +9.2 | 5.9% | +0.15 | 16.1 | 95% | 99% | 16.6 s |
| SrTiO3 | +18.9 | 7.1% | +0.51 | 0.3 | 89% | 98% | 9.0 s |

- The three-body term is worth 4 to 7% of the dispersion energy and up to 0.5 GPa. That is ten times more than the
  difference between a 40 and a 95 bohr pair cutoff. No package for machine-learned potentials turns it on by
  default, and results labelled "D3(BJ)" in this field leave it out.
- It is expensive: the triple count grows with the square of the neighbour count. It converges faster than the pair
  sum, so its own cutoff can be shorter: 30 bohr reaches 98 to 100%, 20 bohr 89 to 98%.

### 3.3 The dampings are different models, not settings

Two-body energy in meV/atom with PBE parameters at 60 bohr:

| Structure | BJ | zero | BJ modified | zero modified |
| --- | ---: | ---: | ---: | ---: |
| NbCl5 | -191.6 | -137.3 | -351.0 | -450.4 |
| naphthalene | -80.3 | -59.0 | -90.7 | -144.5 |
| urea | -59.8 | -52.7 | -61.1 | -74.2 |
| water | -33.4 | -36.4 | -31.5 | -32.1 |
| graphite | -155.1 | -95.2 | -188.3 | -299.9 |
| SrTiO3 | -265.3 | -182.0 | -814.6 | -728.7 |

The modified sets give three times the BJ energy on SrTiO3 (and -15 GPa against -3.7 GPa). Offering every damping
is cheap, since the kernel is the same, but a user should have to ask for anything other than BJ.

### 3.4 The coordination-number cutoff

Cutting the coordination-number sum hard, against the 40 bohr reference:

| Structure | at 11.3 bohr (6 A): energy, largest force | at 20 bohr |
| --- | --- | --- |
| NbCl5, SrTiO3 | below 0.001 meV/atom, below 0.003 meV/A | the same |
| urea | 0.012 meV/atom, 0.29 meV/A | 0.008, 0.21 |
| water | 0.006, 0.58 | 0.004, 0.42 |
| naphthalene | 0.060, 0.95 | 0.036, 0.55 |
| graphite | 0.469, 1.71 | 0.255, 0.94 |

The effect is small against a model's own force error but not at rounding level, and it is largest for sp2 carbon.
FourierD3 replaces the counting function by one that reaches zero at 6 A; its paper reports a difference of 0.0025
in the pair function. The cutoff should be a visible option with the reference value as default.

### 3.5 Cost

Nothing here was timed on a GPU for this note. The figures on record:

- **FastMACE's D3 on the RTX 5060 Ti** (`FastMACE/docs/validation/phases/3b-T06.md`): its three passes take about
  4.2 ms per step at 1.8 million pairs on NbCl5 (1,944 atoms, 40 bohr), about 2.4 ns per pair and step. That was 24%
  of a 20.7 ms MH-1 step.
- **Against LoupMD's step:** OMAT-0 medium runs NbCl5 in about 1.85 ms and SrTiO3 (5,000 atoms) in about 8.3 ms.
  At FastMACE's speed the NbCl5 sum costs 2.3 times the model. SrTiO3 has about 8.3 million pairs at 40 bohr, which
  would be about 20 ms, 2.4 times the model. Both are arithmetic on recorded numbers, not measurements.
- **FourierD3's paper** (CUDA version, H100, neighbour list excluded): 1 to 2 ms per evaluation up to 10,000 atoms
  for water and benzene, about 8 ms for a seven-element alloy at 2,500 to 10,000 atoms, about 75 ms at 25,000. Its
  mesh carries one channel per element and rank, so cost rises with the number of elements. On an RTX 5060 Ti these
  times would be longer.
- **Three-body term:** at 20 bohr SrTiO3 has roughly eight times as many triples per atom as it has pairs at
  40 bohr, and a triple costs more than a pair. At 40 bohr the ratio is in the hundreds. (Both from the neighbour
  counts, taking 15/32 of a sphere's neighbour pairs to lie within the cutoff of each other.) In MD at 20 bohr it
  is likely to cost about as much as the model or more.

| SrTiO3, 5,000 atoms, estimated D3 cost per step | 40 bohr | 30 bohr | 20 bohr |
| --- | ---: | ---: | ---: |
| Pairs | 8.3 M | 3.5 M | 1.0 M |
| At FastMACE's recorded speed | 20 ms | 8.4 ms | 2.5 ms |
| With a kernel five times faster | 4 ms | 1.7 ms | 0.5 ms |
| Truncation error (§3.1) | 1.1 meV, 0.03 GPa | 2.7 meV, 0.08 GPa | 9.5 meV, 0.27 GPa |

For the pair sum to stay under 20% of the SrTiO3 step at 40 bohr, the kernel must reach about 0.2 ns per pair and
step, twelve times FastMACE's recorded speed. Whether that is reachable is the first thing the build has to measure.

### 3.6 Energy conservation at the cutoff

Not measured in MD. An estimate from §3.1: a pair at 40 bohr carries about 6e-7 eV in SrTiO3, and at 300 K about
5,000 pairs cross the cutoff per step in a 5,000-atom cell. That gives an energy noise of about 3e-4 meV/atom after
1 ps, growing with the square root of time, against the engine's drift criterion of 0.05 meV/atom/ps. At 20 bohr a
pair carries 64 times more, and the noise reaches about 0.01 meV/atom after 1 ps. Smoothing therefore matters for
short cutoffs and hardly at 40 bohr. The reference code's maintainers say the same in general terms ("jumps in the
energy when atom distances move over the boundaries") and note that smoothing "does not remove the truncation
error".

## 4. Large cells and the tiled mode

- **No stored pair list.** At 40 bohr a half list is 3 to 9 KB per atom, which is 3 to 9 GB at one million atoms.
  Large cells need a kernel that walks a cell grid.
- **Real-space D3 fits the three-sweep driver.** D3 needs three passes: coordination numbers, pair energies with
  the derivative of each atom's energy by its coordination number, then forces. The first two leave 4 bytes per
  atom each as whole-system state. A tile needs the positions, species and coordination numbers of atoms within
  21 A, not their model features: a wider but much thinner halo than the model's.
- **The mesh sum does not reach the largest cells on one GPU.** FourierD3 puts the whole damped kernel on the mesh,
  with no real-space part, so the mesh must resolve the damping length, and it carries elements times rank
  channels. At the accuracy of the paper's 32^3 row, a million SrTiO3 atoms would need roughly 5 GB of mesh before
  FFT workspace (my estimate). A hundred million atoms would need a distributed transform.
- **Recommendation for tiled runs:** the real-space sum with a cell walk, the tail correction of §5 for what lies
  beyond the cutoff, and the long-range part updated every few steps if the measurement of §6.1 supports it.

## 5. Proposed user experience

1. **One argument, three levels.** `dispersion=None` (default, off), a preset name, or an `lm.Dispersion` record.
   The same in the CLI (`--dispersion`) and in TOML (`[dispersion]`).
2. **Presets are named for what they reproduce.**

   | Preset | Meaning | Reproduces |
   | --- | --- | --- |
   | `"d3bj"` | BJ, the model's functional, no three-body term. The pair sum is evaluated by the fastest method that meets the accuracy contract of §6.2, chosen for the system at setup | the same model as `mace_mp(dispersion=True)` and FastMACE's `"d3bj"`, at least as close to the converged sum as their 40 bohr cutoff |
   | `"d3bj-mace"` | BJ, the model's functional, a plain real-space sum cut at 40 bohr for pairs and coordination numbers, no smoothing | the numbers of `mace_mp(dispersion=True)` and FastMACE's `"d3bj"`, for direct comparisons |
   | `"d3bj-atm"` | `"d3bj"` plus the three-body term with a 20 bohr smoothed cutoff | nothing else; the affordable three-body setting |
   | `"d3-reference"` | BJ, 60 bohr pairs, 40 bohr coordination and three-body cutoffs, three-body term on | simple-dftd3's Python defaults |
   | `"d3zero"`, `"d3bjm"`, `"d3zerom"` | the other dampings with `"d3bj"`'s cutoffs | torch-dftd, SevenNet and LAMMPS with the same cutoffs |
   | `"auto"` | `"d3bj"` if the model's functional has a parameter set and its data has no dispersion; an error otherwise | |

3. **The record exposes everything.** `method` (`"d3"`), `damping` (`"bj"`, `"zero"`, `"bjm"`, `"zerom"`),
   `functional` or the raw parameters, `three_body` with `three_body_cutoff_bohr`, `pair_cutoff_bohr`,
   `cn_cutoff_bohr`, `smoothing` (`"none"` or a switch width), `long_range` (`"auto"`, `"none"`, `"tail"`, `"split"`,
   `"mesh"`) with the mesh options, and `update_every` for the long-range part if §6.1 adopts it.
4. **Every method can be chosen by name, and the automatic choice never overrides the user** (owner, 14 November
   2026: a method may be inaccurate for a specific system).
   - `long_range="auto"` is the default: the fastest method that passes the check below. Any explicit value is used
     as given, with no substitution, and the report says "chosen by the user".
   - `auto_methods=(...)` restricts what `"auto"` may pick, for a user who trusts some methods but not others.
   - **`"auto"` checks on the user's own structure, not only on the validation set.** At setup it evaluates each
     candidate once on the starting structure against the plain real-space sum and drops any candidate outside the
     accuracy contract. The report lists every candidate with its measured deviation and its time. The check
     repeats when the cell or the composition changes by an event, and on request.
   - Each method carries its known limits, shown by `loupmd dispersion list` and in the documentation:

     | Method | Exact up to | Known limits |
     | --- | --- | --- |
     | plain real-space sum (`"none"`) | its cutoff | cost grows with the cube of the cutoff |
     | tail correction (`"tail"`) | a uniform density beyond the cutoff | wrong for slabs, pores, interfaces and anything with vacuum; corrects energy and pressure, not forces. `"auto"` considers it only for cells whose density is uniform on the scale of the cutoff |
     | split sum (`"split"`) | the mesh tolerance | 3D periodic; accuracy set by mesh spacing |
     | mesh sum (`"mesh"`) | the mesh tolerance | Becke-Johnson only; a modified coordination number that vanishes at 6 A (§3.4: up to 1.7 meV/A in graphite); cost grows with the number of elements |
     | long-range part every n steps (`update_every`) | n = 1 | long-range forces lag by up to n steps; never chosen by `"auto"` above the n the bake-off validates | This covers the union of what users
   of the packages in §1.2 can set today, plus the three-body cutoff and the long-range choice.
5. **Refusals, each with the reason and the way out:** a model whose data already contains dispersion; a model
   loaded from a file with no known functional, unless the functional or the parameters are given; a damping with no
   parameter set for the functional (zero damping with r2SCAN); a functional that differs from the model's gives a
   warning, as in FastMACE.
6. **The report says what was computed and what it cost:** one line with damping, parameters, cutoffs, neighbours
   per atom and the share of the step, and a warning with the cheaper presets when dispersion exceeds the model's
   own cost.
7. **A check the user can run:** `loupmd validate --dispersion` compares energy, forces and stress with
   simple-dftd3 for the same settings. That package covers every damping, the three-body term, the cutoffs, the
   switch and, since 1.6.0, an Ewald sum, so it is a complete reference.
8. **`loupmd dispersion list`** prints the presets, the dampings, the summation methods with their limits, and the
   functionals with their parameters and sources.

## 6. Recommendation

### 6.1 Build, in this order

| Step | What | Effort (rough) |
| --- | --- | --- |
| B1 | Kernel bake-off on the RTX 5060 Ti. Time and measure the error of each candidate for the pair sum: a plain real-space sum at 40 bohr (receiver-owned kernel with a cell walk, float32 pair arithmetic with float64 sums); a shorter cutoff with the tail correction; a split sum (damped short range in real space, the smooth long range on a coarse mesh); the long-range part updated every few steps. The default is the fastest that meets the accuracy contract of §6.2 | 1.5 weeks |
| B2 | Framework and the real-space pair term with four dampings, raw parameters, both cutoffs, the smooth switch, stress and the energy-only path | 2 weeks |
| B3 | The long-range methods the bake-off keeps (tail correction, split sum), each checked against the 150 bohr sum | 1 week |
| B4 | Reciprocal-space sum for BJ, after FourierD3, with a tighter default than the public code's 1.2 A mesh spacing (which sits near the paper's 16^3 row: 0.5 to 1.5 meV/atom) | 2.5 weeks |
| B5 | Automatic choice at setup: check each candidate on the user's starting structure against the plain sum, time those that pass and keep the fastest, as the engine does for its GEMM plans; every candidate's deviation and time appear in the report; an explicit choice is never replaced | 1 week |
| B6 | Three-body term with its own smoothed cutoff, off by default | 1.5 weeks |
| B7 | Integration: NPT, Monte Carlo, restart, the tiled driver; `loupmd validate --dispersion`; documentation | 2 weeks |

About 11.5 weeks, against the 7.5 the plan assumed before this research.

### 6.2 Decisions (owner, 14 November 2026)

1. **The default is as fast as possible at the same accuracy.** The default is not a fixed cutoff. It is the
   fastest way to evaluate the pair sum that is as accurate as the conventional setting.
   - **Accuracy contract.** On every structure of the dispersion validation set, the default's error against the
     converged pair sum (the reference code at 150 bohr) is no larger than the error of a plain 40 bohr real-space
     sum, in energy per atom, in pressure and in the largest force. In MD it meets the engine's energy-drift
     criterion (0.05 meV/atom/ps).
   - **Validation set.** The six cells of §3 plus one metal and one layered compound other than graphite.
   - **Speed.** Measured in ms per step on the RTX 5060 Ti for NbCl5 (1,944 atoms), SrTiO3 (5,000), the 5,184-atom
     water box and one tiled cell; chosen per system at setup where candidates differ.
   - **Fallback.** If nothing beats it within the contract, the default is the 40 bohr real-space sum.
   - `"d3bj-mace"` keeps the plain 40 bohr sum available by name, for comparisons with other packages.
   - **Explicit choice.** The automatic mode may decide by speed, but every method stays selectable by name and an
     explicit choice is never replaced, because a method can be inaccurate for a specific system. The automatic
     mode applies the contract to the user's own structure before it picks (§5, item 4).
2. **Three-body term:** offered, off by default, with a 20 bohr smoothed cutoff in its preset. It changes energies
   by 4 to 7% of the dispersion energy, so "on by default" would make LoupMD's D3 differ from everyone else's.
3. **Mesh sum in 1.0.** It helps cells of ten thousand to a million atoms and is a candidate for the default there;
   it does not serve the largest tiled cells and covers BJ only.
4. **D4 left out of 1.0** (below).

### 6.3 Not at 1.0, with reasons

- **D4.** No periodic GPU implementation exists to build on or compare with: the Grimme group's PyTorch version
  states that periodic boundary conditions are not implemented, and ALCHEMI, LAMMPS and SevenNet have D3 only. The
  reference charge solve is a dense matrix factorisation, so thousands to millions of atoms need an iterative or
  mesh solver that would have to be designed. The evidence that it helps machine-learned potentials is thin: the
  one comparison found (M3GNet with D3 and with D4) prefers D3. A user who needs D4 for single points can sum the
  `dftd4` ASE calculator with `lm.calc`; that recipe goes into the documentation.
- **D2, D3(op), D3(CSO), D3(Z).** Rare; the raw-parameter path leaves room to add them.
- **MBD, TS, XDM, dDsC, VV10.** They need the density or a trained substitute, and MBD scales cubically.
- **Adapters to outside packages at run time.** They would bring PyTorch back into a run.

### 6.4 Changes made to the plan

Stage 6D of `PLAN.md` now follows this note: it starts with the kernel bake-off (B1), defines the default by the
accuracy contract, keeps the mesh sum in 1.0, adds tasks for the long-range methods and for the automatic choice,
sets the three-body preset's cutoff to 20 bohr, and lists D4 among the cuts. The stage grows from about 7.5 to about
11.5 weeks.

## 7. Not done, and open

- No GPU timing of any dispersion code. Every cost figure for LoupMD is an estimate from recorded or published
  numbers.
- No MD run: the energy-conservation statement of §3.6 is an estimate.
- Measurements used PBE parameters only, six cells, one rattle each; no metal and no layered compound other than
  graphite.
- FourierD3's code was not run; its accuracy and timings are the paper's, some read off log-scale figures. Its CUDA
  version's repository shows one commit.
- The LASP-D3 paper could not be opened; its speed claim is from a search snippet.
- The installed reference package is 1.6.0; its Ewald option and its smooth switch were not exercised.
- POLAR-1's training functional is taken from the MACE documentation and contradicts FastMACE's table.
- The web facts were collected by five readers on 14 November 2026. Quotes they marked as read directly are given
  as such above; the rest should be re-read before they go into user documentation.

## 8. Sources

Read on 14 November 2026.

- D3 definitions and reference code: DFT-D3 manual V3.2
  (`https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/man.pdf`); simple-dftd3 repository and documentation
  (`https://github.com/dftd3/simple-dftd3`, `https://dftd3.readthedocs.io/en/latest/`), including
  `assets/parameters.toml`, `src/dftd3/cutoff.f90` and issues 182 and 199.
- D4: `https://github.com/dftd4/dftd4`, `https://dftd4.readthedocs.io/en/latest/`,
  `https://pypi.org/project/tad-dftd4/`, `https://github.com/grimme-lab/multicharge`,
  `https://arxiv.org/html/2411.01197v1` (M3GNet with D3 and D4).
- GPU implementations: `https://github.com/pfnet-research/torch-dftd`,
  `https://github.com/ACEsuit/mace` (`mace/calculators/foundations_models.py`),
  `https://sevennet.readthedocs.io/en/latest/user_guide/d3.html`,
  `https://docs.lammps.org/pair_dispersion_d3.html`,
  `https://nvidia.github.io/nvalchemi-toolkit-ops/main/userguide/components/dispersion.html`,
  `https://github.com/dftd3/tad-dftd3`, `https://github.com/LipidL/LASP-D3`,
  `https://github.com/CheukHinHoJerry/torch-dftd`.
- Reciprocal-space sum: arXiv:2607.15103 v3 (Valeeva, Ho, Geiger, Pellegrini, Csanyi, Kucukbenli, Ortner),
  `https://github.com/vicvaleeva/FourierD3`, `https://github.com/NVIDIA-BioNeMo/FourierD3`.
- Beyond pairwise sums: `https://arxiv.org/abs/2308.03140` (libMBD), `https://arxiv.org/abs/2602.22086` (MBD-ML).
- Models: `https://arxiv.org/abs/2401.00096` (MACE-MP-0), `https://arxiv.org/abs/2312.15211` (MACE-OFF),
  `https://arxiv.org/abs/2410.12771` (OMat24), `https://arxiv.org/abs/2503.04070` (MatPES),
  `https://arxiv.org/abs/2505.08762` (OMol25), `https://arxiv.org/abs/2510.25380` (MACE-MH-1),
  `https://huggingface.co/mace-foundations/mace-mh-1`,
  `https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html`.
- Local: `FastMACE/fastmace/longrange/d3.py`, `FastMACE/fastmace/model/labels.py`,
  `FastMACE/docs/validation/phases/3b-T06.md`, `LoupMD/STATUS_AND_PROFILE.md` (step times), and the measurements in
  `docs/design/dispersion-measurements/`.
