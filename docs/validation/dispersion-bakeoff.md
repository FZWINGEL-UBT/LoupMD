# Dispersion kernel bake-off (plan task S6-D02)

16 November 2026. RTX 5060 Ti only (`sm_120`); nothing here was measured on another GPU.

## What was compared

The D3(BJ) two-body sum with PBE parameters (s6 1, s8 0.7875, a1 0.4289, a2 4.4407), no three-body term.

- **Systems**: NbCl5 3x3x3 (1,944 atoms), SrTiO3 10x10x10 (5,000) and the water box 216 2x2x2 (5,184), the
  benchmark cells of `loupmd.bench.suites`, each rattled by 0.05 A (seed 1) so that forces do not vanish by
  symmetry.
- **Reference**: simple-dftd3 1.6.0 (float64, CPU, used as an external program) with a 150 bohr pair cutoff and its
  40 bohr coordination-number cutoff. Reference dispersion energies: -191.73, -265.60 and -33.40 meV/atom;
  pressures -1.201, -3.659 and -0.593 GPa. These agree with the design note's 72-, 135- and 648-atom cells.
- **Accuracy contract** (`docs/design/dispersion.md` §6.2): a candidate passes on a system when its error against
  the reference is no larger than the plain 40 bohr sum's in energy per atom, in pressure and in the largest force
  (norm of the per-atom force difference).
- **Prototype**: `src/dispersion/d3_proto.cu`, built and driven by `tools/dev/d3_bakeoff.py` (not part of
  `libloupmd.so`). Tables from torch-dftd 0.5.1 (MIT): C6 references, covalent radii, r2r4.
- **Times**: CUDA events around each pass, mean of 20 evaluations after one untimed round, one run per candidate.
  "ms" is coordination numbers + pair pass + coordination-number force pass. The binning (bins, a radix sort of the
  atoms, bin starts on the device, 0.06 to 0.19 ms) is reported apart, since the engine would share it with list
  rebuilds.

Commands (WSL, runtime environment of `AGENTS.md`; `LOUPMD_BENCH_STRUCTURES` must find `NbCl5_R2SCAN.cif` and
`water_216.extxyz`):

```
python tools/dev/d3_bakeoff.py build
python tools/dev/d3_bakeoff.py check
python tools/dev/d3_bakeoff.py reference
python tools/dev/d3_bakeoff.py time
python tools/dev/d3_bakeoff.py run docs/validation/dispersion-bakeoff.json
```

Raw numbers: `docs/validation/dispersion-bakeoff.json`.

## The prototype against simple-dftd3 at the same cutoffs

`check`, on the rattled 72-atom NbCl5, SrTiO3 3x3x3 and water 216 cells, at 40/40 and 20/20 bohr (pair / CN):
worst differences 9.6e-6 meV/atom (2.6e-7 relative), 4.7e-4 meV/A and 1.2e-7 GPa. This is float32 pair
arithmetic with float64 sums. S6-D04's tolerances (1e-6 eV/atom, 1e-5 eV/A, 1e-7 eV/A^3) hold with a wide
margin.

## Kernel layout

Both layouts own each receiving atom with one warp and walk a grid of bins in fractional coordinates. A bin offset
along an axis maps to a bin and an image shift, so the cell walk covers cutoffs larger than half the cell. They give
the same results to 5e-7 eV and 4e-7 eV/A.

| Layout | NbCl5 (CN, pairs, CN forces) | SrTiO3 | water |
| --- | --- | --- | --- |
| v0: atom order, one bin per inner loop, the 5x5 C6 contraction per pair | 0.71, 2.53, 2.11 ms | 1.66, 6.77, 5.79 | 1.75, 7.83, 6.69 |
| v1: atoms in bin order, rows of bins contiguous in memory, per-receiver C6 projections in shared memory | 0.33, 1.14, 0.97 ms | 0.64, 4.22, 3.69 | 0.70, 4.86, 4.24 |

v1 is used below. On SrTiO3 the pair pass takes 0.25 ns per directed pair: 16.7 million directed pairs in 4.2 ms.
The whole sum takes about 1 ns per pair. The design note puts FastMACE's recorded speed at 2.4 ns per pair
(§3.5): 4.2 ms on NbCl5, against 2.48 ms here.

## Candidates

Errors against the 150 bohr reference: energy in meV/atom, pressure in GPa, largest force error in meV/A.

| System | Candidate | ms | Energy | Pressure | Force | Contract |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| NbCl5 | plain 40, CN 40 | 2.48 | 0.583 | 0.0065 | 0.0066 | baseline |
| | plain 40 + tail | 2.48 | 0.015 | 0.0002 | 0.0066 | pass |
| | plain 40, CN 30 | 2.01 | 0.583 | 0.0065 (+1.7e-6) | 0.0065 | **fail** by 1.7e-6 GPa |
| | plain 40, CN 20 | 1.78 | 0.583 | 0.0065 (+3.9e-6) | 0.0064 | fail by 3.9e-6 GPa |
| | plain 30 + tail | 1.96 | 0.037 | 0.0004 | 0.024 | fail (force) |
| | plain 20 + tail | 1.68 | 0.143 | 0.0017 | 0.411 | fail (force) |
| | shell 25-40 every 5 steps | 2.27 | | | stale +0.048 | fail (force) |
| SrTiO3 | plain 40, CN 40 | 8.42 | 1.060 | 0.0289 | 0.0066 | baseline |
| | plain 40 + tail | 8.42 | 0.036 | 0.0010 | 0.0066 | pass |
| | plain 40, CN 30 | 6.68 | 1.060 | 0.0289 | 0.0066 | pass |
| | plain 40, CN 20 | 5.62 | 1.060 | 0.0289 | 0.0066 | pass |
| | plain 30 + tail | 6.42 | 0.048 | 0.0014 | 0.030 | fail (force) |
| | plain 20 + tail | 5.23 | 0.276 | 0.0079 | 0.433 | fail (force) |
| | shell 25-40 every 5 steps | 7.38 | | | stale +0.027 | fail (force) |
| water | plain 40, CN 40 | 9.88 | 0.064 | 0.0020 | 0.392 | baseline |
| | plain 40 + tail | 9.88 | 0.0015 | 0.0001 | 0.392 | pass |
| | plain 40, CN 35 | 8.74 | 0.063 | 0.0020 | 0.176 | pass |
| | plain 40, CN 30 | 7.76 | 0.062 | 0.0020 | 0.209 | pass |
| | plain 40, CN 25 | 7.02 | 0.061 | 0.0020 | 0.393 | fail by 0.0015 meV/A |
| | plain 40, CN 20 | 6.53 | 0.060 | 0.0020 | 0.567 | fail (force) |
| | plain 30 + tail | 7.48 | 0.0002 | 0.0001 | 0.948 | fail (force) |
| | plain 20 + tail | 6.09 | 0.0074 | 0.0004 | 3.22 | fail (force) |
| | shell 25-40 every 2 steps | 11.53 | | | stale +0.635 | fail (force) |

The JSON holds every row: plain sums at 20, 25, 30 and 40 bohr, the tail correction at each, the CN cutoff at 20
to 40 bohr, and the shell at 20 and 25 bohr every 2, 5, 10 and 20 steps.

- **Tail correction** (`plain R + tail`): for a uniform density beyond R, with each atom's C6 from its own
  coordination number. It brings energy and pressure far below the 40 bohr error, even at 20 bohr. It cannot
  correct forces, and the force error of a shorter real-space sum is above the 40 bohr sum's on every system.
  So it fails the contract at every cutoff below 40 bohr. Added to the 40 bohr sum it costs nothing measurable and
  cuts the energy error by a factor of 40 and the pressure error by 30. That makes it more accurate, not faster.
- **Long-range shell every k steps** (short part to 20 or 25 bohr every step, the shell to 40 bohr every k):
  the stale-force error was estimated by moving the atoms as far as they travel ballistically in k steps at
  300 K and comparing with a fresh 40 bohr sum. It exceeds the 40 bohr sum's own force error on every system at
  every k, even k = 2 (0.035 meV/A on NbCl5, 0.028 on SrTiO3, 0.64 on water). This is an estimate, not an MD
  run. It also saves little: a fresh shell repeats the coordination-number passes, so k = 2 is slower than the
  plain sum.
- **Coordination-number cutoff** (plain 40 bohr pairs, CN cutoff 20 to 35 bohr): it changes the energy, pressure
  and forces by a few 1e-6 to 1e-4 in these units on NbCl5 and SrTiO3, where coordination tails are negligible. It
  saves 0.5 to 2.8 ms because the CN force pass shrinks with the cube of its cutoff. On water the force error
  moves non-monotonically (0.18 at 35 bohr, 0.21 at 30, 0.39 at 25): part of the 40 bohr pair-truncation error
  is cancelled by the CN truncation, which says nothing about a different cell. CN 30 passes on water and
  SrTiO3. On NbCl5 its pressure error exceeds the baseline's by 1.7e-6 GPa, 0.03% of that error but 20 times
  float32 rounding (the `check` stress difference is 1e-7 GPa), so taken literally it fails.
- **Split sum** (damped short range in real space, the smooth remainder on a coarse mesh): **not prototyped in
  this round.** It is the only candidate that corrects forces beyond the real-space cutoff, so it is the only one
  that could beat the 40 bohr sum on all three measures at a shorter cutoff. It shares its mesh machinery with
  S6-D06 (FourierD3) and is left to S6-D05/S6-D06.

## Result

**The default stays the plain 40 bohr real-space sum** with a 40 bohr coordination-number cutoff. This is the
contract's fallback: no candidate that was built and is faster than it passes on all three systems.

- Measured cost of the default with the v1 kernel: 2.48 ms (NbCl5), 8.42 ms (SrTiO3), 9.88 ms (water). Against
  the LoupMD OMAT-0 medium step on record (about 1.85 ms and 8.3 ms, §3.5 of the design note), that is 1.3 and
  1.0 times the model on NbCl5 and SrTiO3. S6-D04 starts from this kernel.
- **Owner question** (recorded as D73): should "no larger than" in the contract allow a relative tolerance, such as
  1% of the baseline error? With one, `cn_cutoff_bohr = 30` would pass on these three systems and save 19 to 21%
  (2.01, 6.68, 7.76 ms). It would still have to pass on the full validation set, which includes graphite, the case
  most sensitive to the CN cutoff (§3.4), before it could become the default. Without a tolerance the default stays
  as above.
- **Recommendation for S6-D05**: build the tail correction as a selectable `long_range="tail"` added on top of the
  40 bohr sum, and a split sum, and drop "long-range part every n steps": its stale forces fail at n = 2.

## Not measured

- The full validation set of §6.2 (the six cells, a metal, a layered compound other than graphite); only the three
  speed systems were run.
- MD energy drift of any candidate.
- The 8.9 workstation GPUs and the A100.
- Repeat spread: the table is one run. In two full runs of the bake-off, the plain 40 bohr times agreed to 1%:
  2.48 against 2.48 ms (NbCl5), 8.44 against 8.42 ms (SrTiO3), 9.82 against 9.88 ms (water).
