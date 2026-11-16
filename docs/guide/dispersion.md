# Dispersion (D3)

MACE foundation models trained on PBE or r2SCAN data (MACE-OMAT-0, MP-0, MPA-0, MatPES, MACE-MH-1's `omat_pbe` and
`matpes_r2scan` heads) describe no dispersion beyond their cutoff and little inside it. `dispersion=` adds Grimme's
D3 two-body term to every step: energy, forces, stress, per-atom energies and the energy-only calls of Monte Carlo
moves. It runs on the GPU inside the captured MD step, after the model.

```python
import loupmd as lm

result = lm.md("structure.cif", model="omat0-medium", temperature_K=300, steps=10_000, dispersion="d3bj")
```

```bash
loupmd md structure.cif --model omat0-medium --temperature 300 --steps 10000 --dispersion d3bj
```

In a run file: `dispersion = "d3bj"`, or a table with the fields of `lm.Dispersion`:

```toml
dispersion = {damping = "zero", pair_cutoff_bohr = 50.0}
```

## Choosing a variant

`loupmd dispersion list` prints the presets, the dampings, the summation methods with their known limits and the
parameter sets with their sources.

| preset | what it is |
|---|---|
| `d3bj` | Becke-Johnson damping, the model's functional, pairs and coordination numbers to 40 bohr, no three-body term. The default choice. |
| `d3bj-mace` | the same numbers as `mace_mp(dispersion=True)` (torch-dftd, 40 bohr) and FastMACE's d3bj, with the plain sum fixed |
| `d3zero`, `d3bjm`, `d3zerom` | zero damping and the two modified dampings of Smith et al. 2016, with d3bj's cutoffs |
| `d3bj-atm` | `d3bj` plus the three-body (Axilrod-Teller-Muto) term over triangles whose sides are within 20 bohr |
| `d3-reference` | simple-dftd3's Python defaults: pairs to 60 bohr, coordination numbers and the three-body term to 40 bohr |

The parameters follow the model's training functional: PBE for OMAT-0, MP-0, MPA-0 and MH-1's `omat_pbe` head,
r2SCAN for MatPES-r2SCAN and MH-1's `matpes_r2scan` head (Becke-Johnson damping only: r2SCAN has no zero-damping
set). `lm.Dispersion(functional="pbe")` chooses another functional's set and warns; `lm.Dispersion(parameters={...})`
takes raw parameters, and the run's record then names them as given by the user.

LoupMD refuses dispersion for models whose training data contain it already, because D3 on top counts it twice:
MACE-OFF (wB97M-D3(BJ) data), MACE-OMOL-0 and MH-1's `omol` and `spice_wB97M` heads (wB97M-V, whose VV10
correlation describes dispersion), and MACE-POLAR-1.

## Accuracy

The pair arithmetic is float32 on relative vectors with float64 sums. Against simple-dftd3 at the same settings, on
NbCl5, paracetamol and SrTiO3 and for each damping, energies agree to 1e-6 eV/atom, forces to 1e-5 eV/A and stress
to 1e-7 eV/A^3 (`tests/gpu/test_d3.py`; the measured differences are far below these bounds,
`docs/validation/dispersion-bakeoff.md`). The plain sum at 40 bohr differs from the converged sum (150 bohr) by
what the bake-off lists per system; no faster method met that accuracy contract (PLAN.md D73).

`smoothing=w` (bohr) takes each pair energy to zero over the last `w` bohr before the pair cutoff with a quintic
switch, so energy and forces are continuous at the cutoff. The default is no switch, as in simple-dftd3 and
torch-dftd.

## Checking the term on your structure

`loupmd validate --dispersion PRESET` evaluates LoupMD's term on your structures and on simple-dftd3 (the `dftd3`
Python package, LGPL, installed separately) at the same settings, and passes when they agree within 1e-6 eV/atom,
1e-5 eV/A and 1e-7 eV/A^3. The model is not run; `--model` only names the training functional (or give
`--functional`). Without `--structure` it uses the oxide validation set.

```bash
loupmd validate --dispersion d3bj-atm --model omat0-medium --structure structure.cif --json check.json
```

The switch (`smoothing=`) has no counterpart in simple-dftd3 and is refused by the check.

## The three-body term

`three_body=True` (or the presets `d3bj-atm` and `d3-reference`) adds the Axilrod-Teller-Muto term with s9 = 1,
computed as simple-dftd3 does for every damping: C9 from the three pair C6 values, a zero damping of the triangle with
4/3 R0 and the exponent (alpha + 2) / 3, every side within `three_body_cutoff_bohr`. It is repulsive in crystals, 3 to
9% of the pair term (docs/design/dispersion.md section 3.2). Against simple-dftd3 at the same cutoffs it agrees to the
pair term's tolerances (`tests/gpu/test_d3.py`). With `smoothing=w` each side of a triangle is switched off over the
last `w` bohr before the three-body cutoff too; without it the cutoff is hard, as in simple-dftd3. Its cost grows with
the square of the neighbours within its cutoff: at 40 bohr it is many times the pair term.

## The split sum

`Dispersion(long_range="split", pair_cutoff_bohr=25.0)` sums the pair term without truncating it. The damped
pair energy minus its smooth long-range part runs in real space to `pair_cutoff_bohr`; that smooth part for every
pair and periodic image goes to reciprocal space on a mesh (smooth particle-mesh Ewald). This works because D3's C6
and C8 coefficients separate exactly into a few "charges" per atom (at most 5 per element), so no combination rule
is assumed. The coordination numbers are still summed to `cn_cutoff_bohr`.

- **Accuracy.** Against the infinite sum it is 10 to 300 times closer in energy than the plain 40 bohr sum (0.0008
  meV/atom on NbCl5 at 25 bohr, against 0.59), and closer in forces and stress at 25 bohr. At 20 bohr the forces can
  be slightly worse than the plain sum's, though still within 1e-5 eV/A.
- **Cost.** It is not faster: 1.05 to 1.7 times the plain sum on the RTX 5060 Ti, from 2,000 to 1,000,000 atoms.
  Both walk 40 bohr for the coordination numbers, and the mesh adds its own cost.
- **Mesh.** B-splines of order 8 at `mesh_spacing_A` (default 0.8 A, within 1e-7 eV/A of an exact reciprocal sum).
- **Needs** a 3D periodic cell and cuFFT (`pip install 'cuda-toolkit[cufft]'`; LoupMD's wheel depends on it).
- **Where it fits.** Large cells where the plain sum's truncation matters, for example the pressure or the energy
  per atom of a big cell (1.1 meV/atom on SrTiO3).

## The automatic choice

`long_range="auto"` (the preset `d3bj`) chooses the method at setup, on your starting structure:

1. The reference is the split sum at tight settings: a 30 bohr real-space cutoff and a 0.6 A mesh.
2. The candidates are the plain sum at the record's cutoffs and the split sum at 20 and 25 bohr with the default
   mesh. `auto_methods` restricts them. Without cuFFT the split candidates are left out.
3. Each candidate is evaluated once. It is kept when its energy, largest force and stress deviations from the
   reference are each no larger than the plain sum's.
4. The kept candidates are timed, and the fastest is used.

The report and `metadata.json` (`dispersion.long_range_choice`) list every candidate with its deviations, its time,
and why it was dropped. A change of composition by an event repeats the check, and `Simulation.recheck_dispersion()`
repeats it on request. A resume checks again, so it may choose differently when two candidates are close in time.

The check costs one tight evaluation, one evaluation per candidate and six timed replays per kept candidate. On the
RTX 5060 Ti that was 0.35 s at 5,000 atoms, 5.4 s at 98,000 and 45 s at 976,000 (SrTiO3).

The plain sum is its own bound, so it is always kept. The split sum is chosen only where it is also faster: on the
benchmark cells the plain sum was faster, while on 5,184 atoms of water the split sum at 20 bohr was faster (9.5 ms
against 10.2 ms). To run the more accurate split sum regardless of time, set `long_range="split"`.

## The ASE calculator

`lm.calc("omat0-medium", dispersion="d3bj")` adds the term to every calculation (energy, forces, stress). It has no
setup structure, so `long_range="auto"` runs the plain sum there; name `long_range="split"` to use the split sum.

Relaxations take the term the same way: `lm.optimize.relax(ff, atoms, dispersion="d3bj")`, `loupmd relax
--dispersion d3bj`, and `dispersion=` on the ASE optimisers of `loupmd.ase`. `Simulation.relax` and `loupmd quench`
use the run's term ([relaxation](relax.md)).

## Cost

The term walks every pair within 40 bohr (21 A), which is far more pairs than the model's 6 A cutoff: on small cells
it can cost as much as the model. The report's plan lines name the settings; `docs/validation/dispersion-bakeoff.md`
lists the time per evaluation on the RTX 5060 Ti.

## Not in this version

- The tail correction and the mesh sum (S6-D05, S6-D06), so `long_range="auto"` chooses between the plain and
  the split sum.
- Dispersion with open domains. (A cell that does not fit the GPU whole runs the model tiled and the D3 term for
  the whole system beside it: the term needs no neighbour list and about 200 bytes per atom.)
- D4. Its own ASE calculator (`dftd4.ase.DFTD4`) adds it to `lm.calc` through ASE's `SumCalculator`:

```python
from ase.calculators.mixing import SumCalculator
from dftd4.ase import DFTD4

atoms.calc = SumCalculator([lm.calc("omat0-medium"), DFTD4(method="PBE")])
```
