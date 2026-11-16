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

## The ASE calculator

`lm.calc("omat0-medium", dispersion="d3bj")` adds the term to every calculation (energy, forces, stress).

## Cost

The term walks every pair within 40 bohr (21 A), which is far more pairs than the model's 6 A cutoff: on small cells
it can cost as much as the model. The report's plan lines name the settings; `docs/validation/dispersion-bakeoff.md`
lists the time per evaluation on the RTX 5060 Ti.

## Not in this version

- The tail correction, the split sum and the mesh sum (S6-D05, S6-D06): `long_range="auto"` uses the plain sum.
- Dispersion with open domains. (A cell that does not fit the GPU whole runs the model tiled and the D3 term for
  the whole system beside it: the term needs no neighbour list and about 200 bytes per atom.)
- D4. Its own ASE calculator (`dftd4.ase.DFTD4`) adds it to `lm.calc` through ASE's `SumCalculator`:

```python
from ase.calculators.mixing import SumCalculator
from dftd4.ase import DFTD4

atoms.calc = SumCalculator([lm.calc("omat0-medium"), DFTD4(method="PBE")])
```
