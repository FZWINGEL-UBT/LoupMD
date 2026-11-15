"""``lm.optimize`` on the GPU (plan task S4-T10): the step counts of ASE's own optimisers driven by ``lm.calc``,
and what the optimiser's arithmetic costs beside the force calls at 5,000 atoms."""
from __future__ import annotations

import numpy as np
import pytest
from ase.optimize import FIRE as AseFIRE
from ase.optimize import LBFGS as AseLBFGS
from conftest import srtio3

import loupmd
from loupmd import optimize


def _rattled(rep, seed=9, sigma=0.04):
    at = srtio3(rep)
    at.positions += np.random.default_rng(seed).normal(scale=sigma, size=at.positions.shape)
    return at


@pytest.mark.parametrize("record,ase_class", [(optimize.LBFGS(), AseLBFGS), (optimize.FIRE(), AseFIRE)],
                         ids=["lbfgs", "fire"])
def test_the_step_count_is_ase_s_within_two(model_path, record, ase_class):
    atoms = _rattled(3)                                              # 135 atoms
    force_field = loupmd.load(model_path, trusted=True)
    result = optimize.relax(force_field, atoms, record, fmax=0.05)
    reference = atoms.copy()
    reference.calc = loupmd.calc(model_path, trusted=True)
    ase_optimizer = ase_class(reference, logfile=None)
    ase_optimizer.run(fmax=0.05)
    ase_steps = ase_optimizer.get_number_of_steps()
    print(f"{type(record).__name__}: {result.steps} steps, ASE {ase_steps}; largest force {result.max_force:.4f} eV/A; "
          f"positions differ from ASE's by {np.abs(result.atoms.positions - reference.positions).max():.2e} A")
    assert result.converged and abs(result.steps - ase_steps) <= 2
    assert np.abs(result.atoms.positions - reference.positions).max() < 5e-3
    assert result.energy == pytest.approx(reference.get_potential_energy(), abs=1e-4 * len(atoms))


def test_the_optimiser_s_own_time_at_5000_atoms(model_path):
    atoms = _rattled(10, sigma=0.05)
    force_field = loupmd.load(model_path, trusted=True)
    optimize.relax(force_field, atoms, optimize.FIRE(), fmax=1e-3, steps=3)        # export, engine, tuning
    shares = {}
    for name, record in (("FIRE", optimize.FIRE()), ("LBFGS", optimize.LBFGS())):
        result = optimize.relax(force_field, atoms, record, fmax=1e-3, steps=40)
        shares[name] = result.optimizer_s / result.force_s
        print(f"{name}: {result.steps} steps, {result.force_s / result.force_calls * 1e3:.2f} ms per force call, "
              f"optimiser {result.optimizer_s / result.steps * 1e3:.3f} ms per step: {100 * shares[name]:.2f}% of "
              "the force calls")
    assert shares["FIRE"] < 0.02            # the plan's criterion; LBFGS with its 100-pair history is recorded
    assert shares["LBFGS"] < 0.25


def test_the_relaxed_lattice_constant_is_ase_s(model_path):
    """S4-T11: SrTiO3 3x3x3 strained by 2% and rattled; positions and cell relaxed together."""
    from ase.filters import FrechetCellFilter

    atoms = _rattled(3, sigma=0.02)
    atoms.set_cell(atoms.cell.array * 1.02, scale_atoms=True)
    result = optimize.relax(loupmd.load(model_path, trusted=True), atoms,
                            optimize.LBFGS(cell_filter=optimize.FrechetCell()), fmax=0.002, steps=400)
    reference = atoms.copy()
    reference.calc = loupmd.calc(model_path, trusted=True)
    ase_optimizer = AseLBFGS(FrechetCellFilter(reference), logfile=None)
    ase_optimizer.run(fmax=0.002, steps=400)
    mine = result.atoms.cell.cellpar()[:3] / 3.0
    theirs = reference.cell.cellpar()[:3] / 3.0
    print(f"cell relaxation: {result.steps} steps, ASE {ase_optimizer.get_number_of_steps()}; lattice constants "
          f"{np.round(mine, 5)} A, ASE {np.round(theirs, 5)} A; largest difference {np.abs(mine - theirs).max():.2e} A; "
          f"stress left {np.abs(result.atoms.get_stress()).max():.2e} eV/A^3")
    assert result.converged and np.abs(mine - theirs).max() < 1e-4
    assert 3.85 < mine.mean() < 3.99 and abs(mine.mean() - 3.905 * 1.02) > 0.01     # it moved off the strained cell
