"""The ASE drop-in classes of ``loupmd.ase`` (plan task S4-T13, the part the engine has): ASE's arguments and
units, observers, atoms updated in place. Fake engine and EMT; needs neither a GPU nor the library."""
from __future__ import annotations

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import LBFGS as AseLBFGS
from test_simulation import SpringEngine

import loupmd
from loupmd import _core, optimize
from loupmd.ase import md as ase_md
from loupmd.ase import optimize as ase_optimize
from loupmd.errors import InputError, PlanRefused


@pytest.fixture
def ff(tmp_path):
    checkpoint = tmp_path / "some.model"
    checkpoint.write_bytes(b"never unpickled here")
    SpringEngine.made, SpringEngine.poison_at = [], None
    return loupmd.load(str(checkpoint), trusted=True, engine_factory=SpringEngine,
                       ensure_bundle=lambda path, numbers, **kw: path.with_name("bundle.loup"), probe=lambda p: None)


def _moving():
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_velocities(np.random.default_rng(3).normal(0.0, 0.02, atoms.positions.shape))
    return atoms


def test_langevin_takes_ase_s_units_and_updates_the_atoms_in_place(ff):
    atoms = _moving()
    start = atoms.positions.copy()
    dyn = ase_md.Langevin(atoms, 2.0 * units.fs, temperature_K=350, friction=0.02 / units.fs, fixcm=False, model=ff,
                          seed=4)
    seen = []
    dyn.attach(lambda: seen.append((dyn.get_number_of_steps(), atoms.positions.copy(), atoms.get_potential_energy())),
               interval=10)
    assert dyn.run(40) is True and dyn.get_number_of_steps() == 40
    engine, = SpringEngine.made
    assert [e for e in engine.log if isinstance(e, tuple) and e[0] == "run"] == [("run", 10, _core.LOUP_LANGEVIN, 0)] * 4
    config = dyn.simulation.config
    assert (config.timestep_fs, config.temperature_K, config.seed) == (pytest.approx(2.0), 350.0, 4)
    assert config.friction_per_fs == pytest.approx(0.02)
    assert [step for step, _, _ in seen] == [0, 10, 20, 30, 40] and np.array_equal(seen[0][1], start)
    assert np.array_equal(atoms.positions, engine.pos) and np.array_equal(seen[-1][1], engine.pos)
    assert np.allclose(atoms.get_velocities(), engine.vel) and atoms.get_potential_energy() == engine._energy()
    assert np.array_equal(atoms.get_forces(), engine._forces())
    dyn.run(5)                                             # goes on from where it is; the next observer call at 50
    assert dyn.get_number_of_steps() == 45 and len(seen) == 5


def test_velocity_verlet_and_observers_of_different_intervals(ff):
    atoms = _moving()
    dyn = ase_md.VelocityVerlet(atoms, 1.0 * units.fs, model=ff)
    calls = {"a": [], "b": []}
    dyn.attach(lambda: calls["a"].append(dyn.nsteps), interval=4)
    dyn.attach(lambda: calls["b"].append(dyn.nsteps), interval=6)
    dyn.run(12)
    assert calls == {"a": [0, 4, 8, 12], "b": [0, 6, 12]}
    assert [e[1] for e in SpringEngine.made[0].log if isinstance(e, tuple) and e[0] == "run"] == [2] * 6
    assert dyn.simulation.config.scheme == "nve"


def test_what_is_refused(ff):
    atoms = _moving()
    assert ase_md.Langevin(atoms, units.fs, temperature_K=300, friction=0.01 / units.fs,
                           model=ff).simulation.config.fixcm is True                       # ASE's default
    with pytest.raises(InputError, match="needs model="):
        ase_md.VelocityVerlet(atoms, units.fs)
    with pytest.raises(InputError, match="needs temperature_K and friction"):
        ase_md.Langevin(atoms, units.fs, fixcm=False, model=ff)
    with pytest.raises(TypeError, match="unexpected keyword arguments trajectory"):
        ase_md.VelocityVerlet(atoms, units.fs, model=ff, trajectory="md.traj")
    with pytest.raises(InputError, match="already loaded"):
        ase_md.VelocityVerlet(atoms, units.fs, model=ff, head="x")
    for name in ("LangevinBAOAB", "Bussi", "NoseHooverChainNVT", "MTKNPT", "NVTBerendsen", "NPTBerendsen"):
        with pytest.raises(PlanRefused, match=f"loupmd.ase.md.{name} is not in this version"):
            getattr(ase_md, name)
    with pytest.raises(AttributeError):
        ase_md.Nonsense
    assert SpringEngine.made == []


class EMTField:
    def evaluate(self, atoms, stress=False, skin=0.5):
        at = atoms.copy()
        at.set_constraint()
        at.calc = EMT()
        return at.get_potential_energy(), at.get_forces(), at.get_stress(voigt=False) if stress else None


def _rattled_copper():
    at = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    at.positions += np.random.default_rng(3).normal(0.0, 0.08, at.positions.shape)
    return at


@pytest.mark.parametrize("cls", [ase_optimize.LBFGS, ase_optimize.FIRE])
def test_the_optimisers_relax_in_place_and_count_ase_s_steps(cls, monkeypatch):
    from ase.optimize import FIRE as AseFIRE

    monkeypatch.setattr(ase_optimize, "force_field_of", lambda model, head, options, what: EMTField())
    atoms = _rattled_copper()
    reference = atoms.copy()
    reference.calc = EMT()
    ase_optimizer = (AseLBFGS if cls is ase_optimize.LBFGS else AseFIRE)(reference, logfile=None)
    ase_optimizer.run(fmax=0.02)
    opt = cls(atoms, model="anything")
    steps = []
    opt.attach(lambda: steps.append(opt.get_number_of_steps()))
    assert opt.run(fmax=0.02) is True
    assert opt.get_number_of_steps() == ase_optimizer.get_number_of_steps() == steps[-1] and steps[0] == 0
    assert np.abs(atoms.positions - reference.positions).max() < 1e-9          # the caller's atoms, relaxed
    assert atoms.get_potential_energy() == pytest.approx(reference.get_potential_energy(), abs=1e-9)
    assert opt.run(fmax=0.02) is True and opt.get_number_of_steps() == 0       # already there


def test_a_cell_filter_and_refusals_of_the_optimisers(monkeypatch):
    from ase.filters import FrechetCellFilter

    monkeypatch.setattr(ase_optimize, "force_field_of", lambda model, head, options, what: EMTField())
    atoms = _rattled_copper()
    atoms.set_cell(atoms.cell.array * 1.03, scale_atoms=True)
    reference = atoms.copy()
    reference.calc = EMT()
    AseLBFGS(FrechetCellFilter(reference), logfile=None).run(fmax=0.02, steps=50)
    opt = ase_optimize.LBFGS(atoms, cell_filter=optimize.FrechetCell(), model="anything")
    opt.run(fmax=0.02, steps=50)
    assert np.abs(atoms.cell.array - reference.cell.array).max() < 1e-8
    assert np.abs(atoms.positions - reference.positions).max() < 1e-8
    with pytest.raises(InputError, match="takes an ase.Atoms"):
        ase_optimize.LBFGS(FrechetCellFilter(atoms), model="anything")
    with pytest.raises(InputError, match="use_line_search=True is not offered"):
        ase_optimize.LBFGS(atoms, use_line_search=True, model="anything")
