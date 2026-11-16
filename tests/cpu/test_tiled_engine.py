"""The tiled mode in ``lm.Simulation`` and ``lm.md`` (plan task S5-T12): ``capacity="auto"`` plans the execution and
runs a cell that does not fit in tiles through ``TiledEngine``, whose integrator runs on the host. On a fake engine
whose tiled call is the spring potential of ``fake_core``, so the tiled run can be compared with the whole one.
Needs neither a GPU nor the library."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from ase.build import bulk
from ase.constraints import FixAtoms
from fake_core import K, SpringEngine

import loupmd
from loupmd import _core, integrators, output, provenance, quick
from loupmd.errors import BackendError, PlanRefused
from loupmd.tiled_engine import TiledEngine


class TileSpringEngine(SpringEngine):
    """SpringEngine with the capacity calls: a system of more than ``tile_from`` atoms does not fit whole."""

    tile_from = 100
    shared_sites = {}

    def set_execution(self, mode, receivers=0, memory_fraction=0.9):
        self.execution = (mode, receivers, memory_fraction)

    def plan_execution(self, numbers, positions, cell, skin=0.5):
        n = len(numbers)
        tiled = n > self.tile_from or self.execution[0] == _core.LOUP_EXEC_TILED
        self.log.append("plan")
        return _core.MemoryPlan(1000 * n, 100 * n, 10 ** 6, 2 ** 34, 64 if tiled else 0,
                                _core.LOUP_EXEC_TILED if tiled else _core.LOUP_EXEC_WHOLE)

    def evaluate_tiled(self, numbers, positions, cell, receivers, skin=0.5):
        pos = np.asarray(positions, dtype=float)
        # the springs' sites: the first positions seen for this atom count, by any engine (a resumed run too)
        sites = TileSpringEngine.shared_sites.setdefault(len(numbers), pos.copy())
        self.tiled_calls = getattr(self, "tiled_calls", 0) + 1
        self.receivers_seen = receivers
        d = pos - sites
        cap = receivers or 64
        self.tiles = math.ceil(len(numbers) / cap)
        return 0.5 * K * float((d ** 2).sum()) - 3.0 * len(numbers), -K * d, np.zeros((3, 3)), self.tiles

    def last_execution(self):
        return _core.Execution(_core.LOUP_EXEC_TILED, self.receivers_seen or 64, self.tiles, 2 ** 34, 10 ** 6,
                               True, 1)


@pytest.fixture
def ff(tmp_path):
    checkpoint = tmp_path / "some.model"
    checkpoint.write_bytes(b"never unpickled here")
    SpringEngine.made, SpringEngine.poison_at = [], None
    TileSpringEngine.shared_sites = {}
    return loupmd.load(str(checkpoint), trusted=True, engine_factory=TileSpringEngine,
                       ensure_bundle=lambda path, numbers, **kw: path.with_name("bundle.loup"), probe=lambda p: None)


def _big():
    at = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 2))    # 144 atoms: tiled in the fake
    at.positions += np.random.default_rng(1).normal(scale=0.05, size=at.positions.shape)
    return at


def test_auto_runs_a_cell_that_does_not_fit_in_tiles_with_the_whole_runs_nve_trajectory(ff, tmp_path):
    nve = integrators.NVE(timestep_fs=1.0, fixcm=False)
    runs = {}
    for capacity in ("auto", "whole"):
        runs[capacity] = loupmd.md(_big(), model=ff, integrator=nve, steps=60, output=str(tmp_path / capacity),
                                   seed=2, initial_velocities=300.0, capacity=capacity)
    tiled, whole = runs["auto"].simulation, runs["whole"].simulation
    assert isinstance(tiled._engine, TiledEngine) and tiled.execution["mode"] == "tiled"
    assert not isinstance(whole._engine, TiledEngine) and whole.execution["mode"] == "whole"
    assert "plan" in tiled._engine._eng.log and "plan" not in whole._engine.log
    rows_t = output.read_thermo(tmp_path / "auto")[1]
    rows_w = output.read_thermo(tmp_path / "whole")[1]
    assert [int(r[0]) for r in rows_t] == [int(r[0]) for r in rows_w] and len(rows_t) >= 3
    assert [r[4:8] for r in rows_t] == [r[4:8] for r in rows_w]          # temperature and energies, as written
    assert np.allclose(tiled.atoms.positions, whole.atoms.positions, rtol=0, atol=1e-12)
    # one tiled call per step, plus the first: the thermo steps' state round trip repeats none
    assert tiled._engine._eng.tiled_calls == 61
    plan = runs["auto"].metadata["plan"]
    assert plan["execution"] == "tiled" and plan["receivers"] == 64 and plan["tiles"] == 3
    assert plan["memory"]["tiled_global_bytes"] == 14400
    assert provenance.validate(runs["auto"].metadata) == []
    summary = runs["auto"].summary
    assert "3 tiles of at most 64 atoms" in summary and "the integrator runs on the host" in summary
    assert "the whole system on" in runs["whole"].summary
    record = json.loads((tmp_path / "auto" / "simulation.json").read_text())
    assert record["capacity"] == {"mode": "tiled", "receivers": 64, "memory_fraction": 0.9}


def test_auto_keeps_a_cell_that_fits_whole(ff, tmp_path):
    at = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    res = loupmd.md(at, model=ff, temperature_K=300, steps=10, output=str(tmp_path / "r"), seed=1,
                    initial_velocities="maxwell_boltzmann")
    assert res.simulation.execution["mode"] == "whole" and res.metadata["plan"]["execution"] == "whole"
    assert not isinstance(res.simulation._engine, TiledEngine)


def test_the_tiled_mode_refuses_the_integrators_it_does_not_run(ff, tmp_path):
    csvr = integrators.CSVR(timestep_fs=1.0, temperature_K=300, tau_fs=100.0)
    at = _big()
    quick.draw_initial_velocities(at, 300.0, 1)
    with pytest.raises(PlanRefused, match="the csvr integrator in tiled mode"):
        loupmd.Simulation(at, ff, integrator=csvr, capacity="tiled")        # before anything is exported
    assert SpringEngine.made == []
    sim = loupmd.Simulation(at, ff, integrator=csvr)                         # auto: the plan says tiled
    with pytest.raises(PlanRefused, match="PLAN.md D65"):
        sim.run(steps=5)
    with pytest.raises(PlanRefused, match="'wavefront' is not in this version"):
        loupmd.Simulation(_big(), ff, integrator=integrators.NVE(timestep_fs=1.0), capacity="wavefront")


def test_langevin_in_tiles_holds_its_target_and_its_fixed_atoms(ff):
    at = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((5, 5, 5))     # 1,000 atoms
    fixed = np.zeros(len(at), dtype=bool)
    fixed[:50] = True
    at.set_constraint(FixAtoms(mask=fixed))
    start = at.positions[fixed].copy()
    lang = integrators.Langevin(timestep_fs=2.0, temperature_K=300.0, friction_per_fs=0.02)
    sim = loupmd.Simulation(at, ff, integrator=lang, seed=5, output=None)
    sim.run(steps=400)
    temps = []
    for _ in range(20):
        temps.append(sim.run(steps=40).final_temperature_K)
    assert isinstance(sim._engine, TiledEngine)
    assert abs(np.mean(temps) / 300.0 - 1.0) < 0.05, np.mean(temps)
    assert np.array_equal(sim.atoms.positions[fixed], start)


def test_a_tiled_run_resumes_where_it_stopped(ff, tmp_path):
    lang = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.02)
    full = loupmd.Simulation(_big(), ff, integrator=lang, seed=9, output=loupmd.Output(str(tmp_path / "full")))
    full.run(steps=40)
    half = loupmd.Simulation(_big(), ff, integrator=lang, seed=9, output=loupmd.Output(str(tmp_path / "half")))
    half.run(steps=20)
    half.save_restart()
    half.close()
    again = loupmd.Simulation.resume(tmp_path / "half", force_field=ff)
    assert isinstance(again._engine, TiledEngine) and again.step == 20
    again.run(steps=20)
    assert np.allclose(again.atoms.positions, full.atoms.positions, rtol=0, atol=1e-12)
    assert np.allclose(again.atoms.get_velocities(), full.atoms.get_velocities(), rtol=0, atol=1e-12)


def test_loupmd_md_capacity_auto_runs_a_cell_that_does_not_fit_in_tiles_and_says_so(ff, tmp_path, monkeypatch,
                                                                                     capsys):
    """S5-T12's acceptance on the fake engine (the GPU test runs it on a cell that does not fit the card)."""
    from ase.io import write

    from loupmd import simulation
    from loupmd.cli import main

    write(tmp_path / "salt.cif", bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    monkeypatch.setattr(quick, "load", lambda *a, **k: ff)
    monkeypatch.setattr(simulation, "load", lambda *a, **k: ff)
    monkeypatch.chdir(tmp_path)
    base = ["md", "salt.cif", "--model", "m", "--temperature", "300", "--steps", "40", "--timestep", "1", "--seed",
            "3", "--initial-velocities", "maxwell_boltzmann", "--repeat", "3", "3", "2"]
    assert main([*base, "--capacity", "auto", "--output", "a"]) == 0
    out = capsys.readouterr().out
    assert "3 tiles of at most 64 atoms" in out and "the integrator runs on the host" in out
    metadata = json.loads((tmp_path / "a" / "metadata.json").read_text())
    assert metadata["plan"]["execution"] == "tiled" and metadata["arguments"]["capacity"] == "auto"
    assert main(["run", "--resume", "a", "--steps", "20"]) == 0
    assert "3 tiles of at most 64 atoms" in capsys.readouterr().out
    assert main([*base, "--capacity", "whole", "--output", "b"]) == 0
    assert "the whole system on" in capsys.readouterr().out


def test_a_state_block_of_the_other_engine_is_refused():
    eng = TiledEngine(TileSpringEngine("bundle"))
    eng.set_system(np.full(4, 11), np.full(4, 23.0))
    with pytest.raises(BackendError, match="not a tiled-mode state block"):
        eng.import_state(b"x" * 400)
