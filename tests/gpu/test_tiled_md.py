"""Tiled MD (plan task S5-T06): the tiled call keeps its plan, tile lists and state between calls until an atom has
moved skin/2, and a constant-energy run on the tiled call conserves energy. Every test creates engines and skips
when its model file is absent."""
from __future__ import annotations

import gc

import numpy as np
import pytest
from gpu_support import find_model, maxwell_velocities, srtio3

from loupmd import LoupEngine, capacity, ensure_loup_bundle
from loupmd.config import CapacityConfig

SKIN = 0.5


def _bundle():
    return ensure_loup_bundle(find_model("mace-omat-0-medium.model"), [8, 22, 38])


def _whole(bundle, at, positions):
    eng = LoupEngine(bundle)
    try:
        eng.set_system(at.get_atomic_numbers(), at.get_masses(), skin=SKIN)
        return eng.evaluate(positions, at.cell.array)
    finally:
        del eng
        gc.collect()


@pytest.mark.full
def test_kept_lists_give_the_whole_system_result_until_an_atom_moves_half_the_skin(monkeypatch):
    monkeypatch.setenv("LOUP_NO_HYBRID", "1")     # the plan's bounds against the whole-system call (D60)
    at = srtio3(8)
    rng = np.random.default_rng(8)
    at.positions += rng.normal(scale=0.03, size=at.positions.shape)
    bundle = _bundle()
    n, z, cell = len(at), at.numbers, at.cell.array
    moved = at.positions + rng.uniform(-0.06, 0.06, size=at.positions.shape)        # under skin/2 for every atom
    jumped = moved.copy()
    jumped[17] += [0.2, 0.2, 0.0]                                                     # 0.28 A: over skin/2
    eng = LoupEngine(bundle)
    try:
        eng.evaluate_tiled(z, at.positions, cell, 1000, skin=SKIN)
        first = eng.last_execution()
        e1, f1, s1, tiles = eng.evaluate_tiled(z, moved, cell, 1000, skin=SKIN)
        kept = eng.last_execution()
        e2, f2, s2, _ = eng.evaluate_tiled(z, jumped, cell, 0, skin=SKIN)              # 0: the kept cap
        rebuilt = eng.last_execution()
    finally:
        del eng
        gc.collect()
    assert tiles >= 3 and not first.lists_kept
    assert kept.lists_kept and kept.list_builds == first.list_builds
    assert not rebuilt.lists_kept and rebuilt.list_builds == first.list_builds + 1 and rebuilt.receivers == 1000
    for (e, f, s), pos in (((e1, f1, s1), moved), ((e2, f2, s2), jumped)):
        e0, f0, s0 = _whole(bundle, at, pos)
        assert abs(e - e0) / n <= 1e-7
        assert np.abs(f - f0).max() <= 1e-5
        assert np.abs(s - s0).max() <= 1e-8


@pytest.mark.full
def test_a_tiled_constant_energy_run_conserves_energy():
    at = srtio3(10)
    at.set_velocities(maxwell_velocities(at, 300.0, seed=4))
    eng = LoupEngine(_bundle())
    try:
        res = capacity.run_nve(eng, at, n_steps=400, dt_fs=1.0,
                               config=CapacityConfig(mode="tiled", receivers=2000), skin=SKIN, every=10)
    finally:
        del eng
        gc.collect()
    drift = res.drift_mev_per_atom_ps(len(at))
    print(f"tiled NVE: {len(at)} atoms, {res.tiles} tiles of <= {res.receivers}, {res.list_builds} list builds in "
          f"400 steps, {res.seconds_per_step * 1e3:.1f} ms per step, drift {drift:.4f} meV/atom/ps, total energy "
          f"spread {np.ptp(res.total) / len(at) * 1e3:.4f} meV/atom")
    assert res.mode == "tiled" and res.tiles >= 3
    assert res.list_builds >= 2               # 300 K moves some atom skin/2 within 0.4 ps
    assert abs(drift) < 0.05


# ------------------------------------------------------------------------- the front end (plan task S5-T12)

@pytest.mark.full
def test_a_tiled_simulation_follows_the_whole_systems_nve_trajectory(monkeypatch):
    """lm.Simulation with capacity 'tiled' (TiledEngine, host velocity Verlet) against the device loop on the whole
    system, without hybrids: 20 steps of 1 fs from 300 K stay together to float32 force rounding."""
    import loupmd
    from loupmd import integrators

    monkeypatch.setenv("LOUP_NO_HYBRID", "1")
    at = srtio3(10)
    at.set_velocities(maxwell_velocities(at, 300.0, seed=6))
    ff = loupmd.load(find_model("mace-omat-0-medium.model"), trusted=True)
    nve = integrators.NVE(timestep_fs=1.0, fixcm=False)
    sims = {}
    for name, cap in (("whole", "whole"), ("tiled", CapacityConfig(mode="tiled", receivers=2000))):
        sim = loupmd.Simulation(at, ff, integrator=nve, seed=1, skin=SKIN, capacity=cap)
        sim.run(steps=20)
        sims[name] = (sim.execution["mode"], sim.atoms)
        sim.close()
        del sim
        gc.collect()
    (mode_w, whole), (mode_t, tiled) = sims["whole"], sims["tiled"]
    dx = np.abs(tiled.positions - whole.positions).max()
    de = abs(tiled.get_potential_energy() - whole.get_potential_energy()) / len(at)
    print(f"tiled Simulation vs whole, 20 NVE steps: dx {dx:.2e} A, dE {de:.2e} eV/atom")
    assert (mode_w, mode_t) == ("whole", "tiled")
    assert dx <= 1e-5 and de <= 1e-6


@pytest.mark.full
def test_tiled_langevin_heats_toward_its_target():
    import loupmd
    from loupmd import integrators

    at = srtio3(10)                                   # 5,000 atoms at rest, 3 tiles
    ff = loupmd.load(find_model("mace-omat-0-medium.model"), trusted=True)
    lang = integrators.Langevin(timestep_fs=2.0, temperature_K=300.0, friction_per_fs=0.05)
    sim = loupmd.Simulation(at, ff, integrator=lang, seed=3, skin=SKIN,
                            capacity=CapacityConfig(mode="tiled", receivers=2000))
    try:
        summary = sim.run(steps=200)
        tiles = sim._engine.tiles
    finally:
        sim.close()
        del sim
        gc.collect()
    print(f"tiled Langevin: {tiles} tiles, {summary.ms_per_step:.1f} ms per step, final {summary.final_temperature_K:.1f} K")
    assert tiles >= 3 and 200.0 <= summary.final_temperature_K <= 400.0


@pytest.mark.full
def test_loupmd_md_capacity_auto_tiles_a_cell_that_does_not_fit_and_says_so(tmp_path, monkeypatch, capsys):
    """S5-T12's acceptance: `loupmd md --capacity auto` on 320,000 atoms (32 GiB whole on the plan's count) runs in
    tiles on a 16 GB card and the report says so."""
    from ase.io import write

    from loupmd.cli import main

    big = srtio3(40)
    write(tmp_path / "big.extxyz", big)
    monkeypatch.chdir(tmp_path)
    code = main(["md", "big.extxyz", "--model", find_model("mace-omat-0-medium.model"), "--trusted",
                 "--temperature", "300", "--steps", "10", "--timestep", "1", "--seed", "1", "--initial-velocities",
                 "maxwell_boltzmann", "--skin", str(SKIN), "--capacity", "auto", "--thermo-every", "5",
                 "--trajectory-every", "0", "--output", "run"])
    out = capsys.readouterr().out
    print(out)
    import json

    plan = json.loads((tmp_path / "run" / "metadata.json").read_text())["plan"]
    if plan["execution"] == "whole":
        pytest.skip("this GPU runs 320,000 atoms whole")
    assert code == 0
    assert plan["execution"] == "tiled" and 4096 <= plan["receivers"] <= 65536
    assert "tiles of at most" in out and "the integrator runs on the host" in out
