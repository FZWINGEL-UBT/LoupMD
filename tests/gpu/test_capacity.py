"""The capacity mode (plan task S5-T05): ``capacity.evaluate`` asks the engine for an execution plan (one probe
tile measures the workspace per row and per edge), runs the whole-system call when it fits with 10% headroom and
the tiled call with the largest receiver cap of 4,096 to 65,536 that fits otherwise. Every test creates engines
and skips when its model file is absent."""
from __future__ import annotations

import gc

import numpy as np
import pytest
from gpu_support import find_model, srtio3

from loupmd import LoupEngine, PlanRefused, capacity, ensure_loup_bundle
from loupmd.config import CapacityConfig

SKIN = 0.5


def _bundle():
    return ensure_loup_bundle(find_model("mace-omat-0-medium.model"), [8, 22, 38])


def _whole(bundle, at):
    eng = LoupEngine(bundle)
    try:
        eng.set_system(at.get_atomic_numbers(), at.get_masses(), skin=SKIN)
        return eng.evaluate(at.positions, at.cell.array)
    finally:
        del eng
        gc.collect()


@pytest.mark.full
def test_auto_runs_a_cell_that_fits_whole_as_the_whole_system_call(monkeypatch):
    monkeypatch.setenv("LOUP_NO_HYBRID", "1")     # two engines' hybrid plan picks differ by up to 1.1e-6 eV/atom
    at = srtio3(8)
    at.positions += np.random.default_rng(3).normal(scale=0.03, size=at.positions.shape)
    bundle = _bundle()
    e0, f0, s0 = _whole(bundle, at)
    eng = LoupEngine(bundle)
    try:
        res = capacity.evaluate(eng, at, skin=SKIN)
    finally:
        del eng
        gc.collect()
    assert res.mode == "whole" and res.tiles == 1
    assert res.plan.whole_bytes > 0 and res.plan.whole_bytes < res.plan.free_bytes
    # Without hybrids two engines' whole-system calls agree to float32 rounding (with them, PLAN.md D61: up to
    # 1.1e-6 eV/atom was seen here).
    assert abs(res.energy - e0) / len(at) <= 1e-7
    assert np.abs(res.forces - f0).max() <= 1e-5 and np.abs(res.stress - s0).max() <= 1e-8


@pytest.mark.full
def test_tiled_with_a_set_receiver_cap_keeps_it(monkeypatch):
    # Without hybrid GEMMs on either side, the plan's bounds (PLAN.md D60; with them 1.0e-6 eV/atom was seen).
    monkeypatch.setenv("LOUP_NO_HYBRID", "1")
    at = srtio3(10)
    at.positions += np.random.default_rng(4).normal(scale=0.03, size=at.positions.shape)
    bundle = _bundle()
    e0, f0, _ = _whole(bundle, at)
    eng = LoupEngine(bundle)
    try:
        res = capacity.evaluate(eng, at, CapacityConfig(mode="tiled", receivers=2000), skin=SKIN)
    finally:
        del eng
        gc.collect()
    assert res.mode == "tiled" and res.receivers == 2000 and res.tiles >= 3
    assert abs(res.energy - e0) / len(at) <= 1e-7 and np.abs(res.forces - f0).max() <= 1e-5


@pytest.mark.full
def test_auto_tiles_320000_atoms_without_a_user_setting():
    """S5-T05's acceptance: 320,000 atoms on the 16 GB card run without a user setting, the plan names the mode and
    the cap, and the device peak stays under 90% of the memory free for the call."""
    big, small = srtio3(40), srtio3(4)
    bundle = _bundle()
    e_small, f_small, _ = _whole(bundle, small)
    eng = LoupEngine(bundle)
    try:
        res = capacity.evaluate(eng, big, skin=SKIN)
    finally:
        del eng
        gc.collect()
    print(f"capacity: {len(big)} atoms, mode {res.mode}, receivers {res.receivers}, {res.tiles} tiles; plan whole "
          f"{res.plan.whole_bytes / 2**30:.2f} GiB, tiled global {res.plan.tiled_global_bytes / 2**30:.2f} GiB, "
          f"workspace {res.plan.tile_workspace_bytes / 2**30:.2f} GiB, budget {res.budget_bytes / 2**30:.2f} GiB, "
          f"peak {res.peak_bytes / 2**30:.2f} GiB")
    if res.plan.whole_bytes <= 0.9 * res.budget_bytes:
        pytest.skip("this GPU runs 320,000 atoms whole")
    assert res.mode == "tiled" and 4096 <= res.receivers <= 65536
    assert res.peak_bytes <= 0.9 * res.budget_bytes
    # A perfect crystal: the energy per atom of the small cell, forces at the small cell's rounding level.
    assert abs(res.energy / len(big) - e_small / len(small)) <= 1e-6
    assert np.abs(res.forces).max() <= np.abs(f_small).max() + 1e-5


def test_the_capacity_mode_refuses_what_is_not_built():
    eng = LoupEngine(_bundle())
    try:
        with pytest.raises(PlanRefused):
            capacity.evaluate(eng, srtio3(2), CapacityConfig(mode="wavefront"))
        with pytest.raises(PlanRefused):
            capacity.evaluate(eng, srtio3(2), CapacityConfig(state="host"))
    finally:
        del eng
        gc.collect()
