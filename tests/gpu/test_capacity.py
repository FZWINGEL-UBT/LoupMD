"""The capacity mode (plan task S5-T05): ``capacity.evaluate`` asks the engine for an execution plan (one probe
tile measures the workspace per row and per edge), runs the whole-system call when it fits with 10% headroom and
the tiled call with the largest receiver cap of 4,096 to 65,536 that fits otherwise. Every test creates engines
and skips when its model file is absent."""
from __future__ import annotations

import gc

import numpy as np
import pytest
from gpu_support import find_model, srtio3

from loupmd import LoupEngine, PlanRefused, _core, capacity, ensure_loup_bundle
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
    # a two-layer model's tiles fold the skip chain where its whole system does (ledger tile-chain-fold)
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
            capacity.evaluate(eng, srtio3(2), CapacityConfig(state="host"))
    finally:
        del eng
        gc.collect()


@pytest.mark.full
def test_the_wavefront_mode_plans_a_smaller_state_and_gives_the_tiled_result(monkeypatch):
    """S5-T08: capacity mode 'wavefront' runs the tiled call in the wavefront order; the plan counts the row pool
    instead of the whole system's layer outputs. Without hybrids the plain tiled call's bounds apply."""
    monkeypatch.setenv("LOUP_NO_HYBRID", "1")
    monkeypatch.setenv("LOUP_TILE_FOLD", "0")          # the wavefront never folds: the plain call's same operations
    at = srtio3(1).repeat((64, 6, 6))                  # 11,520 atoms in a rod: 13 layers of tiles at 600
    at.positions += np.random.default_rng(7).normal(scale=0.04, size=at.positions.shape)
    bundle = _bundle()
    runs = {}
    for mode in ("tiled", "wavefront"):
        eng = LoupEngine(bundle)
        try:
            runs[mode] = capacity.evaluate(eng, at, CapacityConfig(mode=mode, receivers=600), skin=SKIN)
        finally:
            del eng
            gc.collect()
    tiled, wave = runs["tiled"], runs["wavefront"]
    print(f"wavefront mode: plan state {wave.plan.tiled_global_bytes / 2**20:.0f} MiB against "
          f"{tiled.plan.tiled_global_bytes / 2**20:.0f} MiB tiled; peak {wave.peak_bytes / 2**20:.0f} MiB against "
          f"{tiled.peak_bytes / 2**20:.0f} MiB")
    assert tiled.mode == "tiled" and wave.mode == "wavefront" and wave.receivers == 600
    # the pool holds 26% of the atoms (test_tiled.py); the kept lists and the forces stay whole (measured 0.61)
    assert wave.plan.tiled_global_bytes < 0.7 * tiled.plan.tiled_global_bytes
    assert abs(wave.energy - tiled.energy) <= 1e-9 * abs(tiled.energy)
    assert np.abs(wave.forces - tiled.forces).max() <= 1e-5


@pytest.mark.full
def test_from_2_20_atoms_a_folding_model_keeps_its_plain_order_on_the_device(monkeypatch):
    """List 21 (ledger tile-fold-plan): from 2^20 atoms the wavefront and the two host-state rules apply by default,
    but a two-layer model whose tiles fold the skip chain plans the folded plain order on the device wherever it fits
    (the plan of LOUP_TILE_WAVEFRONT=0 LOUP_TILE_HOST_STATE=0), and the tiled call runs that order. Where it fits
    nowhere the plan falls back to the wavefront (the plan of LOUP_TILE_FOLD=0 with the host state off). MH-1 on
    1,050,000 atoms in a rod, so that the wavefront applies at every cap."""
    at = srtio3(1).repeat((210, 40, 25))
    bundle = ensure_loup_bundle(find_model("mace-mh-1.model"), [8, 22, 38], head="omat_pbe")
    for k in ("LOUP_TILE_FOLD", "LOUP_TILE_WAVEFRONT", "LOUP_TILE_HOST_STATE", "LOUP_TILE_HOST_LISTS",
              "LOUP_CHAIN_FOLD", "LOUP_DEBUG_PLAN_FREE"):
        monkeypatch.delenv(k, raising=False)
    probe = LoupEngine(bundle)
    cc = probe.info().compute_capability
    del probe
    gc.collect()
    if cc not in (80, 120):
        pytest.skip("the skip-chain fold is built by default on sm_120 and cc 8.0 only")

    def plan(call=False, **env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        eng = LoupEngine(bundle)
        try:
            eng.set_execution(_core.LOUP_EXEC_AUTO, 0, 0.9)
            p = eng.plan_execution(at.numbers, at.positions, at.cell.array, skin=SKIN)
            ex = None
            if call:
                eng.evaluate_tiled(at.numbers, at.positions, at.cell.array, 0, SKIN)
                ex = eng.last_execution()
            return p, ex
        finally:
            for k in env:
                monkeypatch.delenv(k)
            del eng
            gc.collect()

    def show(name, p):
        print(f"{name}: mode {p.mode}, cap {p.receivers}, host state {p.host_state}, host lists {p.host_lists}, "
              f"global {p.tiled_global_bytes / 2**30:.3f} GiB, workspace {p.tile_workspace_bytes / 2**30:.3f} GiB")

    default, ex = plan(call=True)
    plain, _ = plan(LOUP_TILE_WAVEFRONT="0", LOUP_TILE_HOST_STATE="0")
    old, _ = plan(LOUP_TILE_FOLD="0")
    for name, p in (("default", default), ("plain", plain), ("unfolded", old)):
        show(name, p)
    assert default.mode == _core.LOUP_EXEC_TILED and not default.host_state
    assert (default.receivers, default.host_lists) == (plain.receivers, plain.host_lists)
    assert abs(default.tiled_global_bytes - plain.tiled_global_bytes) <= 0.02 * plain.tiled_global_bytes
    # unfolded, the plain order's state does not fit: the host state or the wavefront, which hold far less there
    assert old.tiled_global_bytes < 0.8 * default.tiled_global_bytes
    assert ex.mode == _core.LOUP_EXEC_TILED and ex.receivers == default.receivers
    # a card that holds the wavefront's pool but not the folded state: the wavefront, as without the fold (on the
    # 5060 Ti at 8,000 MiB the folded order still fits at cap 4,096 with host lists, 9.26 GiB planned against the
    # wavefront's 7.15 GiB there)
    fb, _ = plan(LOUP_DEBUG_PLAN_FREE="6700", LOUP_TILE_HOST_STATE="0")
    ref, _ = plan(LOUP_DEBUG_PLAN_FREE="6700", LOUP_TILE_HOST_STATE="0", LOUP_TILE_FOLD="0")
    for name, p in (("fallback", fb), ("fallback unfolded", ref)):
        show(name, p)
    assert fb.receivers == ref.receivers > 0 and fb.host_lists == ref.host_lists
    assert abs(fb.tiled_global_bytes - ref.tiled_global_bytes) <= 0.02 * ref.tiled_global_bytes
    assert fb.tiled_global_bytes < 0.8 * plain.tiled_global_bytes


def test_the_wavefront_mode_refuses_a_three_layer_model(monkeypatch):
    from gpu_support import water_box

    from loupmd import BackendError

    monkeypatch.delenv("LOUP_CHAIN_FOLD", raising=False)   # a default engine: a tiled system leaves the fold off
    at = water_box()
    eng = LoupEngine(ensure_loup_bundle(find_model("MACE-omol-0-extra-large-1024.model"),
                                        sorted(set(at.numbers.tolist()))))
    try:
        with pytest.raises(BackendError, match="the wavefront mode needs a model of two layers"):
            capacity.evaluate(eng, at, CapacityConfig(mode="wavefront", receivers=200), skin=SKIN)
    finally:
        del eng
        gc.collect()
