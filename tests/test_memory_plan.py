"""loup_plan_memory (plan task S1-T10): the device memory of a system, scaled from a small one of the same material.

The comparison with measured memory reads the device's free memory, so it is only valid on a GPU no other process
is allocating on; every test creates an engine.
"""
from __future__ import annotations

import numpy as np
import pytest
from conftest import find_model, srtio3

from loupmd import BackendError, LoupEngine, ensure_loup_bundle
from loupmd import _core


def _probe(bundle, rep=3):
    at = srtio3(rep)
    at.positions += np.random.default_rng(1).normal(scale=0.02, size=at.positions.shape)
    eng = LoupEngine(bundle)
    eng.set_system(at.get_atomic_numbers(), at.get_masses(), skin=0.5)
    eng.evaluate(at.positions, at.cell.array)
    return eng


def _measured(bundle, probe, rep):
    """Device memory a rep x rep x rep system takes from set_system to its first evaluation, read as the drop of
    free memory (the engine and its model are created before the first reading)."""
    at = srtio3(rep)
    at.positions += np.random.default_rng(1).normal(scale=0.02, size=at.positions.shape)
    eng = LoupEngine(bundle)
    before = probe.plan_memory(1).free_bytes
    eng.set_system(at.get_atomic_numbers(), at.get_masses(), skin=0.5)
    eng.evaluate(at.positions, at.cell.array)
    after = probe.plan_memory(1).free_bytes
    own = eng.plan_memory(len(at))  # the same size, from its own buffers
    del eng
    return before - after, own.whole_bytes


@pytest.mark.parametrize("rep", [10, 16, 20], ids=["5000", "20480", "40000"])
def test_prediction_from_a_small_cell_matches_the_measured_memory(sto_bundle, rep):
    probe = _probe(sto_bundle)
    n = 5 * rep ** 3
    predicted = probe.plan_memory(n)
    assert predicted.mode == _core.LOUP_EXEC_WHOLE
    measured, own = _measured(sto_bundle, probe, rep)
    print(f"memory plan {n} atoms: predicted {predicted.whole_bytes / 1e6:.1f} MB, measured {measured / 1e6:.1f} MB, "
          f"own {own / 1e6:.1f} MB")
    assert abs(predicted.whole_bytes - measured) < 0.10 * measured, (predicted.whole_bytes, measured)
    assert abs(predicted.whole_bytes - own) < 0.03 * own, (predicted.whole_bytes, own)  # the scaling itself


def test_prediction_on_the_generic_path(tmp_path):
    bundle = ensure_loup_bundle(find_model("mace-omat-0-small.model"), [8, 22, 38])
    probe = _probe(bundle)
    predicted = probe.plan_memory(5000)
    measured, own = _measured(bundle, probe, 10)
    print(f"memory plan generic 5000 atoms: predicted {predicted.whole_bytes / 1e6:.1f} MB, measured "
          f"{measured / 1e6:.1f} MB, own {own / 1e6:.1f} MB")
    assert abs(predicted.whole_bytes - measured) < 0.10 * measured, (predicted.whole_bytes, measured)
    assert abs(predicted.whole_bytes - own) < 0.03 * own


def test_plan_scales_with_atoms_and_edges_and_names_what_does_not_fit(sto_bundle):
    probe = _probe(sto_bundle)
    small, large = probe.plan_memory(5000), probe.plan_memory(50000)
    assert 9.0 < large.whole_bytes / small.whole_bytes < 10.5
    dense = probe.plan_memory(5000, edges_per_atom=400.0)
    sparse = probe.plan_memory(5000, edges_per_atom=40.0)
    assert dense.whole_bytes > sparse.whole_bytes
    assert small.free_bytes > 0 and small.tiled_global_bytes == small.tile_workspace_bytes == small.receivers == 0
    too_large = probe.plan_memory(50_000_000)  # of the order of terabytes: needs the capacity mode
    assert too_large.whole_bytes > 10 * too_large.free_bytes and too_large.mode == _core.LOUP_EXEC_TILED


def test_plan_refusals(sto_bundle):
    eng = LoupEngine(sto_bundle)
    with pytest.raises(BackendError, match="loup_plan_memory: no system is set"):
        eng.plan_memory(1000)
    at = srtio3(2)
    eng.set_system(at.get_atomic_numbers(), at.get_masses())
    with pytest.raises(BackendError, match="loup_plan_memory: no neighbour list to scale from"):
        eng.plan_memory(1000)
    eng.evaluate(at.positions, at.cell.array)
    with pytest.raises(BackendError, match="loup_plan_memory: n_atoms must be positive, got 0"):
        eng.plan_memory(0)
    assert eng._lib.loup_plan_memory(eng._h, 1000, 0.0, None) == -1
    assert eng.last_error() == "loup_plan_memory: out is NULL"
