"""loup_evaluate2 (plan task S1-T07): evaluation with flags. Built: stress on / off and the energy-only pass, which
runs the forward of the model and the ZBL energy and no gradients. Per-atom energies are refused until built.
Every test creates an engine and skips when its model file is absent."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest
from conftest import find_model, maxwell_velocities, srtio3

from loupmd import BackendError, LoupEngine, ensure_loup_bundle
from loupmd import _core


def _oxide(scale=1.0, seed=7):
    at = srtio3(3)
    at.positions += np.random.default_rng(seed).normal(scale=0.04, size=at.positions.shape)
    at.set_cell(at.cell.array * scale, scale_atoms=True)
    return at


def _nacl(scale=1.0):
    from ase.build import bulk

    at = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    at.positions += np.random.default_rng(21).normal(0.0, 0.05, at.positions.shape)
    at.set_cell(at.cell.array * scale, scale_atoms=True)
    return at


# (model, structure builder, generic=True forces the generic path for a model of the specialised architecture)
CASES = [
    ("mace-omat-0-medium.model", _oxide, False),      # specialised kernels
    ("mace-omat-0-medium.model", _oxide, True),       # the same model on the generic path
    ("mace-omat-0-small.model", _oxide, False),
    ("mace-mh-1.model", _oxide, False),
    ("MACE-POLAR-1-S.model", _nacl, False),           # long range
]
IDS = ["omat0-medium", "omat0-medium-generic", "omat0-small", "mh1", "polar1-s"]


@pytest.mark.parametrize("model,structure,generic", CASES, ids=IDS)
@pytest.mark.parametrize("scale", [1.0, 0.78], ids=["relaxed", "compressed"])
def test_energy_only_equals_the_full_calls_energy(model, structure, generic, scale):
    """The same readout and ZBL arithmetic per atom and per edge, summed in double: the two energies may differ by
    the order of the sums only. The compressed cell brings pairs inside the ZBL range."""
    at = structure(scale)
    zs = sorted(set(at.get_atomic_numbers().tolist()))
    eng = LoupEngine(ensure_loup_bundle(find_model(model), zs, generic=generic or None))
    eng.set_system(at.get_atomic_numbers(), at.get_masses())
    full, _, _ = eng.evaluate(at.positions, at.cell.array, compute_stress=False)
    only = eng.evaluate_energy(at.positions, at.cell.array)
    again, _, _ = eng.evaluate(at.positions, at.cell.array, compute_stress=False)
    assert np.isfinite(full) and np.isfinite(only)
    spread = abs(again - full)
    assert abs(only - full) <= max(5.0 * spread, 1e-11 * abs(full)), (only, full, again)


def test_energy_only_follows_new_positions_and_leaves_md_correct(sto_engine, capfd, monkeypatch):
    monkeypatch.setenv("LOUP_DEBUG_GRAPHS", "1")
    eng = sto_engine
    at = _oxide()
    eng.set_system(at.get_atomic_numbers(), at.get_masses(), skin=0.5)
    eng.set_state(at.positions, maxwell_velocities(at, 300.0, 3), at.cell.array)
    eng.run_nve(10, 1.0)
    pos, _, _, _ = eng.get_state()
    moved = pos.copy()
    moved[11] += (0.05, -0.03, 0.02)                    # a trial move, inside the skin
    e_moved = eng.evaluate_energy(moved, at.cell.array)
    e_back = eng.evaluate_energy(pos, at.cell.array)
    ref_moved, _, _ = eng.evaluate(moved, at.cell.array, compute_stress=False)
    ref_back, f_back, _ = eng.evaluate(pos, at.cell.array, compute_stress=False)
    assert abs(e_moved - ref_moved) <= 1e-9 * abs(ref_moved) and abs(e_back - ref_back) <= 1e-9 * abs(ref_back)
    assert abs(e_moved - e_back) > 1e-4                 # the move was seen

    capfd.readouterr()
    eng.evaluate_energy(pos, at.cell.array)
    st = eng.run_nve(5, 1.0)                            # forces were not computed: the run starts with a force pass
    assert capfd.readouterr().err.count("loupmd: initial force pass") == 1
    assert np.isfinite(st.final_temperature_k)
    assert np.isfinite(f_back).all()


def test_stress_flag_matches_loup_evaluate(sto_engine):
    eng = sto_engine
    at = _oxide()
    n = len(at)
    eng.set_system(at.get_atomic_numbers(), at.get_masses())
    e_ref, f_ref, s_ref = eng.evaluate(at.positions, at.cell.array)
    pos = np.ascontiguousarray(at.positions)
    cell = np.ascontiguousarray(at.cell.array)
    dp = ctypes.POINTER(ctypes.c_double)
    for flags in (_core.LOUP_EVAL_STRESS, 0):
        energy = ctypes.c_double()
        forces = np.zeros((n, 3))
        stress = np.full((3, 3), 7.0)
        rc = eng._lib.loup_evaluate2(eng._h, pos.ctypes.data_as(dp), cell.ctypes.data_as(dp), flags,
                                     ctypes.byref(energy), forces.ctypes.data_as(dp), stress.ctypes.data_as(dp), None)
        assert rc == 0, eng.last_error()
        assert abs(energy.value - e_ref) < 1e-6 * n and np.abs(forces - f_ref).max() < 1e-5
        if flags:
            assert np.abs(stress - s_ref).max() < 1e-7
        else:
            assert (stress == 7.0).all()                # not written without the flag


@pytest.mark.parametrize("flags,reason", [
    (_core.LOUP_EVAL_PER_ATOM, "loup_evaluate2: LOUP_EVAL_PER_ATOM is not built in this version"),
    (_core.LOUP_EVAL_ENERGY_ONLY | _core.LOUP_EVAL_STRESS, "loup_evaluate2: LOUP_EVAL_ENERGY_ONLY excludes LOUP_EVAL_STRESS"),
    (64, "loup_evaluate2: unknown evaluation flags 0x40"),
], ids=["per_atom", "energy_only_with_stress", "unknown_flag"])
def test_evaluate2_refusals(sto_engine, flags, reason):
    eng = sto_engine
    at = _oxide()
    eng.set_system(at.get_atomic_numbers(), at.get_masses())
    pos = np.ascontiguousarray(at.positions)
    cell = np.ascontiguousarray(at.cell.array)
    dp = ctypes.POINTER(ctypes.c_double)
    energy = ctypes.c_double()
    rc = eng._lib.loup_evaluate2(eng._h, pos.ctypes.data_as(dp), cell.ctypes.data_as(dp), flags,
                                 ctypes.byref(energy), None, None, None)
    assert rc == -1 and eng.last_error().startswith(reason), eng.last_error()


def test_energy_only_without_a_system_is_refused(sto_bundle):
    eng = LoupEngine(sto_bundle)
    with pytest.raises(BackendError, match="loup_evaluate2: no system is set"):
        eng.evaluate_energy(np.zeros((0, 3)), np.eye(3))
