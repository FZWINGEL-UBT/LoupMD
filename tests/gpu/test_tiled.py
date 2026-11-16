"""The tiled call (plan task S5-T03): a periodic system evaluated tile by tile with the per-layer sweeps, the state
between sweeps in whole-system arrays on the device. Energy, forces and stress must be those of the whole-system
call. Every test creates engines and skips when its model file is absent."""
from __future__ import annotations

import ctypes
import gc

import numpy as np
import pytest
from gpu_support import find_model, srtio3, water_box

from loupmd import BackendError, LoupEngine, ensure_loup_bundle
from loupmd import _core

SKIN = 0.5
c_ll, c_int, c_double, ptr = ctypes.c_longlong, ctypes.c_int, ctypes.c_double, ctypes.POINTER


def _tiled(bundle, at, receivers):
    eng = LoupEngine(bundle)
    lib = eng._lib
    lib.loup_debug_evaluate_tiled.argtypes = [ctypes.c_void_p, c_int, ptr(c_int), ptr(c_double), ptr(c_double), c_int,
                                              c_double, ptr(c_double), ptr(c_double), ptr(c_double), ptr(c_ll)]
    z = np.ascontiguousarray(at.get_atomic_numbers(), dtype=np.int32)
    pos = np.ascontiguousarray(at.positions, dtype=np.float64)
    cell = np.ascontiguousarray(at.cell.array, dtype=np.float64)
    energy, tiles = c_double(), c_ll()
    forces, stress = np.zeros((len(at), 3)), np.zeros((3, 3))
    rc = lib.loup_debug_evaluate_tiled(eng._h, len(at), z.ctypes.data_as(ptr(c_int)), pos.ctypes.data_as(ptr(c_double)),
                                       cell.ctypes.data_as(ptr(c_double)), receivers, SKIN, ctypes.byref(energy),
                                       forces.ctypes.data_as(ptr(c_double)), stress.ctypes.data_as(ptr(c_double)),
                                       ctypes.byref(tiles))
    error = _core.last_error(eng._h) if rc != 0 else None
    del eng                                            # also on failure: pytest keeps a failed test's frames
    gc.collect()
    if error is not None:
        raise BackendError(error, call="loup_debug_evaluate_tiled")
    return energy.value, forces, stress, tiles.value


def _whole(bundle, at):
    eng = LoupEngine(bundle)
    try:
        eng.set_system(at.get_atomic_numbers(), at.get_masses(), skin=SKIN)
        return eng.evaluate(at.positions, at.cell.array)
    finally:
        del eng                                        # also on failure: pytest keeps a failed test's frames
        gc.collect()


def _oxide(rep):
    at = srtio3(rep)
    at.positions += np.random.default_rng(5).normal(scale=0.04, size=at.positions.shape)
    at.set_cell(at.cell.array * np.array([[1.0, 0.006, 0.0], [0.0, 1.0, 0.004], [0.003, 0.0, 1.0]]), scale_atoms=True)
    return at


def _water(reps):
    at = water_box().repeat(reps)
    at.positions += np.random.default_rng(6).normal(scale=0.02, size=at.positions.shape)
    return at


# (model file, structure, receivers per tile, layers). SrTiO3 20x20x20 and the water box 4x4x4 are 40,000 and
# 41,472 atoms; MH-1's whole-system call does not fit 16 GB at 40,000 atoms, so 17x17x17 (24,565); OMOL-0 extra
# large (1024 channels) on 2 x 648 atoms.
CASES = [
    ("mace-omat-0-medium.model", lambda: _oxide(20), 10000, 2),
    ("mace-omat-0-small.model", lambda: _oxide(20), 10000, 2),
    ("mace-mh-1.model", lambda: _oxide(17), 6000, 2),
    ("MACE-OFF23_medium.model", lambda: _water((4, 4, 4)), 10000, 2),
    ("MACE-omol-0-extra-large-1024.model", lambda: _water((2, 1, 1)), 400, 3),
]
IDS = [c[0].split(".model")[0] for c in CASES]


@pytest.mark.parametrize("model,structure,receivers,layers", CASES, ids=IDS)
@pytest.mark.full
def test_the_tiled_call_gives_the_whole_system_result(model, structure, receivers, layers, monkeypatch):
    if layers == 3:
        monkeypatch.setenv("LOUP_CHAIN_FOLD", "0")   # the fold never forms the layer outputs a sweep hands on
    at = structure()
    n = len(at)
    zs = sorted(set(at.get_atomic_numbers().tolist()))
    bundle = ensure_loup_bundle(find_model(model), zs)
    e_a, f_a, s_a = _whole(bundle, at)
    e_t, f_t, s_t, tiles = _tiled(bundle, at, receivers)
    assert tiles >= 2
    # The tensor-core hybrid GEMMs are chosen per GEMM shape, and a tile's GEMMs have other row counts than the
    # whole system's: where one side gets a hybrid kernel and the other a float32 one, the energies differ by the
    # hybrid's rounding (PLAN.md D16). Without hybrids on either side the plan's bounds apply.
    monkeypatch.setenv("LOUP_NO_HYBRID", "1")
    e_c, f_c, s_c = _whole(bundle, at)
    e_n, f_n, s_n, _ = _tiled(bundle, at, receivers)
    monkeypatch.delenv("LOUP_NO_HYBRID")

    def diff(e, f, s, e0, f0, s0):
        return abs(e - e0) / n, np.abs(f - f0).max(), np.abs(s - s0).max()

    default, plain, kernels = diff(e_t, f_t, s_t, e_a, f_a, s_a), diff(e_n, f_n, s_n, e_c, f_c, s_c),         diff(e_c, f_c, s_c, e_a, f_a, s_a)
    print(f"tiled {model}: {n} atoms, {tiles} tiles; default GEMMs dE {default[0]:.2e} eV/atom, dF {default[1]:.2e} "
          f"eV/A, dS {default[2]:.2e} eV/A^3; no hybrids dE {plain[0]:.2e}, dF {plain[1]:.2e}, dS {plain[2]:.2e}; "
          f"whole system between the two kernel sets {kernels[0]:.1e}, {kernels[1]:.1e}, {kernels[2]:.1e}")
    # Without hybrids: the plan's bounds (1e-7 eV/atom, 1e-5 eV/A; stress a tenth of the gate's floor). With the
    # default GEMMs: forces as the plan; energy and stress at the open domain's bounds (D16, D60): a tenth of the
    # gate's energy floor (0.01 meV/atom) and its stress floor (1e-7 eV/A^3).
    for name, d, bound in zip(("energy", "forces", "stress"), plain, (1e-7, 1e-5, 1e-8)):
        assert d <= bound, f"{name}, no hybrids: the tiled call differs by {d:.3e}"
    for name, d, bound in zip(("energy", "forces", "stress"), default, (1e-6, 1e-5, 1e-7)):
        assert d <= bound, f"{name}, default GEMMs: the tiled call differs by {d:.3e}"


def test_a_tiled_call_on_a_kept_workspace_gives_a_fresh_engines_result():
    """The handle keeps the tile workspace between tiled calls (ledger `tile-kept-lists`): a later call with
    other tiles, or on a smaller system, must give what a new engine gives."""
    big, small = _oxide(8), _oxide(6)
    bundle = ensure_loup_bundle(find_model("mace-omat-0-small.model"), [8, 22, 38])

    def fresh(at, receivers):
        eng = LoupEngine(bundle)
        try:
            return eng.evaluate_tiled(at.numbers, at.positions, at.cell.array, receivers, skin=SKIN)
        finally:
            del eng
            gc.collect()

    eng = LoupEngine(bundle)
    try:
        calls = [(big, 1000), (big, 400), (small, 300), (big, 1000)]
        kept = [eng.evaluate_tiled(at.numbers, at.positions, at.cell.array, r, skin=SKIN) for at, r in calls]
    finally:
        del eng
        gc.collect()
    for (at, r), (e, f, s, tiles) in zip(calls, kept):
        e0, f0, s0, tiles0 = fresh(at, r)
        assert tiles == tiles0 >= 2
        # The same GEMM shapes can get other plans on a new engine only through the autotune's timing; the hybrid
        # rounding bounds of the tiled test apply (D60).
        assert abs(e - e0) / len(at) <= 1e-6 and np.abs(f - f0).max() <= 1e-5 and np.abs(s - s0).max() <= 1e-7
    e1, f1, _, _ = kept[0]
    e4, f4, _, _ = kept[3]
    # Same engine, tiles and plans: the same numbers up to the order of the float32 atomic sums (measured up to
    # 3.5e-9 eV/atom and 1.03e-6 eV/A); stale state from an earlier call would be far larger.
    assert abs(e4 - e1) / len(big) <= 1e-8 and np.abs(f4 - f1).max() <= 1e-5


def test_the_tiled_call_is_refused_for_long_range_models():
    at = _oxide(1)
    bundle = ensure_loup_bundle(find_model("MACE-POLAR-1-S.model"), [8, 22, 38])
    with pytest.raises(BackendError, match="the tiled call needs a bundle of the generic path without long range"):
        _tiled(bundle, at, 100)
