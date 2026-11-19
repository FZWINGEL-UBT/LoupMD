"""loup_compute_device (LAMMPS_KOKKOS_PLAN.md stage B): the evaluation on arrays that stay on the GPU, as the LAMMPS
Kokkos pair style calls it, and loup_evaluate's one-graph path, on every neighbour-rebuild path of the step graph they
share with the MD loop. The device arrays here are torch CUDA tensors."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import find_model, srtio3

from loupmd import LoupEngine, ensure_loup_bundle

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _bundle():
    return ensure_loup_bundle(find_model("mace-omat-0-small.model"), [8, 22, 38])


def _engine(n_rep=3, seed=3):
    at = srtio3(n_rep)
    at.positions += np.random.default_rng(seed).normal(scale=0.05, size=at.positions.shape)
    eng = LoupEngine(_bundle())
    eng.set_system(at.get_atomic_numbers(), at.get_masses(), skin=0.5)
    return at, eng


def _virial(stress, volume):
    """loup_evaluate's stress (dE/d strain / V, 3 x 3) as loup_compute_device's virial (xx yy zz xy xz yz, eV)."""
    return -volume * np.array([stress[0, 0], stress[1, 1], stress[2, 2], stress[0, 1], stress[0, 2], stress[1, 2]])


def test_device_call_matches_evaluate_for_shuffled_rows_and_both_layouts():
    at, eng = _engine()
    n = len(at)
    cell = at.cell.array
    e_ref, f_ref, s_ref = eng.evaluate(at.positions, cell)

    # rows in another order than the engine's atoms, as LAMMPS's local atoms are after sorting
    order = np.random.default_rng(0).permutation(n)
    ids = torch.tensor(order, dtype=torch.int32, device="cuda")
    x = torch.tensor(at.positions[order], dtype=torch.float64, device="cuda")

    # row-major [n][3] forces (strides 3, 1)
    f = torch.zeros((n, 3), dtype=torch.float64, device="cuda")
    e, v = eng.compute_device(x.data_ptr(), cell, f.data_ptr(), ids.data_ptr(), virial=True)
    assert abs(e - e_ref) < 1e-6 * n
    assert np.abs(f.cpu().numpy() - f_ref[order]).max() < 2e-5
    assert np.abs(v - _virial(s_ref, at.get_volume())).max() < 2e-4

    # column-major [n_alloc][3] forces with n_alloc > n (Kokkos::LayoutLeft, strides 1, n_alloc); forces are added
    n_alloc = n + 7
    g = torch.full((3, n_alloc), 1.0, dtype=torch.float64, device="cuda")
    e2, _ = eng.compute_device(x.data_ptr(), cell, g.data_ptr(), ids.data_ptr(), f_strides=(1, n_alloc))
    assert abs(e2 - e_ref) < 1e-6 * n
    got = g.cpu().numpy()
    assert np.abs(got[:, :n].T - 1.0 - f_ref[order]).max() < 2e-5
    assert np.all(got[:, n:] == 1.0)  # rows beyond n_rows untouched


# The step graph's rebuild paths (LOUP_MD_REBUILD, read when an engine is created) and the atom reorder.
PATHS = [
    pytest.param({}, id="device-rebuild"),
    pytest.param({"LOUP_MD_REBUILD": "host"}, id="host-rebuild"),
    pytest.param({"LOUP_MD_REBUILD": "fallback"}, id="fallback-rebuild"),
    pytest.param({"LOUP_REORDER": "1"}, id="atom-reorder"),
]


@pytest.mark.parametrize("env", PATHS)
@pytest.mark.parametrize("call", ["device", "evaluate"])
def test_calls_follow_moves_and_neighbour_rebuilds(env, call, monkeypatch):
    """Small moves keep the list, moves beyond skin / 2 rebuild it (the first one host-driven, with the variant
    timing; later ones on the path under test): every result equals a freshly built list's."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    at, eng = _engine()
    for k in env:
        monkeypatch.delenv(k)
    ref = LoupEngine(_bundle())  # the reference: default paths, the system set again (a new list) for every call
    n = len(at)
    cell = at.cell.array
    z, m = at.get_atomic_numbers(), at.get_masses()
    rng = np.random.default_rng(1)
    x = torch.tensor(at.positions, dtype=torch.float64, device="cuda")
    f = torch.zeros((n, 3), dtype=torch.float64, device="cuda")
    pos = at.positions.copy()
    for step, scale in enumerate([0.0, 0.01, 0.15, 0.01, 0.15, 0.01, 0.15, 0.01]):
        pos = pos + rng.normal(scale=scale, size=pos.shape)
        if call == "device":
            x.copy_(torch.tensor(pos, dtype=torch.float64))
            f.zero_()
            # without energy the call returns early: the forces are read after the default stream is synchronised
            eng.compute_device(x.data_ptr(), cell, f.data_ptr(), energy=False)
            torch.cuda.synchronize()
            forces = f.cpu().numpy()
            e, _ = eng.compute_device(x.data_ptr(), cell, 0)
        else:
            e, forces, _ = eng.evaluate(pos, cell)
        ref.set_system(z, m, skin=0.5)
        e_ref, f_ref, _ = ref.evaluate(pos, cell)
        assert abs(e - e_ref) < 1e-6 * n, step
        assert np.abs(forces - f_ref).max() < 5e-5, step


def test_device_call_refuses_a_wrong_row_count_and_a_bad_id():
    at, eng = _engine(2)
    n = len(at)
    cell = at.cell.array
    x = torch.tensor(at.positions, dtype=torch.float64, device="cuda")
    f = torch.zeros((n, 3), dtype=torch.float64, device="cuda")
    ids = torch.arange(n, dtype=torch.int32, device="cuda")
    e0, _ = eng.compute_device(x.data_ptr(), cell, f.data_ptr(), ids.data_ptr())
    ids[0] = n  # outside 0 .. n-1
    with pytest.raises(RuntimeError):
        eng.compute_device(x.data_ptr(), cell, f.data_ptr(), ids.data_ptr())
    ids[0] = 0  # the corrected call is accepted again
    e1, _ = eng.compute_device(x.data_ptr(), cell, f.data_ptr(), ids.data_ptr())
    assert abs(e1 - e0) < 1e-6 * n
    eng._n = n - 1  # a row count that is not the system's
    with pytest.raises(RuntimeError):
        eng.compute_device(x.data_ptr(), cell, f.data_ptr())
    eng._n = n


def test_run_calls_in_blocks_continue_one_trajectory():
    """Run calls with the same parameters reuse their graphs: two NVE blocks give the positions of one run."""
    at, eng = _engine()
    vel = np.random.default_rng(5).normal(scale=0.02, size=at.positions.shape)
    eng.set_state(at.positions, vel, at.cell.array)
    eng.run_nve(40, dt_fs=0.5)
    pos_one, _, _, _ = eng.get_state()
    at2, eng2 = _engine()
    eng2.set_state(at2.positions, vel, at2.cell.array)
    eng2.run_nve(20, dt_fs=0.5)
    eng2.run_nve(20, dt_fs=0.5)
    pos_two, _, _, _ = eng2.get_state()
    assert np.abs(pos_one - pos_two).max() < 1e-6


def test_model_facts():
    at, eng = _engine(2)
    assert eng.cutoff() == pytest.approx(6.0)
    assert sorted(eng.species()) == [8, 22, 38]
