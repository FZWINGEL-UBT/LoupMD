"""The step-displacement maximum decays (plan task S1-T11, review item S2).

The pipelined loop needs four times the largest per-step displacement to stay below skin / 2. That maximum used to
be kept from set_state on, so one fast step held a run unpipelined for good. It now covers the current and the
previous window only. Every test creates an engine.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from gpu_support import find_model, maxwell_velocities, srtio3

from loupmd import LoupEngine, ensure_loup_bundle

KICK = 0.75  # A per ASE time unit: 0.074 A in a 1 fs step, above skin / 8 = 0.0625 A at skin 0.5


def _kicked_start(eng, temp_k):
    """SrTiO3 4x4x4 at ``temp_k`` with one oxygen fast enough to forbid pipelining for as long as it is remembered."""
    at = srtio3(4)
    eng.set_system(at.get_atomic_numbers(), at.get_masses(), skin=0.5)
    vel = maxwell_velocities(at, temp_k, 3) if temp_k > 0 else np.zeros((len(at), 3))
    assert at.get_chemical_symbols()[7] == "O"
    vel[7] = (KICK, 0.0, 0.0)
    eng.set_state(at.positions, vel, at.cell.array)
    return at


def test_one_fast_step_no_longer_keeps_a_run_unpipelined(sto_engine):
    eng = sto_engine
    _kicked_start(eng, 300.0)
    first = eng.run_nve(40, 1.0)
    assert first.pipelined_steps == 0, "the kick was meant to forbid pipelining at first (the test's premise)"
    eng.run_nve(1200, 1.0)  # two windows pass: by rebuilds, or by the 512-step limit at the latest
    later = eng.run_nve(300, 1.0)
    # Before this task the kick's displacement was remembered until the next set_state: 0 pipelined steps here.
    assert later.pipelined_steps > 200, later.pipelined_steps
    assert first.dangerous_builds == later.dangerous_builds == 0


def test_a_cold_system_without_rebuilds_forgets_a_fast_start_too(sto_engine):
    """With no rebuild to end a window, a window still ends after 512 steps."""
    eng = sto_engine
    _kicked_start(eng, 0.0)
    cold = dict(dt_fs=1.0, temp_k=1.0, friction_per_fs=0.2, seed=1)  # the kick is damped within a few steps
    first = eng.run_langevin(40, **cold)
    assert first.pipelined_steps == 0, "the test's premise"
    middle = eng.run_langevin(1100, **cold)
    later = eng.run_langevin(100, **cold)
    assert middle.rebuilds == 0 and later.rebuilds == 0, "the premise: nothing moves far enough to rebuild"
    assert later.pipelined_steps > 90, later.pipelined_steps


def test_reheating_after_the_maximum_has_decayed_uses_no_incomplete_list(sto_engine):
    """The decayed maximum must follow a system that speeds up again: the current window sees every step before the
    check that uses it."""
    eng = sto_engine
    _kicked_start(eng, 100.0)
    eng.run_langevin(1300, 1.0, temp_k=100.0, friction_per_fs=0.05, seed=2)
    dangerous = piped = 0
    for cycle in range(4):
        up = eng.run_langevin(150, 1.0, temp_k=1500.0, friction_per_fs=0.05, seed=10 + cycle)
        down = eng.run_langevin(300, 1.0, temp_k=100.0, friction_per_fs=0.1, seed=20 + cycle)
        dangerous += up.dangerous_builds + down.dangerous_builds
        piped += up.pipelined_steps + down.pipelined_steps
    assert dangerous == 0
    assert piped > 900, piped  # of 1,800: the cycles do run pipelined, so the check above tests the pipelined loop


@pytest.mark.parametrize("skin,pipelines", [(0.5, False), (1.0, True)])
def test_water_pipelines_at_a_skin_its_hydrogens_allow(skin, pipelines):
    """Hydrogen at 300 K moves about 0.08 A per 1 fs step, up to 0.11 A (ledger, md-skin, S46): four times that is
    above skin / 2 at skin 0.5 in every step, so no decay can pipeline water there; at skin 1.0 it fits."""
    from ase.io import read

    from loupmd.bench import suites

    box = suites.find_file("water_216.extxyz", {})      # shipped with the package
    at = read(box)
    eng = LoupEngine(ensure_loup_bundle(find_model("MACE-OFF23_small.model"), [1, 8]))
    eng.set_system(at.get_atomic_numbers(), at.get_masses(), skin=skin)
    eng.set_state(at.positions, maxwell_velocities(at, 300.0, 5), at.cell.array)
    eng.run_langevin(200, 1.0, temp_k=300.0, friction_per_fs=0.01, seed=1)
    st = [eng.run_langevin(500, 1.0, temp_k=300.0, friction_per_fs=0.01, seed=1) for _ in range(4)]
    piped = sum(s.pipelined_steps for s in st)
    assert sum(s.dangerous_builds for s in st) == 0
    assert 250.0 < st[-1].final_temperature_k < 350.0
    if pipelines:
        assert piped > 1500, piped
    else:
        assert piped == 0, piped
