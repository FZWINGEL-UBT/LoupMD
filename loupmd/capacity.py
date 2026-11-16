"""The capacity mode (plan task S5-T05): a single-point evaluation that runs the whole-system call when the
system fits the GPU and the tiled call when it does not.

The engine plans the execution (``loup_plan_execution``): one probe tile of at most 4,096 receivers measures the
workspace per row and per edge, the whole-system call is chosen when it fits in ``memory_fraction`` of the memory
free for it, and otherwise the tiled call with the largest receiver cap of 4,096 to 65,536 that fits. A mode or a
cap set in :class:`~loupmd.config.CapacityConfig` is kept as given.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _core
from .config import CapacityConfig
from .errors import PlanRefused


@dataclass(frozen=True)
class CapacityResult:
    """Energy (eV), forces (eV/A, [n, 3]) and stress (eV/A^3, [3, 3]) with the execution that produced them."""

    energy: float
    forces: np.ndarray
    stress: np.ndarray
    mode: str               # "whole" or "tiled"
    receivers: int          # receiver cap of the tiled call; 0 for the whole-system call
    tiles: int
    plan: _core.MemoryPlan
    budget_bytes: int       # device memory free for the call (the handle's own buffers included)
    peak_bytes: int         # tiled: measured device peak of the call; whole: the plan's estimate


def evaluate(engine, atoms, config: CapacityConfig | None = None, skin: float = 0.5) -> CapacityResult:
    """Evaluate ``atoms`` (periodic) on ``engine`` (a ``LoupEngine``) in the mode the capacity plan chooses. The
    handle afterwards holds the system (whole) or the tile workspace (tiled): call ``set_system`` before other
    calls."""
    config = config or CapacityConfig()
    reasons = []
    if config.mode == "wavefront":
        reasons.append("capacity mode 'wavefront' is not in this version")
    if config.state == "host":
        reasons.append("state='host' (the tiled state in pinned host memory) is not in this version (plan task S5-T07)")
    if reasons:
        raise PlanRefused(reasons, next_step="Use capacity mode 'auto', 'whole' or 'tiled' with state 'auto' or "
                                             "'device'.")
    numbers = np.asarray(atoms.get_atomic_numbers())
    positions, cell = np.asarray(atoms.positions), np.asarray(atoms.cell.array)
    engine.set_execution(config.exec_mode, config.receivers or 0, config.memory_fraction)
    plan = engine.plan_execution(numbers, positions, cell, skin)
    if plan.mode == _core.LOUP_EXEC_WHOLE:
        engine.set_system(numbers, atoms.get_masses(), skin=skin)
        energy, forces, stress = engine.evaluate(positions, cell)
        return CapacityResult(energy, forces, stress, "whole", 0, 1, plan, plan.free_bytes, plan.whole_bytes)
    energy, forces, stress, tiles = engine.evaluate_tiled(numbers, positions, cell, plan.receivers, skin=skin)
    done = engine.last_execution()
    return CapacityResult(energy, forces, stress, "tiled", plan.receivers, tiles, plan, done.budget_bytes,
                          done.peak_bytes)
