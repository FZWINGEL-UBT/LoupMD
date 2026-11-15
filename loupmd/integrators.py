"""``lm.integrators``: the integrator records a ``Simulation`` takes as ``integrator=``.

Names and arguments are FastMACE's (``fm.integrators``). Units: fs, K, friction per fs. A thermostat's temperature
and friction have no defaults: physics is not chosen silently. Each record maps onto an ``IntegratorConfig``
(``to_config``), which is what the engine runs. Built so far: ``NVE`` and ``Langevin``; the other schemes of
FastMACE arrive with the engine's integrators (PLAN.md Stage 4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import IntegratorConfig
from .errors import InputError, PlanRefused

__all__ = ["Langevin", "NVE"]


def _resolve_fixcm(record):
    """``fixcm`` as the engine can honour it today. FastMACE removes the centre-of-mass momentum by default
    (``"auto"``); LoupMD's engine cannot yet (PLAN.md S4-T01), so ``"auto"`` is off here and ``True`` is refused."""
    if not isinstance(record.fixcm, bool) and record.fixcm != "auto":
        raise InputError(f"{type(record).__name__}.fixcm must be True, False or 'auto', got {record.fixcm!r}")
    if record.fixcm is True:
        raise PlanRefused("fixcm=True: removing the centre-of-mass momentum is not in this version",
                          why="the engine's integrators do not project it out yet (PLAN.md S4-T01)",
                          next_step="Pass fixcm=False (or leave 'auto', which is off until then), and start from "
                                    "velocities without a net momentum.")


@dataclass(frozen=True, kw_only=True)
class NVE:
    """Velocity Verlet: ``lm.integrators.NVE(timestep_fs=1.0)``. The conserved quantity is E_kin + E_pot."""

    timestep_fs: float
    fixcm: bool | Literal["auto"] = "auto"

    def __post_init__(self):
        IntegratorConfig(ensemble="nve", timestep_fs=self.timestep_fs)      # checks the values
        if not isinstance(self.fixcm, bool) and self.fixcm != "auto":
            raise InputError(f"NVE.fixcm must be True, False or 'auto', got {self.fixcm!r}")

    def to_config(self, *, seed: int) -> IntegratorConfig:
        _resolve_fixcm(self)
        return IntegratorConfig(ensemble="nve", timestep_fs=float(self.timestep_fs), seed=seed)


@dataclass(frozen=True, kw_only=True)
class Langevin:
    """Langevin NVT: ``lm.integrators.Langevin(timestep_fs=0.5, temperature_K=300, friction_per_fs=0.002)``."""

    timestep_fs: float = 0.5
    temperature_K: float
    friction_per_fs: float
    fixcm: bool | Literal["auto"] = "auto"

    def __post_init__(self):
        IntegratorConfig(timestep_fs=self.timestep_fs, temperature_K=self.temperature_K,
                         friction_per_fs=self.friction_per_fs)               # checks the values
        if not isinstance(self.fixcm, bool) and self.fixcm != "auto":
            raise InputError(f"Langevin.fixcm must be True, False or 'auto', got {self.fixcm!r}")

    def to_config(self, *, seed: int) -> IntegratorConfig:
        _resolve_fixcm(self)
        return IntegratorConfig(ensemble="nvt", timestep_fs=float(self.timestep_fs),
                                temperature_K=float(self.temperature_K),
                                friction_per_fs=float(self.friction_per_fs), seed=seed)
