"""``lm.integrators``: the integrator records a ``Simulation`` takes as ``integrator=``.

Names and arguments are FastMACE's (``fm.integrators``). Units: fs, K, friction per fs. A thermostat's temperature
and friction have no defaults: physics is not chosen silently. Each record maps onto an ``IntegratorConfig``
(``to_config``), which is what the engine runs. Built so far: ``NVE`` and ``Langevin``; the other schemes of
FastMACE arrive with the engine's integrators (PLAN.md Stage 4).

``fixcm`` keeps the centre of mass at rest, as in FastMACE: ``"auto"`` (the default) is on, and off when the
structure has fixed atoms, which pin the frame already; ``True`` with fixed atoms is refused. Langevin's noise then
loses its mass-weighted mean every step (FastMACE's ``"mass_weighted"`` projection); NVE's total momentum is removed
at the start of every run call. The temperature counts three degrees of freedom fewer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import IntegratorConfig
from .errors import InputError, PlanRefused

__all__ = ["Langevin", "NVE"]


def resolve_fixcm(fixcm, fixed_atoms) -> bool:
    """A record's ``fixcm`` for a structure with (``fixed_atoms``) or without fixed atoms: ``"auto"`` is off with
    fixed atoms and on without; ``True`` with fixed atoms is refused."""
    if fixcm == "auto":
        return not fixed_atoms
    if fixcm and fixed_atoms:
        raise PlanRefused("fixcm=True with fixed atoms", why="the fixed atoms pin the frame already",
                          next_step="Leave fixcm at 'auto' (off with fixed atoms), or remove the FixAtoms constraint.")
    return bool(fixcm)


@dataclass(frozen=True, kw_only=True)
class NVE:
    """Velocity Verlet: ``lm.integrators.NVE(timestep_fs=1.0)``. The conserved quantity is E_kin + E_pot."""

    timestep_fs: float
    fixcm: bool | Literal["auto"] = "auto"

    def __post_init__(self):
        IntegratorConfig(ensemble="nve", timestep_fs=self.timestep_fs)      # checks the values
        if not isinstance(self.fixcm, bool) and self.fixcm != "auto":
            raise InputError(f"NVE.fixcm must be True, False or 'auto', got {self.fixcm!r}")

    def to_config(self, *, seed: int, fixed_atoms: bool = False) -> IntegratorConfig:
        return IntegratorConfig(ensemble="nve", timestep_fs=float(self.timestep_fs), seed=seed,
                                fixcm=resolve_fixcm(self.fixcm, fixed_atoms))


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

    def to_config(self, *, seed: int, fixed_atoms: bool = False) -> IntegratorConfig:
        return IntegratorConfig(ensemble="nvt", timestep_fs=float(self.timestep_fs),
                                temperature_K=float(self.temperature_K),
                                friction_per_fs=float(self.friction_per_fs), seed=seed,
                                fixcm=resolve_fixcm(self.fixcm, fixed_atoms))
