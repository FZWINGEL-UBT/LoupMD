"""``lm.integrators``: the integrator records a ``Simulation`` takes as ``integrator=``.

Names and arguments are FastMACE's (``fm.integrators``). Units: fs, K, friction per fs. A thermostat's temperature
and friction have no defaults: physics is not chosen silently. Each record maps onto an ``IntegratorConfig``
(``to_config``), which is what the engine runs. Built so far: ``NVE``, ``Langevin``, ``CSVR``,
``NoseHooverChain`` and ``Berendsen`` at constant volume; the constant-pressure schemes of FastMACE arrive with
the engine's integrators (PLAN.md Stage 4).

``fixcm`` keeps the centre of mass at rest, as in FastMACE: ``"auto"`` (the default) is on, and off when the
structure has fixed atoms, which pin the frame already; ``True`` with fixed atoms is refused. Langevin's noise then
loses its mass-weighted mean every step (FastMACE's ``"mass_weighted"`` projection); the total momentum of NVE,
CSVR and the Nose-Hoover chain is removed at the start of every run call. The temperature counts three degrees
of freedom fewer, and so do the thermostats that count them (CSVR, the Nose-Hoover chain).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import IntegratorConfig
from .errors import InputError, PlanRefused

__all__ = ["Berendsen", "CSVR", "Langevin", "NVE", "NoseHooverChain"]


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


@dataclass(frozen=True, kw_only=True)
class CSVR:
    """Canonical sampling through velocity rescaling (Bussi, Donadio, Parrinello), as ASE's ``Bussi``:
    ``lm.integrators.CSVR(timestep_fs=1.0, temperature_K=300, tau_fs=100)``.

    ``tau_fs`` is ASE's ``taut``. The conserved quantity is E_kin + E_pot minus the energy the thermostat has
    transferred. The velocities are rescaled, so the structure must move at the start (as ASE's ``Bussi``)."""

    timestep_fs: float
    temperature_K: float
    tau_fs: float
    fixcm: bool | Literal["auto"] = "auto"

    def __post_init__(self):
        IntegratorConfig(scheme="csvr", timestep_fs=self.timestep_fs, temperature_K=self.temperature_K,
                         tau_fs=self.tau_fs)                                 # checks the values
        if not isinstance(self.fixcm, bool) and self.fixcm != "auto":
            raise InputError(f"CSVR.fixcm must be True, False or 'auto', got {self.fixcm!r}")

    def to_config(self, *, seed: int, fixed_atoms: bool = False) -> IntegratorConfig:
        return IntegratorConfig(ensemble="nvt", scheme="csvr", timestep_fs=float(self.timestep_fs),
                                temperature_K=float(self.temperature_K), tau_fs=float(self.tau_fs), seed=seed,
                                fixcm=resolve_fixcm(self.fixcm, fixed_atoms))


@dataclass(frozen=True, kw_only=True)
class NoseHooverChain:
    """Nose-Hoover chains, as ASE's ``NoseHooverChainNVT``:
    ``lm.integrators.NoseHooverChain(timestep_fs=1.0, temperature_K=300, tdamp_fs=100)``.

    ``tdamp_fs``, ``tchain`` and ``tloop`` are ASE's ``tdamp``, ``tchain`` and ``tloop``. The thermostat masses
    are made for ``temperature_K`` and stay when a schedule moves the target, as ASE's and FastMACE's. The
    conserved quantity is E_kin + E_pot plus the chain's energy. The degrees of freedom of the reported
    temperature stand where ASE counts 3 N: 3 N - 3 with the centre of mass at rest, 3 (N - fixed) with fixed
    atoms."""

    timestep_fs: float
    temperature_K: float
    tdamp_fs: float
    tchain: int = 3
    tloop: int = 1
    fixcm: bool | Literal["auto"] = "auto"

    def __post_init__(self):
        if not isinstance(self.temperature_K, bool) and isinstance(self.temperature_K, (int, float))                 and self.temperature_K == 0:
            raise InputError(f"NoseHooverChain.temperature_K must be above 0 K (the thermostat masses are made "
                             f"for it), got {self.temperature_K!r}",
                             next_step="For a run that starts at 0 K give the temperature the masses should be "
                                       "made for here and the targets as a schedule.")
        IntegratorConfig(scheme="nose_hoover", timestep_fs=self.timestep_fs, temperature_K=self.temperature_K,
                         tau_fs=self.tdamp_fs, tchain=self.tchain, tloop=self.tloop)       # checks the values
        if not isinstance(self.fixcm, bool) and self.fixcm != "auto":
            raise InputError(f"NoseHooverChain.fixcm must be True, False or 'auto', got {self.fixcm!r}")

    def to_config(self, *, seed: int, fixed_atoms: bool = False) -> IntegratorConfig:
        return IntegratorConfig(ensemble="nvt", scheme="nose_hoover", timestep_fs=float(self.timestep_fs),
                                temperature_K=float(self.temperature_K), tau_fs=float(self.tdamp_fs),
                                tchain=self.tchain, tloop=self.tloop, seed=seed,
                                fixcm=resolve_fixcm(self.fixcm, fixed_atoms))


@dataclass(frozen=True, kw_only=True)
class Berendsen:
    """Berendsen thermostat, as ASE's ``NVTBerendsen``:
    ``lm.integrators.Berendsen(timestep_fs=1.0, temperature_K=300, taut_fs=100)``; with ``pressure_GPa``,
    ``taup_fs`` and ``compressibility_per_GPa`` also Berendsen's barostat, as ASE's ``NPTBerendsen``
    (experimental):
    ``lm.integrators.Berendsen(timestep_fs=1.0, temperature_K=300, taut_fs=100, pressure_GPa=0.0,
    taup_fs=1000, compressibility_per_GPa=0.006)``.

    Not canonical: for equilibration only. ``taut_fs`` and ``taup_fs`` are ASE's ``taut`` and ``taup``,
    ``compressibility_per_GPa`` the material's compressibility (ASE's ``compressibility_au`` times
    ``units.GPa``). Without ``mask`` the cell keeps its shape; ``mask=(1, 1, 0)`` lets the first two cell
    vectors follow their own stress and leaves the third alone (ASE's ``Inhomogeneous_NPTBerendsen``). The
    barostat makes one force call per step: the first kick of a step uses the forces of the cell before its
    scaling, where ASE's classes evaluate again. Not with fixed atoms."""

    timestep_fs: float
    temperature_K: float
    taut_fs: float
    pressure_GPa: float | None = None
    taup_fs: float | None = None
    compressibility_per_GPa: float | None = None
    mask: tuple | None = None
    fixcm: bool | Literal["auto"] = "auto"

    def __post_init__(self):
        self._config(seed=0, fixcm=False)                                    # checks the values
        if not isinstance(self.fixcm, bool) and self.fixcm != "auto":
            raise InputError(f"Berendsen.fixcm must be True, False or 'auto', got {self.fixcm!r}")

    @property
    def barostat(self) -> bool:
        return any(getattr(self, name) is not None
                   for name in ("pressure_GPa", "taup_fs", "compressibility_per_GPa", "mask"))

    def _config(self, *, seed, fixcm) -> IntegratorConfig:
        if not self.barostat:
            return IntegratorConfig(ensemble="nvt", scheme="berendsen", timestep_fs=self.timestep_fs,
                                    temperature_K=self.temperature_K, tau_fs=self.taut_fs, seed=seed, fixcm=fixcm)
        return IntegratorConfig(ensemble="npt", scheme="berendsen", timestep_fs=self.timestep_fs,
                                temperature_K=self.temperature_K, tau_fs=self.taut_fs,
                                pressure_GPa=self.pressure_GPa, taup_fs=self.taup_fs,
                                compressibility_per_GPa=self.compressibility_per_GPa, mask=self.mask, seed=seed,
                                fixcm=fixcm)

    def to_config(self, *, seed: int, fixed_atoms: bool = False) -> IntegratorConfig:
        if self.barostat and fixed_atoms:
            raise PlanRefused("Berendsen's barostat with fixed atoms",
                              why="the cell's scaling would move the fixed atoms",
                              next_step="Leave the barostat's fields out, or free the atoms.")
        return self._config(seed=seed, fixcm=resolve_fixcm(self.fixcm, fixed_atoms))


RECORDS = {"NVE": NVE, "Langevin": Langevin, "CSVR": CSVR, "NoseHooverChain": NoseHooverChain,
           "Berendsen": Berendsen}


def to_data(record) -> dict:
    """An integrator record as a table: ``{"type": "CSVR", "timestep_fs": ..., ...}`` (a run file's
    ``[integrator]``)."""
    import dataclasses

    if type(record) not in RECORDS.values():
        raise InputError(f"integrator must be an lm.integrators record, got {type(record).__name__}")
    return {"type": type(record).__name__, **dataclasses.asdict(record)}


def from_data(data, *, timestep_fs=None, temperature_K=None):
    """The record of a table (``to_data``). ``timestep_fs`` and ``temperature_K`` fill those fields where the
    table leaves them out (a run file states them once, outside the table). Unknown keys are refused by name."""
    import dataclasses
    from collections.abc import Mapping

    if not isinstance(data, Mapping):
        raise InputError(f"[integrator] is not a table of keys and values, got {type(data).__name__}")
    data = dict(data)
    kind = data.pop("type", None)
    if kind not in RECORDS:
        raise InputError(f"[integrator] type {kind!r} is not one of {', '.join(RECORDS)}",
                         why="the other integrators of FastMACE are not in this version")
    names = [field.name for field in dataclasses.fields(RECORDS[kind])]
    unknown = [key for key in data if key not in names]
    if unknown:
        raise InputError(f"unknown key{'s' if len(unknown) > 1 else ''} {', '.join(repr(k) for k in unknown)} in "
                         f"[integrator] of type {kind}", next_step=f"Its keys: type, {', '.join(names)}.")
    for name, value in (("timestep_fs", timestep_fs), ("temperature_K", temperature_K)):
        if name in names and name not in data and value is not None:
            data[name] = value
    missing = [field.name for field in dataclasses.fields(RECORDS[kind])
               if field.default is dataclasses.MISSING and field.name not in data]
    if missing:
        raise InputError(f"[integrator] of type {kind} lacks {', '.join(missing)}",
                         why="a thermostat's temperature, friction and time constant have no defaults")
    return RECORDS[kind](**data)
