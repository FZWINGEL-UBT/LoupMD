"""Config records: frozen dataclasses that are checked when they are made.

Names and defaults are FastMACE's wherever the feature exists (``Output``, the integrator's ``ensemble`` /
``scheme`` / ``timestep_fs`` / ``temperature_K``). Units at the API: fs, K, A, eV; friction per fs. A value outside
its range raises ``InputError`` naming the record and the field. ``from_mapping`` makes a record from a table of a
run file and refuses unknown keys by name.
"""
from __future__ import annotations

import dataclasses
import math
import os
import typing
from dataclasses import dataclass
from typing import Literal

from .errors import InputError

# The 3x accuracy rule of the gate (PLAN.md section 2). These are the rule, so they are not settings: a quantity
# passes when LoupMD's deviation from MACE float64 is at most GATE_FACTOR times MACE's own float32 deviation, or
# below the floor.
GATE_FACTOR = 3.0
GATE_ENERGY_FLOOR_MEV_PER_ATOM = 0.01
GATE_FORCE_FLOOR_EV_PER_A = 1e-5
GATE_STRESS_FLOOR_EV_PER_A3 = 1e-7

# Integrator schemes of each ensemble, the first being its default. BUILT are those the engine has.
SCHEMES = {"nve": ("nve",), "nvt": ("langevin", "csvr", "nose_hoover", "berendsen"),
           "npt": ("baoab", "mtk", "berendsen")}
BUILT = {"nve": ("nve",), "nvt": ("langevin",), "npt": ()}


def _name(record, field):
    return f"{type(record).__name__}.{field}"


def _check_choices(record):
    hints = typing.get_type_hints(type(record))
    for field in dataclasses.fields(record):
        hint, value = hints[field.name], getattr(record, field.name)
        if typing.get_origin(hint) is Literal:
            choices = typing.get_args(hint)
            if not isinstance(value, str) or value not in choices:
                raise InputError(f"{_name(record, field.name)}: {value!r} is not one of {choices}")


def _is_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _positive(record, *fields):
    for field in fields:
        value = getattr(record, field)
        if not _is_number(value) or value <= 0:
            raise InputError(f"{_name(record, field)} must be positive and finite, got {value!r}")


def _not_negative(record, *fields):
    for field in fields:
        value = getattr(record, field)
        if not _is_number(value) or value < 0:
            raise InputError(f"{_name(record, field)} must be finite and not negative, got {value!r}")


def _count(record, field, optional=False, what="an int of at least 1", unit=""):
    value = getattr(record, field)
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InputError(f"{_name(record, field)} must be {what}{' or None' if optional else ''}{unit}, got {value!r}")


# ----------------------------------------------------------------------------------------------- integrator

@dataclass(frozen=True)
class IntegratorConfig:
    """What the engine integrates with. ``scheme=None`` takes the ensemble's default (``nve``; ``langevin`` for
    NVT). A thermostat's temperature and friction have no defaults: physics is not chosen silently."""

    ensemble: Literal["nve", "nvt", "npt"] = "nvt"
    scheme: str | None = None
    timestep_fs: float = 0.5
    temperature_K: float | None = None
    friction_per_fs: float | None = None       # Langevin
    seed: int = 20261001

    def __post_init__(self):
        _check_choices(self)
        scheme = self.scheme if self.scheme is not None else (SCHEMES[self.ensemble][0])
        if scheme not in SCHEMES[self.ensemble]:
            raise InputError(f"{_name(self, 'scheme')}: scheme {scheme!r} is not a scheme of ensemble "
                             f"{self.ensemble!r}", next_step=f"Its schemes: {', '.join(SCHEMES[self.ensemble])}.")
        if scheme not in BUILT[self.ensemble]:
            built = ", ".join(name for names in BUILT.values() for name in names)
            raise InputError(f"{_name(self, 'scheme')}: {self.ensemble} with scheme {scheme!r} is not in this version",
                             next_step=f"Built schemes: {built}.")
        object.__setattr__(self, "scheme", scheme)
        _positive(self, "timestep_fs")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 64:
            raise InputError(f"{_name(self, 'seed')} must be an int from 0 to 2**64 - 1, got {self.seed!r}")
        if scheme == "nve":
            for field in ("temperature_K", "friction_per_fs"):
                if getattr(self, field) is not None:
                    raise InputError(f"{_name(self, field)}: NVE has no thermostat, got {getattr(self, field)!r}")
        elif scheme == "langevin":
            if self.temperature_K is None or self.friction_per_fs is None:
                raise InputError("IntegratorConfig: Langevin needs temperature_K and friction_per_fs",
                                 why="neither has a default")
            _not_negative(self, "temperature_K")
            _positive(self, "friction_per_fs")

    def to_struct(self, thermo_every: int = 0, temperature_K=None, rate_K_per_step: float = 0.0):
        """The ``LoupIntegrator`` of this record for ``loup_run``; ``thermo_every`` is the output's cadence. A
        schedule passes the call's starting target and its change per step (``temperature_K``,
        ``rate_K_per_step``): step i of the call runs at ``temperature_K + i * rate_K_per_step``."""
        from . import _core

        if self.scheme == "nve":
            return _core.integrator(_core.LOUP_NVE, self.timestep_fs, thermo_every=int(thermo_every))
        temperature = self.temperature_K if temperature_K is None else temperature_K
        return _core.integrator(_core.LOUP_LANGEVIN, self.timestep_fs, temp_k=float(temperature),
                                temp_rate_k_per_step=float(rate_K_per_step),
                                friction_per_fs=float(self.friction_per_fs), seed=self.seed,
                                thermo_every=int(thermo_every))


# --------------------------------------------------------------------------------------------------- output

@dataclass(frozen=True)
class Output:
    """Output directory and cadences in MD steps (FastMACE's record and defaults). ``trajectory_every`` and
    ``restart_every`` may be ``None`` (no trajectory, no restart files); thermo rows are always written.
    ``directory`` is kept as given, as a string."""

    directory: str
    trajectory_every: int | None = 20
    thermo_every: int = 20
    restart_every: int | None = 10_000

    def __post_init__(self):
        directory = os.fspath(self.directory) if isinstance(self.directory, os.PathLike) else self.directory
        if not isinstance(directory, str) or not directory:
            raise InputError(f"Output.directory must be a non-empty path, got {self.directory!r}")
        object.__setattr__(self, "directory", directory)
        _count(self, "trajectory_every", optional=True, what="a positive int", unit=" (steps)")
        _count(self, "thermo_every", what="a positive int", unit=" (steps)")
        _count(self, "restart_every", optional=True, what="a positive int", unit=" (steps)")


# ------------------------------------------------------------------------------------------------- the gate

@dataclass(frozen=True)
class GateConfig:
    """How the accuracy gate samples the user's structure and where its references run. Whether it runs is
    ``check=``; the rule itself is fixed (``GATE_FACTOR`` and the floors above)."""

    rattle_A: float = 0.05                    # standard deviation of the displacement of each copy
    copies: int = 2                           # rattled copies gated
    strain: float = 0.02                      # two strained copies more for NPT
    reference_device: Literal["auto", "cuda", "cpu"] = "auto"   # auto: the GPU, then the CPU when memory runs out
    cpu_float64_time_limit_s: float = 300.0   # the CPU is tried only if its estimated time is within this; 0: never

    def __post_init__(self):
        _check_choices(self)
        _positive(self, "rattle_A", "strain")
        _count(self, "copies")
        _not_negative(self, "cpu_float64_time_limit_s")


# ------------------------------------------------------------------------------------------------- capacity

@dataclass(frozen=True)
class CapacityConfig:
    """How a system that may not fit the GPU whole is run (``capacity=``; the tiled modes are Stage 5)."""

    mode: Literal["auto", "whole", "tiled", "wavefront"] = "auto"
    memory_fraction: float = 0.9              # of the free GPU memory the plan may use
    receivers: int | None = None              # receiving atoms per tile; None: from the memory plan
    state: Literal["auto", "device", "host"] = "auto"           # where the layer states of a tiled run live

    def __post_init__(self):
        _check_choices(self)
        if not _is_number(self.memory_fraction) or not 0.0 < self.memory_fraction <= 1.0:
            raise InputError(f"CapacityConfig.memory_fraction must be above 0 and at most 1, got "
                             f"{self.memory_fraction!r}")
        _count(self, "receivers", optional=True)

    @property
    def exec_mode(self) -> int:
        """The ``LOUP_EXEC_*`` constant of ``mode``."""
        from . import _core

        return {"auto": _core.LOUP_EXEC_AUTO, "whole": _core.LOUP_EXEC_WHOLE, "tiled": _core.LOUP_EXEC_TILED,
                "wavefront": _core.LOUP_EXEC_WAVEFRONT}[self.mode]


# ---------------------------------------------------------------------------------------------- from_mapping

def from_mapping(record_class, mapping, section=None):
    """``record_class(**mapping)`` for a table of a run file. Unknown keys are refused by name, all at once, with
    the keys the record has; a required key that is missing is named too."""
    where = f"[{section}]" if section else record_class.__name__
    if not isinstance(mapping, typing.Mapping):
        raise InputError(f"{where} is not a table of keys and values, got {type(mapping).__name__}")
    fields = dataclasses.fields(record_class)
    names = [f.name for f in fields]
    unknown = [key for key in mapping if key not in names]
    if unknown:
        listing = ", ".join(repr(key) for key in unknown)
        raise InputError(f"unknown key{'s' if len(unknown) > 1 else ''} {listing} in {where}",
                         next_step=f"Keys of {where}: {', '.join(names)}.")
    required = [f.name for f in fields
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
    missing = [name for name in required if name not in mapping]
    if missing:
        raise InputError(f"{where} lacks {', '.join(missing)}")
    return record_class(**mapping)
