"""Drop-in MD classes with ASE's arguments and units: ``from loupmd.ase.md import Langevin, VelocityVerlet, Bussi``.

A script written for ``ase.md`` runs after changing its import line and naming the model:

    dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=300, friction=0.01 / units.fs, model="omat0-medium",
                   trajectory="md.traj", logfile="md.log", loginterval=10)
    dyn.attach(report, interval=100)
    dyn.run(1000)

The engine owns the loop: ``run`` hands it the steps between two observer calls in one piece, and the ``atoms`` are
brought up to date (positions, momenta, a calculator holding the energy and forces) before every observer call and
at the end of ``run``. ASE's ``trajectory``, ``logfile``, ``loginterval`` and ``append_trajectory`` mean what they
mean in ASE (an ASE trajectory file and ASE's ``MDLogger`` lines); they are observers, so the run stops at every
``loginterval`` steps for them. LoupMD's own output (``output=``, a directory or ``lm.Output``) is written without
stopping the run and is the faster choice for long runs.

Beyond ASE's arguments: ``model=`` (a label, path or ``lm.load`` result; required), ``head=``, ``seed=`` (the
thermostat's noise seed), ``output=``, ``check=`` and what ``lm.load`` takes. ASE's ``rng``, ``comm`` and ``master``
are refused when given: the noise is drawn on the GPU from ``seed``.

Built: ``Langevin``, ``VelocityVerlet`` and ``Bussi``. Langevin's ``fixcm=True`` (ASE's default) keeps the centre
of mass at rest with the mass-weighted projection of the noise, which is canonical for unequal masses too; ASE's
own projection differs from it for unequal masses. ``VelocityVerlet`` and ``Bussi`` remove no centre-of-mass
motion and count ASE's degrees of freedom, as ASE's classes. The other classes of FastMACE's ``fastmace.ase`` are
refused by name until their integrators are built.
"""
from __future__ import annotations

import os
from math import gcd

from ..errors import InputError, PlanRefused
from ..forcefield import ForceField, load

_NOT_BUILT = ("LangevinBAOAB", "NoseHooverChainNVT", "IsotropicMTKNPT", "MTKNPT", "NVTBerendsen", "NPTBerendsen",
              "Inhomogeneous_NPTBerendsen")
_LOAD_OPTIONS = ("checkpoint", "sha256", "trusted", "download", "charge", "spin_multiplicity", "device")
# ASE's arguments that have no meaning here: accepted at None, refused otherwise.
_ASE_REFUSED = {"rng": "the noise is drawn on the GPU: pass seed=", "comm": "LoupMD runs in one process",
                "master": "LoupMD runs in one process"}


def __getattr__(name):
    if name in _NOT_BUILT:
        raise PlanRefused(f"loupmd.ase.md.{name} is not in this version",
                          why="its integrator is not built in the engine yet (PLAN.md Stage 4)",
                          next_step="Use Langevin, Bussi or VelocityVerlet, or ASE's class with lm.calc as the "
                                    "calculator.")
    raise AttributeError(f"module 'loupmd.ase.md' has no attribute {name!r}")


def take_ase_arguments(options, what) -> dict:
    """Remove ASE's own optional arguments from ``options`` and return those that are set. ``rng``, ``comm`` and
    ``master`` are refused unless ``None``."""
    for name, why in _ASE_REFUSED.items():
        if options.pop(name, None) is not None:
            raise InputError(f"{what}: {name}= is not taken", why=why)
    return {name: options.pop(name) for name in ("trajectory", "logfile", "loginterval", "append_trajectory",
                                                 "restart") if name in options}


def open_trajectory(trajectory, atoms, append=False):
    """ASE's ``trajectory=``: ``(write function, object to close or None)``. A path opens an ASE trajectory file
    of ``atoms``; anything else is taken as a trajectory object and its ``write`` is called."""
    if isinstance(trajectory, (str, os.PathLike)):
        from ase.io.trajectory import Trajectory

        writer = Trajectory(os.fspath(trajectory), mode="a" if append else "w", atoms=atoms)
        return writer.write, writer
    if not hasattr(trajectory, "write"):
        raise InputError(f"trajectory must be a path or a trajectory object, got {type(trajectory).__name__}")
    return trajectory.write, None


def force_field_of(model, head, options, what):
    """The ``ForceField`` of a drop-in's ``model=``."""
    unknown = sorted(set(options) - set(_LOAD_OPTIONS))
    if unknown:
        raise TypeError(f"{what} got unexpected keyword arguments {', '.join(unknown)}")
    if model is None:
        raise InputError(f"{what} needs model=: a label (`loupmd models`), a checkpoint path or an lm.load result")
    if isinstance(model, ForceField):
        if head is not None or options:
            raise InputError("the model is already loaded; head and the load options belong to lm.load")
        return model
    return load(model, head, **options)


class _Dynamics:
    """What the drop-ins share: observers, ``run`` in pieces, the ``atoms`` kept up to date."""

    def __init__(self, atoms, timestep, integrator, model, head, seed, output, check, options):
        from ..config import Output
        from ..simulation import Simulation

        what = type(self).__name__
        ase_arguments = take_ase_arguments(options, what)
        if ase_arguments.pop("restart", None) is not None:
            raise TypeError(f"{what} got an unexpected keyword argument restart")
        if output is not None and not isinstance(output, Output):
            output = Output(output)
        self.atoms = atoms
        self.dt = timestep                                   # ASE's attribute, in ASE time units
        self.simulation = Simulation(atoms, force_field_of(model, head, options, what),
                                     integrator=integrator, output=output, seed=seed, check=check, skin="auto")
        self.observers = []
        self.nsteps = 0
        self._closables = []
        interval = ase_arguments.get("loginterval", 1)
        if ase_arguments.get("trajectory") is not None:
            write, closable = open_trajectory(ase_arguments["trajectory"], atoms,
                                              ase_arguments.get("append_trajectory", False))
            self.attach(write, interval=interval)
            if closable is not None:
                self._closables.append(closable)
        if ase_arguments.get("logfile"):
            from ase.md.logger import MDLogger

            logger = MDLogger(dyn=self, atoms=atoms, logfile=ase_arguments["logfile"])
            self.attach(logger, interval=interval)
            self._closables.append(logger)

    def attach(self, function, interval=1, *args, **kwargs):
        """Call ``function(*args, **kwargs)`` every ``interval`` steps (and at step 0), with ``atoms`` up to date."""
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise InputError(f"interval must be a positive int, got {interval!r}")
        if hasattr(function, "write") and not callable(function):
            function = function.write
        self.observers.append((function, interval, args, kwargs))

    def get_number_of_steps(self):
        return self.nsteps

    def get_time(self):
        """The simulated time in ASE time units, as ASE's."""
        return self.nsteps * self.dt

    def _sync(self):
        current = self.simulation.atoms
        self.atoms.set_positions(current.positions, apply_constraint=False)
        self.atoms.set_momenta(current.get_momenta(), apply_constraint=False)
        self.atoms.calc = current.calc
        self.atoms.calc.atoms = self.atoms.copy()

    def _call_observers(self):
        for function, interval, args, kwargs in self.observers:
            if self.nsteps % interval == 0:
                function(*args, **kwargs)

    def run(self, steps):
        """``steps`` MD steps. Returns ``True``, as ASE's ``run`` does when it completes."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise InputError(f"steps must be a whole number of at least 0, got {steps!r}")
        if self.nsteps == 0 and self.observers:
            self.simulation.start()                      # ASE calls its observers before the first step, with the
            self._sync()                                 # starting structure's energy and forces
            self._call_observers()
        piece = 0
        for _, interval, _, _ in self.observers:
            piece = gcd(piece, interval)
        done = 0
        while done < steps:
            n = steps - done if not piece else min(steps - done, piece - self.nsteps % piece)
            self.simulation.run(steps=n)
            done += n
            self.nsteps += n
            if self.observers:
                self._sync()
                self._call_observers()
        if steps:
            self._sync()
        return True

    def close(self):
        """Close the files this object opened (``trajectory=``, ``logfile=``) and the simulation's output."""
        for closable in self._closables:
            closable.close()
        self._closables = []
        self.simulation.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Langevin(_Dynamics):
    """``ase.md.langevin.Langevin`` on LoupMD's engine: ``timestep`` and ``friction`` in ASE units, ``temperature_K``
    in K (or ASE's older ``temperature`` in eV)."""

    def __init__(self, atoms, timestep, temperature=None, friction=None, fixcm=True, *, temperature_K=None,
                 model=None, head=None, seed=None, output=None, check="skip", **options):
        from ase import units

        from .. import integrators

        if temperature is not None and temperature_K is not None:
            raise TypeError("Langevin takes temperature (in eV) or temperature_K, not both")
        if temperature is not None:
            temperature_K = temperature / units.kB
        if temperature_K is None or friction is None:
            raise InputError("Langevin needs temperature_K and friction (in inverse ASE time units, for example "
                             "0.01 / units.fs)")
        record = integrators.Langevin(timestep_fs=timestep / units.fs, temperature_K=temperature_K,
                                      friction_per_fs=friction * units.fs, fixcm=fixcm)
        super().__init__(atoms, timestep, record, model, head, seed, output, check, options)


class VelocityVerlet(_Dynamics):
    """``ase.md.verlet.VelocityVerlet`` on LoupMD's engine: ``timestep`` in ASE units."""

    def __init__(self, atoms, timestep, trajectory=None, logfile=None, loginterval=1, *, model=None, head=None,
                 seed=None, output=None, check="skip", **options):
        from ase import units

        from .. import integrators

        options.update(trajectory=trajectory, logfile=logfile, loginterval=loginterval)
        super().__init__(atoms, timestep, integrators.NVE(timestep_fs=timestep / units.fs, fixcm=False), model,
                         head, seed, output, check, options)


class Bussi(_Dynamics):
    """``ase.md.bussi.Bussi`` (canonical sampling through velocity rescaling) on LoupMD's engine: ``timestep`` and
    ``taut`` in ASE units, ``temperature_K`` in K. As ASE's class, it refuses a structure at rest and counts
    ASE's degrees of freedom."""

    def __init__(self, atoms, timestep, temperature_K, taut, rng=None, *, model=None, head=None, seed=None,
                 output=None, check="skip", **options):
        from ase import units

        from .. import integrators

        options["rng"] = rng
        record = integrators.CSVR(timestep_fs=timestep / units.fs, temperature_K=temperature_K,
                                  tau_fs=taut / units.fs, fixcm=False)
        super().__init__(atoms, timestep, record, model, head, seed, output, check, options)
        self.transferred_energy = 0.0                    # ASE's attribute; the engine keeps the running sum
