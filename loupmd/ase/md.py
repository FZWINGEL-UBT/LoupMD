"""Drop-in MD classes with ASE's arguments and units: ``from loupmd.ase.md import Langevin, VelocityVerlet``.

``dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=300, friction=0.01 / units.fs, fixcm=False, model="omat0-medium")``
then ``dyn.run(1000)``, as with ``ase.md``. The engine owns the loop: ``run`` hands it the steps between two
observer calls in one piece, and the ``atoms`` are brought up to date (positions, momenta, a calculator holding the
energy and forces) before every observer call and at the end of ``run``.

Beyond ASE's arguments: ``model=`` (a label, path or ``lm.load`` result; required), ``head=``, ``seed=``,
``output=`` (a directory or ``lm.Output``), ``check=`` and what ``lm.load`` takes. ASE's ``trajectory``, ``logfile``
and ``loginterval`` are not taken: use ``output=``.

Built: ``Langevin`` and ``VelocityVerlet``. ASE's ``fixcm=True`` (its default for Langevin) is refused until the
engine removes the centre-of-mass motion; pass ``fixcm=False``. The other classes of FastMACE's ``fastmace.ase``
are refused by name until their integrators are built.
"""
from __future__ import annotations

from math import gcd

from ..errors import InputError, PlanRefused
from ..forcefield import ForceField, load

_NOT_BUILT = ("LangevinBAOAB", "Bussi", "NoseHooverChainNVT", "IsotropicMTKNPT", "MTKNPT", "NVTBerendsen",
              "NPTBerendsen", "Inhomogeneous_NPTBerendsen")
_LOAD_OPTIONS = ("checkpoint", "sha256", "trusted", "download", "charge", "spin_multiplicity", "device")


def __getattr__(name):
    if name in _NOT_BUILT:
        raise PlanRefused(f"loupmd.ase.md.{name} is not in this version",
                          why="its integrator is not built in the engine yet (PLAN.md Stage 4)",
                          next_step="Use Langevin or VelocityVerlet, or ASE's class with lm.calc as the calculator.")
    raise AttributeError(f"module 'loupmd.ase.md' has no attribute {name!r}")


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

    def __init__(self, atoms, integrator, model, head, seed, output, check, options):
        from ..config import Output
        from ..simulation import Simulation

        if output is not None and not isinstance(output, Output):
            output = Output(output)
        self.atoms = atoms
        self.simulation = Simulation(atoms, force_field_of(model, head, options, type(self).__name__),
                                     integrator=integrator, output=output, seed=seed, check=check)
        self.observers = []
        self.nsteps = 0

    def attach(self, function, interval=1, *args, **kwargs):
        """Call ``function(*args, **kwargs)`` every ``interval`` steps (and at step 0), with ``atoms`` up to date."""
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise InputError(f"interval must be a positive int, got {interval!r}")
        if hasattr(function, "write") and not callable(function):
            function = function.write
        self.observers.append((function, interval, args, kwargs))

    def get_number_of_steps(self):
        return self.nsteps

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
        self.simulation.close()


class Langevin(_Dynamics):
    """``ase.md.langevin.Langevin`` on LoupMD's engine: ``timestep`` and ``friction`` in ASE units, ``temperature_K``
    in K."""

    def __init__(self, atoms, timestep, temperature_K=None, friction=None, fixcm=True, *, model=None, head=None,
                 seed=None, output=None, check="skip", **options):
        from ase import units

        from .. import integrators

        if temperature_K is None or friction is None:
            raise InputError("Langevin needs temperature_K and friction (in inverse ASE time units, for example "
                             "0.01 / units.fs)")
        record = integrators.Langevin(timestep_fs=timestep / units.fs, temperature_K=temperature_K,
                                      friction_per_fs=friction * units.fs, fixcm=fixcm)
        super().__init__(atoms, record, model, head, seed, output, check, options)


class VelocityVerlet(_Dynamics):
    """``ase.md.verlet.VelocityVerlet`` on LoupMD's engine: ``timestep`` in ASE units."""

    def __init__(self, atoms, timestep, *, model=None, head=None, seed=None, output=None, check="skip", **options):
        from ase import units

        from .. import integrators

        super().__init__(atoms, integrators.NVE(timestep_fs=timestep / units.fs, fixcm=False), model, head, seed,
                         output, check, options)
