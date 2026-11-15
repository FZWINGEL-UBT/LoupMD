"""Drop-in optimisers with ASE's arguments: ``from loupmd.ase.optimize import LBFGS, FIRE``.

``opt = LBFGS(atoms, model="omat0-medium"); opt.run(fmax=0.05)`` relaxes ``atoms`` in place, as ASE's optimisers
do, and returns whether it converged. ``cell_filter=lm.optimize.FrechetCell(...)`` relaxes the cell too (in place
of wrapping the atoms in ASE's ``FrechetCellFilter``). ASE's ``restart``, ``logfile`` and ``trajectory`` are not
taken; ``attach`` calls a function after every step.
"""
from __future__ import annotations

from .. import optimize as _optimize
from ..errors import InputError
from .md import force_field_of


class _Optimizer:
    def __init__(self, atoms, record, model, head, options):
        from ase import Atoms

        if not isinstance(atoms, Atoms):
            raise InputError(f"{type(self).__name__} takes an ase.Atoms, got {type(atoms).__name__}",
                             next_step="For a cell relaxation pass cell_filter=lm.optimize.FrechetCell() instead of "
                                       "wrapping the atoms in a filter.")
        self.atoms, self.record = atoms, record
        self.force_field = force_field_of(model, head, options, type(self).__name__)
        self.observers = []
        self.nsteps = 0
        self.result = None

    def attach(self, function, interval=1, *args, **kwargs):
        self.observers.append((function, interval, args, kwargs))

    def get_number_of_steps(self):
        return self.nsteps

    def run(self, fmax=0.05, steps=100_000_000):
        """Relax until the largest force is below ``fmax`` (eV/A) or ``steps`` steps were taken; ``atoms`` is
        updated in place. Returns whether it converged."""
        def observe(step, positions, energy, largest):
            self.nsteps = step
            if self.observers:
                self.atoms.set_positions(positions, apply_constraint=False)
                for function, interval, args, kwargs in self.observers:
                    if step % interval == 0:
                        function(*args, **kwargs)

        self.result = _optimize.relax(self.force_field, self.atoms, self.record, fmax=fmax, steps=steps,
                                      callback=observe)
        relaxed = self.result.atoms
        if self.record.cell_filter is not None:
            self.atoms.set_cell(relaxed.cell, scale_atoms=False)
        self.atoms.set_positions(relaxed.positions, apply_constraint=False)
        self.atoms.calc = relaxed.calc
        self.atoms.calc.atoms = self.atoms.copy()
        self.nsteps = self.result.steps
        return self.result.converged


class LBFGS(_Optimizer):
    """``ase.optimize.LBFGS`` (without line search) on LoupMD's forces."""

    def __init__(self, atoms, maxstep=None, memory=100, damping=1.0, alpha=70.0, use_line_search=False, *,
                 cell_filter=None, model=None, head=None, **options):
        if use_line_search:
            raise InputError("use_line_search=True is not offered", next_step="Use LBFGS without it, or FIRE.")
        record = _optimize.LBFGS(maxstep_A=0.2 if maxstep is None else maxstep, memory=memory, damping=damping,
                                 alpha=alpha, cell_filter=cell_filter)
        super().__init__(atoms, record, model, head, options)


class FIRE(_Optimizer):
    """``ase.optimize.FIRE`` on LoupMD's forces."""

    def __init__(self, atoms, dt=0.1, maxstep=None, dtmax=1.0, Nmin=5, finc=1.1, fdec=0.5, astart=0.1, fa=0.99,
                 a=0.1, downhill_check=False, *, cell_filter=None, model=None, head=None, **options):
        record = _optimize.FIRE(dt=dt, maxstep_A=0.2 if maxstep is None else maxstep, dtmax=dtmax, Nmin=Nmin,
                                finc=finc, fdec=fdec, astart=astart, fa=fa, a=a, downhill_check=downhill_check,
                                cell_filter=cell_filter)
        super().__init__(atoms, record, model, head, options)
