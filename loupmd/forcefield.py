"""``lm.load`` and ``ForceField``: a model resolved, trusted and ready to be exported for a structure's elements.

``load`` finds the checkpoint file and applies the trust rule; it reads nothing else and needs no torch. The bundle
for a structure is exported at the first evaluation (or simulation), for the structure's elements; a structure with
an element the model has no embedding for is refused before that. The engine is kept while the elements fit the
loaded bundle.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from . import models
from .errors import InputError, ModelElementError, PlanRefused


def probe_checkpoint(path):
    """What a checkpoint outside the label table holds (``loupmd._probe`` in a process of its own, which needs
    torch and mace-torch): a dict with ``elements``, ``heads`` and ``r_max``, or ``None`` when it cannot be read
    here. The caller has applied the trust rule."""
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("mace") is None:
        return None
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(Path(__file__).resolve().parents[1]),
                                                    env.get("PYTHONPATH")) if p)
    done = subprocess.run([sys.executable, "-m", "loupmd._probe", str(path)], env=env, capture_output=True, text=True)
    if done.returncode != 0:
        return None
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def _symbols(numbers):
    from ase.data import chemical_symbols

    return [chemical_symbols[z] for z in numbers]


def _element_text(numbers):
    """``H, C, N`` for a short list; ranges for a long one: ``H to Bi (Z 1-83) and Ac to Pu (Z 89-94)``."""
    numbers = sorted(numbers)
    if len(numbers) <= 20:
        return ", ".join(_symbols(numbers))
    runs, start = [], numbers[0]
    for previous, current in zip(numbers, numbers[1:] + [None]):
        if current != previous + 1:
            runs.append((start, previous))
            start = current
    names = [f"{_symbols([a])[0]} to {_symbols([b])[0]} (Z {a}-{b})" if a != b else _symbols([a])[0] for a, b in runs]
    return " and ".join(names) if len(names) <= 2 else ", ".join(names)


class ForceField:
    """A loaded model. Made by ``lm.load``; holds the engine once a structure was evaluated."""

    def __init__(self, resolved, digest, *, head=None, charge=None, spin_multiplicity=None, device=0,
                 engine_factory=None, ensure_bundle=None, probe=None, environ=None):
        entry = resolved.label
        known = entry.checkpoint if entry is not None else models.checkpoint_of_digest(digest)
        self.path, self.sha256 = resolved.path, digest
        self.label = entry.label if entry is not None else None
        self.head = head if head is not None else (entry.head if entry is not None else None)
        self.charge, self.spin_multiplicity, self.device = charge, spin_multiplicity, int(device)
        self.elements = known.elements if known is not None else None     # None: not known before a probe
        self.r_max_A = known.r_max_A if known is not None else None
        self.heads = known.heads if known is not None else None
        self.licence = known.licence if known is not None else None
        self.functional = entry.functional if entry is not None else None
        self.engine = None                 # the LoupEngine of the last structure
        self.last_bundle = None
        self.bundle = None                 # its bundle file
        self._bundle_elements = frozenset()
        self._system = None                # (atomic numbers, skin) the engine's system was set with
        self._probed = known is not None
        self._engine_factory, self._ensure_bundle, self._probe, self._environ = (engine_factory, ensure_bundle, probe,
                                                                                 environ)

    @property
    def name(self) -> str:
        return self.label or self.path.name

    def __repr__(self):
        head = f", head {self.head}" if self.head else ""
        return f"ForceField({self.name}: {self.path.name}, SHA-256 {self.sha256[:16]}...{head})"

    def _learn(self):
        """For a file outside the table: its elements and heads, read once (when torch is there to read them)."""
        if self._probed:
            return
        self._probed = True
        found = (self._probe or probe_checkpoint)(self.path)
        if not found:
            return
        self.elements = tuple(sorted(int(z) for z in found["elements"]))
        self.heads = tuple(found.get("heads") or ())
        self.r_max_A = found.get("r_max")
        if self.head is None and len(self.heads) > 1:
            raise InputError(f"{self.path.name} has several heads and no default",
                             next_step=f"Pass head= one of {', '.join(self.heads)}.")
        if self.head is not None and self.heads and self.head not in self.heads:
            raise InputError(f"{self.path.name} has no head {self.head!r}",
                             next_step=f"Its heads are {', '.join(self.heads)}.")

    def check_elements(self, atomic_numbers):
        """Raise ``ModelElementError`` if the model has no embedding for one of ``atomic_numbers``. Nothing is
        exported before this passed."""
        self._learn()
        if self.elements is None:
            return                                   # unknown here: the exporter reports it
        missing = sorted({int(z) for z in atomic_numbers} - set(self.elements))
        if missing:
            raise ModelElementError(
                f"the structure has {', '.join(_symbols(missing))}, which {self.name} has no embedding for",
                why=f"its elements are {_element_text(self.elements)}")

    def _check_periodic(self, atoms):
        if not all(atoms.pbc):
            raise PlanRefused(
                f"the structure has pbc={tuple(bool(p) for p in atoms.pbc)}; LoupMD needs a cell periodic in all "
                "three directions",
                next_step="Put a molecule or a slab in a box with vacuum (at least the model's cutoff on each open "
                          "side) and set pbc=True.")

    def new_engine(self, atoms, skin=0.5):
        """An engine of its own for ``atoms`` with the system set: what a ``Simulation`` runs on, so that
        evaluating other structures with this force field does not disturb the run."""
        self._check_periodic(atoms)
        numbers = sorted({int(z) for z in atoms.numbers})
        self.check_elements(numbers)
        ensure_bundle = self._ensure_bundle
        if ensure_bundle is None:
            from . import bundles

            ensure_bundle = bundles.ensure_bundle
        factory = self._engine_factory
        if factory is None:
            from . import _core

            factory = _core.LoupEngine
        bundle = ensure_bundle(self.path, numbers, head=self.head, charge=self.charge,
                               spin_multiplicity=self.spin_multiplicity, environ=self._environ)
        engine = factory(bundle, device_id=self.device)
        engine.set_system(atoms.numbers, atoms.get_masses(), skin=float(skin))
        self.last_bundle = bundle                  # of the engine made last, for the run's record
        return engine

    def engine_for(self, atoms, skin=0.5):
        """The engine with ``atoms``' system set: exports the bundle and creates the engine when the elements are
        not in the loaded bundle, and sets the system when the atomic numbers or the skin changed."""
        self._check_periodic(atoms)
        numbers = tuple(int(z) for z in atoms.numbers)
        system = (numbers, float(skin))
        if system == self._system:
            return self.engine
        if not set(numbers) <= self._bundle_elements:
            self.check_elements(numbers)
            ensure_bundle = self._ensure_bundle
            if ensure_bundle is None:
                from . import bundles

                ensure_bundle = bundles.ensure_bundle
            factory = self._engine_factory
            if factory is None:
                from . import _core

                factory = _core.LoupEngine
            # Elements seen before stay in the bundle, so going back to an earlier structure exports nothing.
            elements = sorted(self._bundle_elements | set(numbers))
            bundle = ensure_bundle(self.path, elements, head=self.head, charge=self.charge,
                                   spin_multiplicity=self.spin_multiplicity, environ=self._environ)
            self.engine = self._system = None                       # free the old engine before the next is made
            self.engine, self.bundle = factory(bundle, device_id=self.device), bundle
            self._bundle_elements = frozenset(elements)
        self._system = None
        self.engine.set_system(atoms.numbers, atoms.get_masses(), skin=float(skin))
        self._system = system
        return self.engine

    def evaluate(self, atoms, stress=False, skin=0.5):
        """``(energy in eV, forces in eV/A, stress)`` of ``atoms``; the stress (3x3, eV/A^3) is ``None`` unless
        ``stress=True``."""
        engine = self.engine_for(atoms, skin)
        return engine.evaluate(atoms.positions, atoms.cell.array, compute_stress=bool(stress))


def load(model, head=None, *, checkpoint=None, sha256=None, trusted=False, download=False, charge=None,
         spin_multiplicity=None, device=0, engine_factory=None, ensure_bundle=None, probe=None,
         environ=None) -> ForceField:
    """``ff = lm.load("omat0-medium")``: find the model's file, apply the trust rule and return its ``ForceField``.

    - ``model``: a label (``loupmd models`` lists them), a checkpoint path or a URL; ``head`` for a multi-head
      model (there is no default head);
    - ``checkpoint``: the file of a label, when it is not in ``$LOUPMD_MODELS`` or the cache; ``download=True``
      fetches a label's file when it is missing;
    - ``sha256`` or ``trusted=True``: needed for a file that is not in the label table, because a checkpoint is a
      pickle and runs code when it is exported;
    - ``charge``, ``spin_multiplicity``: total charge and spin of the cell for models that take them (OMOL-0);
    - ``device``: the GPU index.

    Nothing is exported and no engine is made here. ``engine_factory``, ``ensure_bundle`` and ``probe`` replace the
    engine, the exporter and the checkpoint reader (tests).
    """
    resolved, digest, _ = models.trusted_checkpoint(model, head, checkpoint=checkpoint, sha256=sha256,
                                                    trusted=trusted, download=download, environ=environ)
    return ForceField(resolved, digest, head=head, charge=charge, spin_multiplicity=spin_multiplicity, device=device,
                      engine_factory=engine_factory, ensure_bundle=ensure_bundle, probe=probe, environ=environ)
