"""``LoupCalculator`` and ``lm.calc``: LoupMD as an ASE calculator.

ASE calls a calculator once per step and moves the atoms itself, so this front end runs one force evaluation per
call, with its copies between host and device: it is for relaxations, phonons, NEB and for checking LoupMD against
other codes. MD at full speed needs LoupMD to own the loop (``LoupEngine.run``).

The model is resolved and the trust rule applied when the calculator is made. The bundle and the engine are made
at the first calculation, for the structure's elements. The engine is kept while the atomic numbers stay; a change
of their order or count sets the system again on the same engine; an element outside the loaded bundle gets a new
bundle and engine.
"""
from __future__ import annotations

from ase.calculators.calculator import Calculator, all_changes

from .. import models
from ..errors import PlanRefused


class LoupCalculator(Calculator):
    """Energy, free energy (the same number), forces and Voigt stress in ASE units, for cells periodic in all
    three directions. See ``loupmd.calc`` for the arguments."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model, *, head=None, checkpoint=None, sha256=None, trusted=False, download=False, charge=None,
                 spin_multiplicity=None, device=0, skin=0.5, engine_factory=None, ensure_bundle=None, environ=None):
        super().__init__()
        self.resolved, self.model_sha256, _ = models.trusted_checkpoint(
            model, head, checkpoint=checkpoint, sha256=sha256, trusted=trusted, download=download, environ=environ)
        entry = self.resolved.label
        self.head = head if head is not None else (entry.head if entry is not None else None)
        self.charge, self.spin_multiplicity = charge, spin_multiplicity
        self.device, self.skin = int(device), float(skin)
        self.engine = None                 # the LoupEngine, once a structure was calculated
        self.bundle = None                 # its bundle file
        self._elements = frozenset()       # the elements of that bundle
        self._numbers = None               # the atomic numbers the engine's system was set with
        self._engine_factory, self._ensure_bundle, self._environ = engine_factory, ensure_bundle, environ

    def _prepare(self, atoms):
        if not all(atoms.pbc):
            raise PlanRefused(
                f"the structure has pbc={tuple(bool(p) for p in atoms.pbc)}; LoupMD needs a cell periodic in all "
                "three directions",
                next_step="Put a molecule or a slab in a box with vacuum (at least the model's cutoff on each open "
                          "side) and set pbc=True.")
        numbers = tuple(int(z) for z in atoms.numbers)
        if numbers == self._numbers:
            return
        if not set(numbers) <= self._elements:
            ensure_bundle = self._ensure_bundle
            if ensure_bundle is None:
                from .. import bundles

                ensure_bundle = bundles.ensure_bundle
            factory = self._engine_factory
            if factory is None:
                from .. import _core

                factory = _core.LoupEngine
            # Elements seen before stay in the bundle, so going back to an earlier structure exports nothing.
            elements = sorted(self._elements | set(numbers))
            bundle = ensure_bundle(self.resolved.path, elements, head=self.head, charge=self.charge,
                                   spin_multiplicity=self.spin_multiplicity, environ=self._environ)
            self.engine = self._numbers = None                      # free the old engine before the next is made
            self.engine, self.bundle, self._elements = factory(bundle, device_id=self.device), bundle, frozenset(elements)
        self._numbers = None
        self.engine.set_system(atoms.numbers, atoms.get_masses(), skin=self.skin)
        self._numbers = numbers

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        from ase.stress import full_3x3_to_voigt_6_stress

        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms
        self._prepare(atoms)
        with_stress = "stress" in properties
        energy, forces, stress = self.engine.evaluate(atoms.positions, atoms.cell.array, compute_stress=with_stress)
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}
        if with_stress:
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress)


def calc(model, *, head=None, checkpoint=None, sha256=None, trusted=False, download=False, charge=None,
         spin_multiplicity=None, device=0, skin=0.5, engine_factory=None, ensure_bundle=None, environ=None):
    """``atoms.calc = lm.calc("omat0-medium")``: an ASE calculator for ``model``.

    - ``model``: a label (``loupmd models`` lists them), a checkpoint path or a URL; ``head`` for a multi-head
      model (there is no default head);
    - ``checkpoint``: the file of a label, when it is not in ``$LOUPMD_MODELS`` or the cache; ``download=True``
      fetches a label's file when it is missing;
    - ``sha256`` or ``trusted=True``: needed for a file that is not in the label table, because a checkpoint is a
      pickle and runs code when it is exported;
    - ``charge``, ``spin_multiplicity``: total charge and spin of the cell for models that take them (OMOL-0);
      they are part of the bundle;
    - ``device``: the GPU index; ``skin``: the neighbour-list skin in A.

    The stress is computed only when it is asked for. ``engine_factory`` and ``ensure_bundle`` replace the engine
    and the exporter (tests).
    """
    return LoupCalculator(model, head=head, checkpoint=checkpoint, sha256=sha256, trusted=trusted, download=download,
                          charge=charge, spin_multiplicity=spin_multiplicity, device=device, skin=skin,
                          engine_factory=engine_factory, ensure_bundle=ensure_bundle, environ=environ)
