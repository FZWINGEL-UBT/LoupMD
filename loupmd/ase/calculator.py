"""``LoupCalculator`` and ``lm.calc``: LoupMD as an ASE calculator.

ASE calls a calculator once per step and moves the atoms itself, so this front end runs one force evaluation per
call, with its copies between host and device: it is for relaxations, phonons, NEB and for checking LoupMD against
other codes. MD at full speed needs LoupMD to own the loop (``LoupEngine.run``).

The calculator is a thin layer over a ``ForceField`` (``loupmd.forcefield``): the model is resolved and the trust
rule applied when the calculator is made; the bundle and the engine are made at the first calculation, for the
structure's elements. The engine is kept while the atomic numbers stay; a change of their order or count sets the
system again on the same engine; an element outside the loaded bundle gets a new bundle and engine.
"""
from __future__ import annotations

from ase.calculators.calculator import Calculator, all_changes

from ..errors import InputError
from ..forcefield import ForceField, load


class LoupCalculator(Calculator):
    """Energy, free energy (the same number), forces and Voigt stress in ASE units, for cells periodic in all
    three directions. See ``loupmd.calc`` for the arguments."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, force_field: ForceField, *, skin: float = 0.5):
        super().__init__()
        self.force_field, self.skin = force_field, float(skin)

    @property
    def engine(self):
        """The ``LoupEngine`` of the last structure (``None`` before the first calculation)."""
        return self.force_field.engine

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        from ase.stress import full_3x3_to_voigt_6_stress

        Calculator.calculate(self, atoms, properties, system_changes)
        with_stress = "stress" in properties
        energy, forces, stress = self.force_field.evaluate(self.atoms, stress=with_stress, skin=self.skin)
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}
        if with_stress:
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress)


def calc(model, *, head=None, checkpoint=None, sha256=None, trusted=False, download=False, charge=None,
         spin_multiplicity=None, device=None, skin=0.5, engine_factory=None, ensure_bundle=None, probe=None,
         environ=None):
    """``atoms.calc = lm.calc("omat0-medium")``: an ASE calculator for ``model``.

    ``model`` is what ``lm.load`` takes (a label, a checkpoint path or a URL, with ``head``, ``checkpoint``,
    ``sha256`` / ``trusted``, ``download``, ``charge``, ``spin_multiplicity`` and ``device`` as there), or a
    ``ForceField`` that ``lm.load`` returned (then without those). ``skin`` is the neighbour-list skin in A.

    The stress is computed only when it is asked for.
    """
    if isinstance(model, ForceField):
        given = {"head": head, "checkpoint": checkpoint, "sha256": sha256, "charge": charge,
                 "spin_multiplicity": spin_multiplicity, "device": device}
        named = [name for name, value in given.items() if value is not None] + [
            name for name, value in (("trusted", trusted), ("download", download)) if value]
        if named:
            raise InputError(f"the model is already loaded, so {', '.join(named)} cannot be given here",
                             next_step="Pass them to lm.load instead.")
        return LoupCalculator(model, skin=skin)
    return LoupCalculator(load(model, head, checkpoint=checkpoint, sha256=sha256, trusted=trusted, download=download,
                               charge=charge, spin_multiplicity=spin_multiplicity, device=device or 0,
                               engine_factory=engine_factory, ensure_bundle=ensure_bundle, probe=probe,
                               environ=environ), skin=skin)
