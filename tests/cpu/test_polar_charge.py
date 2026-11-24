"""MACE-POLAR-1's total charge and spin multiplicity in the bundle's metadata (no GPU, no model file)."""
import pytest

polar = pytest.importorskip("loupmd.export.polar", reason="the export extra (torch, e3nn)")


def test_defaults_are_the_neutral_singlet():
    meta = polar.charge_spin_meta(None, None)
    assert meta == {"LR.charge": 0, "LR.spin_multiplicity": 1, "charge": 0, "spin_multiplicity": 1}


def test_charge_and_spin_are_written_as_given():
    meta = polar.charge_spin_meta(-2, 3)
    assert (meta["LR.charge"], meta["LR.spin_multiplicity"]) == (-2, 3)
    assert (meta["charge"], meta["spin_multiplicity"]) == (-2, 3)


def test_a_spin_multiplicity_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        polar.charge_spin_meta(0, 0)
