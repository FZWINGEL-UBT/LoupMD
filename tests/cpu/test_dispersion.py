"""The dispersion record, its presets and refusals (plan task S6-D03, ``docs/design/dispersion.md``). Needs neither a
GPU nor the library; the term's numbers are tests/gpu/test_d3.py's."""
from __future__ import annotations

import warnings

import pytest
from ase.build import bulk
from fake_core import SpringEngine

import loupmd
from loupmd import dispersion as d, runfile
from loupmd.cli import main
from loupmd.errors import InputError, PlanRefused


@pytest.fixture
def ff(tmp_path):
    checkpoint = tmp_path / "some.model"
    checkpoint.write_bytes(b"never unpickled here")
    SpringEngine.made, SpringEngine.poison_at = [], None
    return loupmd.load(str(checkpoint), trusted=True, engine_factory=SpringEngine,
                       ensure_bundle=lambda path, numbers, **kw: path.with_name("bundle.loup"), probe=lambda p: None)


# ------------------------------------------------------------------------------------------- presets and records

@pytest.mark.parametrize("name", list(d.PRESETS))
def test_every_preset_expands_to_a_complete_record(name):
    record = d.resolve(name, functional="PBE")
    assert record.preset == name and record.method == "d3" and record.functional == "pbe"
    assert set(record.parameters) == set(d._PARAMETER_NAMES[record.damping])
    assert all(isinstance(v, float) for v in record.parameters.values())
    entry = d.entry(record)
    assert entry["parameter_source"] and entry["long_range_chosen_by"]
    assert d.parse(entry) == record                       # an entry reads back as the same record


def test_the_preset_values_are_the_documented_ones():
    assert d.preset("d3bj-mace").long_range == "none"
    atm = d.preset("d3bj-atm")
    assert atm.three_body and atm.three_body_cutoff_bohr == 20.0
    ref = d.preset("d3-reference")
    assert (ref.pair_cutoff_bohr, ref.cn_cutoff_bohr, ref.three_body_cutoff_bohr) == (60.0, 40.0, 40.0)
    assert d.resolve("d3bj", functional="pbe").parameters == {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407}
    assert d.resolve("d3bj", functional="r2scan").parameters["a2"] == 5.7308


def test_equal_records_give_equal_entries():
    by_name = d.resolve("d3bj", functional="pbe")
    by_hand = d.resolve(loupmd.Dispersion(damping="bj"), functional="pbe")
    by_table = d.resolve({"damping": "bj", "pair_cutoff_bohr": 40}, functional="pbe")
    assert by_name == by_hand == by_table
    assert d.entry(by_name) == d.entry(by_hand) == d.entry(by_table)
    assert d.entry(by_name)["long_range_chosen_by"] == "automatic, at setup"


def test_an_explicit_long_range_is_reported_as_the_users_choice():
    record = d.resolve(loupmd.Dispersion(long_range="split", mesh_spacing_A=0.5), functional="pbe")
    assert d.entry(record)["long_range_chosen_by"] == "the user"
    assert "chosen by the user" in d.describe(record)


def test_raw_parameters_are_named_as_given_by_the_user():
    record = d.resolve(loupmd.Dispersion(parameters={"s6": 1, "s8": 0.5, "a1": 0.4, "a2": 5}), functional=None)
    assert d.entry(record)["parameter_source"] == "given by the user"
    zero = loupmd.Dispersion(damping="zero", parameters={"s6": 1, "s8": 0.7, "rs6": 1.2})
    assert zero.parameters == {"s6": 1.0, "s8": 0.7, "rs6": 1.2, "rs8": 1.0, "alpha": 14.0}


@pytest.mark.parametrize("fields,message", [
    ({"method": "d4"}, "D4 is not in this version"),
    ({"damping": "becke"}, "damping='becke' is not one of"),
    ({"pair_cutoff_bohr": 0}, "pair_cutoff_bohr must be a positive number"),
    ({"cn_cutoff_bohr": True}, "cn_cutoff_bohr must be a positive number"),
    ({"three_body": True, "three_body_cutoff_bohr": 50}, "longer than pair_cutoff_bohr"),
    ({"three_body": 1}, "three_body must be True or False"),
    ({"smoothing": 45}, "smoothing must be 'none' or a switch width"),
    ({"long_range": "ewald"}, "long_range='ewald' is not one of"),
    ({"long_range": "mesh", "damping": "zero"}, "derived for Becke-Johnson damping only"),
    ({"long_range": "mesh", "three_body": True}, "the mesh sum has no three-body term"),
    ({"long_range": "none", "mesh_spacing_A": 0.5}, "only 'split' and 'mesh' use a mesh"),
    ({"long_range": "split", "mesh_spacing_A": -1}, "mesh_spacing_A must be a positive number"),
    ({"update_every": 2}, "stale forces above the accuracy contract"),
    ({"update_every": 0}, "update_every must be a positive int"),
    ({"auto_methods": ("fmm",)}, "each must be one of"),
    ({"auto_methods": ("tail",), "long_range": "none"}, "restricts the automatic choice only"),
    ({"parameters": {"s6": 1, "s8": 1, "a1": 0.4}}, "needs parameters ['a2']"),
    ({"parameters": {"s6": 1, "s8": 1, "a1": 0.4, "a2": 5, "rs6": 1}}, "are not parameters of damping 'bj'"),
    ({"parameters": {"s6": 1, "s8": "x", "a1": 0.4, "a2": 5}}, "parameter s8 must be a number"),
    ({"parameters": [1, 2]}, "parameters must be a table"),
    ({"functional": 3}, "functional must be a name"),
])
def test_unknown_or_inconsistent_combinations_are_refused(fields, message):
    with pytest.raises(InputError) as caught:
        loupmd.Dispersion(**fields)
    assert message in str(caught.value)


@pytest.mark.parametrize("value,message", [
    ("d3xyz", "dispersion='d3xyz' is not a preset"),
    (3, "dispersion= takes None, a preset name"),
    ({"damping": "bj", "cutoff": 40}, "unknown keys ['cutoff']"),
])
def test_parse_refuses_what_is_not_a_record(value, message):
    with pytest.raises(InputError) as caught:
        d.parse(value)
    assert message in str(caught.value)


def test_auto_methods_are_kept_in_a_fixed_order():
    assert loupmd.Dispersion(auto_methods=["mesh", "none", "mesh"]).auto_methods == ("none", "mesh")


# ----------------------------------------------------------------------------------- refusals against the model

@pytest.mark.parametrize("functional,family,why", [
    ("wB97M-D3BJ", "off", "contains a D3(BJ) correction already"),           # MACE-OFF
    ("wB97M-V", "omol", "VV10 nonlocal correlation"),                        # OMOL-0, MH-1's omol / spice heads
    (None, "polar", "MACE-POLAR-1 models long-range interactions"),
])
def test_models_with_dispersion_in_their_data_are_refused_with_the_reason(functional, family, why):
    with pytest.raises(PlanRefused) as caught:
        d.resolve("d3bj", functional=functional, family=family, model="that model")
    text = str(caught.value)
    assert why in text and "count dispersion twice" in text and "omat_pbe" in text


def test_a_damping_without_a_parameter_set_is_refused():
    with pytest.raises(PlanRefused) as caught:
        d.resolve("d3zero", functional="r2scan")
    assert "no 'zero' damping parameters for the functional 'r2scan'" in str(caught.value)
    assert "r2scan has parameter sets for bj" in str(caught.value)
    with pytest.raises(PlanRefused, match="LoupMD has no parameter set for b3lyp"):
        d.resolve("d3bj", functional="b3lyp")


def test_an_unknown_functional_needs_a_functional_or_parameters():
    with pytest.raises(PlanRefused, match="training functional is not known"):
        d.resolve("d3bj", functional=None)
    assert d.resolve(loupmd.Dispersion(functional="PBE"), functional=None).parameters["a1"] == 0.4289


def test_a_functional_other_than_the_models_warns():
    with pytest.warns(UserWarning, match="the parameter set of 'r2scan' on m, which was trained with 'PBE'"):
        record = d.resolve(loupmd.Dispersion(functional="r2scan"), functional="PBE", model="m")
    assert record.functional == "r2scan"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d.resolve(loupmd.Dispersion(functional="pbe"), functional="PBE")


def test_none_is_off():
    assert d.parse(None) is None and d.resolve(None, functional="pbe") is None


# ------------------------------------------------------------------------------------------------ the front ends

def test_lm_md_installs_the_term_and_records_it(ff, tmp_path):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    ff.functional, ff.family = "PBE", "omat"
    result = loupmd.md(atoms, model=ff, temperature_K=300, steps=10, seed=1, dispersion="d3bj",
                       output=str(tmp_path / "run"))
    engine = result.simulation._engine
    assert engine.dispersion == d.resolve("d3bj", functional="pbe")
    entry = result.metadata["dispersion"]
    assert entry["long_range_used"] == "none" and entry["long_range_chosen_by"] == "automatic, at setup"
    assert entry["parameter_source"].startswith("S. Grimme, S. Ehrlich, L. Goerigk")
    assert "dispersion: D3(bj) pbe [s6 1, s8 0.7875, a1 0.4289, a2 4.4407], pairs to 40 bohr" in result.summary
    assert result.metadata["arguments"]["dispersion"] == "d3bj"
    import json

    record = json.loads((tmp_path / "run" / "simulation.json").read_text())
    assert d.parse(record["dispersion"]) == engine.dispersion           # a resume installs the same term
    resumed = loupmd.Simulation.resume(tmp_path / "run", force_field=ff)
    assert resumed.dispersion == engine.dispersion


def test_the_three_body_preset_runs_and_is_reported(ff, tmp_path):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    ff.functional, ff.family = "PBE", "omat"
    result = loupmd.md(atoms, model=ff, temperature_K=300, steps=10, seed=1, dispersion="d3bj-atm",
                       output=str(tmp_path / "run"))
    assert result.simulation._engine.dispersion.three_body
    assert result.metadata["dispersion"]["three_body_cutoff_bohr"] == 20.0
    assert "three-body term, long range: none" in result.summary


def test_the_switch_must_be_shorter_than_the_three_body_cutoff():
    with pytest.raises(InputError, match="must be shorter than three_body_cutoff_bohr"):
        loupmd.Dispersion(three_body=True, three_body_cutoff_bohr=20.0, smoothing=25.0)
    text = d.describe(d.resolve(loupmd.Dispersion(three_body=True, three_body_cutoff_bohr=20.0, smoothing=4.0),
                                functional="pbe"))
    assert "three-body term to 20 bohr, switched off over the last 4 bohr" in text


@pytest.mark.parametrize("value,message", [
    (loupmd.Dispersion(long_range="mesh"), "long_range='mesh' is not built"),
    (loupmd.Dispersion(auto_methods=("tail",)), "none of them is built"),
])
def test_what_is_not_built_is_refused_before_the_run(ff, value, message):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    ff.functional, ff.family = "PBE", "omat"
    with pytest.raises(PlanRefused, match=message):
        loupmd.md(atoms, model=ff, temperature_K=300, steps=10, dispersion=value)


def test_models_with_dispersion_in_their_data_are_refused_by_lm_md(ff):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    ff.functional, ff.family = "wB97M-V", "omol"
    with pytest.raises(PlanRefused, match="VV10"):
        loupmd.md(atoms, model=ff, temperature_K=300, steps=10, dispersion="d3bj")


def test_the_run_file_takes_a_preset_or_a_table(tmp_path):
    for value, text in (("d3bj-atm", 'dispersion = "d3bj-atm"'),
                        ({"damping": "zero", "pair_cutoff_bohr": 50.0, "functional": None},
                         'dispersion = {damping = "zero", pair_cutoff_bohr = 50.0}')):
        written = runfile.to_toml({"structure": "salt.cif", "model": "m", "dispersion": value})
        assert text in written
        path = tmp_path / "run.toml"
        path.write_text(written)
        arguments = runfile.load(path)
        _, options = runfile.md_options(arguments)
        assert d.parse(options["dispersion"]) == d.parse(value)


def test_dispersion_list_shows_the_presets_methods_and_limits(capsys):
    assert main(["dispersion", "list"]) == 0
    out = capsys.readouterr().out
    for name in d.PRESETS:
        assert name in out
    assert "wrong for slabs, pores, interfaces" in out and "Becke-Johnson damping only" in out
    assert "doi:10.1002/jcc.21759" in out and "[not built yet]" in out and "MACE-POLAR-1" in out
    plain = next(line for line in out.splitlines() if line.startswith("  none "))
    assert "the plain real-space sum" in plain and "[not built yet]" not in plain


# ----------------------------------------------------------------------------- loupmd validate --dispersion (S6-D10)

def test_validate_dispersion_resolves_the_preset_against_the_label_and_reports(tmp_path, monkeypatch, capsys):
    from ase.io import write

    from loupmd.validate import dispersion as check

    seen = {}

    def fake_check(structures, record, **kw):
        seen["record"], seen["names"] = record, [n for n, _ in structures]
        return {"schema": check.SCHEMA, "dispersion": d.entry(record), "description": d.describe(record),
                "reference": "simple-dftd3 x", "bounds": dict(check.BOUNDS), "passed": False,
                "rows": [{"structure": "salt", "atoms": 8, "energy_eV": -1.0, "reference_energy_eV": -1.0,
                          "deviation": {"energy_eV_atom": 2e-6, "force_eV_A": 0.0, "stress_eV_A3": 0.0},
                          "passed": False}]}

    monkeypatch.setattr(check, "check", fake_check)
    path = tmp_path / "salt.cif"
    write(path, bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    code = main(["validate", "--dispersion", "d3bj-atm", "--model", "omat0-medium", "--structure", str(path),
                 "--json", str(tmp_path / "out.json")])
    out = capsys.readouterr().out
    assert code == 1 and "FAIL" in out and "dispersion check FAILED" in out
    assert seen["record"] == d.resolve("d3bj-atm", functional="pbe") and seen["names"] == ["salt"]
    import json

    assert json.loads((tmp_path / "out.json").read_text())["schema"] == "loupmd.dispersion-check/1"


def test_validate_needs_a_model_or_a_dispersion_preset(capsys):
    assert main(["validate"]) == 1
    assert "needs --model" in capsys.readouterr().err


def test_the_dispersion_check_refuses_the_switch():
    from loupmd.validate import dispersion as check

    with pytest.raises(PlanRefused, match="simple-dftd3 has no switch"):
        check.checkable(d.resolve(loupmd.Dispersion(smoothing=4.0), functional="pbe"))
