"""The config records of loupmd.config (plan task S3-T01): validation, refused keys, the engine struct. No GPU."""
from __future__ import annotations

import dataclasses
import pickle

import pytest

from loupmd import _core, config
from loupmd.config import CapacityConfig, GateConfig, IntegratorConfig, Output
from loupmd.errors import InputError


def test_the_records_are_frozen_and_picklable():
    records = [IntegratorConfig(ensemble="nve", timestep_fs=1.0), Output("run"), GateConfig(), CapacityConfig()]
    for record in records:
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.seed = 3
        assert pickle.loads(pickle.dumps(record)) == record
        assert type(record).__name__ in repr(record)


# ----------------------------------------------------------------------------------------------- integrator

def test_langevin_is_the_default_scheme_of_nvt_and_needs_its_physics_stated():
    record = IntegratorConfig(temperature_K=300, friction_per_fs=0.01)
    assert (record.ensemble, record.scheme, record.timestep_fs, record.seed) == ("nvt", "langevin", 0.5, 20261001)
    for missing in ({"temperature_K": 300}, {"friction_per_fs": 0.01}, {}):
        with pytest.raises(InputError, match="needs temperature_K and friction_per_fs"):
            IntegratorConfig(**missing)


def test_nve_takes_no_thermostat_settings():
    assert IntegratorConfig(ensemble="nve", timestep_fs=1.0).scheme == "nve"
    with pytest.raises(InputError, match="IntegratorConfig.temperature_K: NVE has no thermostat"):
        IntegratorConfig(ensemble="nve", temperature_K=300)
    with pytest.raises(InputError, match="scheme 'langevin' is not a scheme of ensemble 'nve'"):
        IntegratorConfig(ensemble="nve", scheme="langevin")


@pytest.mark.parametrize("fields,message", [
    ({"timestep_fs": 0.0}, "IntegratorConfig.timestep_fs must be positive and finite, got 0.0"),
    ({"timestep_fs": float("nan")}, "IntegratorConfig.timestep_fs must be positive and finite"),
    ({"timestep_fs": "1"}, "IntegratorConfig.timestep_fs must be positive and finite, got '1'"),
    ({"temperature_K": -1.0}, "IntegratorConfig.temperature_K must be finite and not negative, got -1.0"),
    ({"friction_per_fs": 0.0}, "IntegratorConfig.friction_per_fs must be positive and finite, got 0.0"),
    ({"seed": -1}, "IntegratorConfig.seed must be an int from 0 to 2**64 - 1, got -1"),
    ({"seed": 1.5}, "IntegratorConfig.seed must be an int"),
    ({"ensemble": "nph"}, "IntegratorConfig.ensemble: 'nph' is not one of ('nve', 'nvt', 'npt')"),
])
def test_integrator_values_are_checked(fields, message):
    base = {"temperature_K": 300.0, "friction_per_fs": 0.01}
    base.update(fields)
    with pytest.raises(InputError) as caught:
        IntegratorConfig(**base)
    assert message in str(caught.value)


@pytest.mark.parametrize("fields,word", [
    ({"ensemble": "npt"}, "npt"), ({"scheme": "csvr"}, "csvr"), ({"scheme": "nose_hoover"}, "nose_hoover"),
])
def test_schemes_that_are_not_built_are_refused_naming_what_is(fields, word):
    with pytest.raises(InputError) as caught:
        IntegratorConfig(temperature_K=300.0, friction_per_fs=0.01, **fields)
    text = str(caught.value)
    assert word in text and "not in this version" in text and "nve, langevin" in text


def test_the_engine_struct_of_a_record():
    it = IntegratorConfig(temperature_K=350.0, friction_per_fs=0.02, timestep_fs=2.0, seed=7).to_struct(thermo_every=20)
    assert isinstance(it, _core.LoupIntegrator)
    assert (it.kind, it.dt_fs, it.temp_k, it.friction_per_fs, it.seed, it.thermo_every) == (
        _core.LOUP_LANGEVIN, 2.0, 350.0, 0.02, 7, 20)
    nve = IntegratorConfig(ensemble="nve", timestep_fs=1.0).to_struct()
    assert (nve.kind, nve.dt_fs, nve.temp_k, nve.friction_per_fs, nve.thermo_every) == (_core.LOUP_NVE, 1.0, 0.0, 0.0, 0)


# --------------------------------------------------------------------------------------------------- output

def test_output_has_fastmace_defaults_and_keeps_the_directory_as_a_string(tmp_path):
    out = Output(tmp_path / "run")
    assert out.directory == str(tmp_path / "run")
    assert (out.trajectory_every, out.thermo_every, out.restart_every) == (20, 20, 10_000)
    assert Output("run", trajectory_every=None, restart_every=None).trajectory_every is None


@pytest.mark.parametrize("fields,message", [
    ({"directory": ""}, "Output.directory must be a non-empty path, got ''"),
    ({"directory": 3}, "Output.directory must be a non-empty path, got 3"),
    ({"thermo_every": None}, "Output.thermo_every must be a positive int (steps), got None"),
    ({"thermo_every": 0}, "Output.thermo_every must be a positive int (steps), got 0"),
    ({"trajectory_every": 2.5}, "Output.trajectory_every must be a positive int or None (steps), got 2.5"),
    ({"restart_every": True}, "Output.restart_every must be a positive int or None (steps), got True"),
])
def test_output_values_are_checked(fields, message):
    base = {"directory": "run"}
    base.update(fields)
    with pytest.raises(InputError) as caught:
        Output(**base)
    assert message in str(caught.value)


# ------------------------------------------------------------------------------------------- gate, capacity

def test_gate_defaults_are_the_plan_s_and_the_rule_itself_is_not_a_setting():
    gate = GateConfig()
    assert (gate.rattle_A, gate.copies, gate.strain, gate.reference_device) == (0.05, 2, 0.02, "auto")
    assert (config.GATE_FACTOR, config.GATE_ENERGY_FLOOR_MEV_PER_ATOM, config.GATE_FORCE_FLOOR_EV_PER_A,
            config.GATE_STRESS_FLOOR_EV_PER_A3) == (3.0, 0.01, 1e-5, 1e-7)
    assert {f.name for f in dataclasses.fields(GateConfig)}.isdisjoint({"factor", "tolerance", "force_floor"})
    for fields, message in (({"rattle_A": 0.0}, "GateConfig.rattle_A must be positive and finite"),
                            ({"copies": 0}, "GateConfig.copies must be an int of at least 1, got 0"),
                            ({"strain": -0.1}, "GateConfig.strain must be positive and finite"),
                            ({"reference_device": "tpu"}, "GateConfig.reference_device: 'tpu' is not one of"),
                            ({"cpu_float64_time_limit_s": -1}, "GateConfig.cpu_float64_time_limit_s must be finite and "
                                                               "not negative")):
        with pytest.raises(InputError) as caught:
            GateConfig(**fields)
        assert message in str(caught.value)


def test_capacity_modes_and_limits():
    capacity = CapacityConfig()
    assert (capacity.mode, capacity.memory_fraction, capacity.receivers, capacity.state) == ("auto", 0.9, None, "auto")
    assert CapacityConfig(mode="tiled", receivers=20_000, state="host").exec_mode == _core.LOUP_EXEC_TILED
    assert [CapacityConfig(mode=m).exec_mode for m in ("auto", "whole", "tiled", "wavefront")] == [
        _core.LOUP_EXEC_AUTO, _core.LOUP_EXEC_WHOLE, _core.LOUP_EXEC_TILED, _core.LOUP_EXEC_WAVEFRONT]
    for fields, message in (({"mode": "sharded"}, "CapacityConfig.mode: 'sharded' is not one of"),
                            ({"memory_fraction": 0.0}, "CapacityConfig.memory_fraction must be above 0 and at most 1"),
                            ({"memory_fraction": 1.2}, "CapacityConfig.memory_fraction must be above 0 and at most 1"),
                            ({"receivers": 0}, "CapacityConfig.receivers must be an int of at least 1 or None, got 0"),
                            ({"state": "disk"}, "CapacityConfig.state: 'disk' is not one of")):
        with pytest.raises(InputError) as caught:
            CapacityConfig(**fields)
        assert message in str(caught.value)


# ---------------------------------------------------------------------------------------------- from_mapping

def test_a_mapping_becomes_a_record():
    record = config.from_mapping(IntegratorConfig, {"ensemble": "nvt", "temperature_K": 300, "friction_per_fs": 0.01},
                                 section="integrator")
    assert record == IntegratorConfig(temperature_K=300, friction_per_fs=0.01)
    assert config.from_mapping(Output, {"directory": "run", "restart_every": 500}).restart_every == 500


def test_unknown_keys_are_refused_by_name_all_at_once():
    with pytest.raises(InputError) as caught:
        config.from_mapping(IntegratorConfig, {"temperature_K": 300, "friction": 0.01, "timestep": 1.0},
                            section="integrator")
    text = str(caught.value)
    assert text.startswith("unknown keys 'friction', 'timestep' in [integrator]")
    assert "Next: Keys of [integrator]: ensemble, scheme, timestep_fs, temperature_K, friction_per_fs, seed." in text
    with pytest.raises(InputError, match=r"unknown key 'every' in Output"):
        config.from_mapping(Output, {"directory": "run", "every": 5})
    with pytest.raises(InputError, match="is not a table"):
        config.from_mapping(Output, ["run"], section="output")


def test_a_bad_value_in_a_mapping_names_its_section():
    with pytest.raises(InputError) as caught:
        config.from_mapping(Output, {"directory": "run", "thermo_every": 0}, section="output")
    assert "Output.thermo_every must be a positive int (steps), got 0" in str(caught.value)
    with pytest.raises(InputError, match=r"\[output\] lacks directory"):
        config.from_mapping(Output, {"thermo_every": 5}, section="output")
