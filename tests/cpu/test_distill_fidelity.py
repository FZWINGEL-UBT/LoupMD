"""Analytic checks of the label-only fidelity bar (never an MD acceptance test)."""
import json

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from loupgym.distill.fidelity import compare_round, main, score_frames


def frame(symbols="OH", *, name="a", split="fixed-student-test", sampler="student"):
    atoms = Atoms(symbols, cell=[10] * 3, pbc=True)
    atoms.info.update(frame_id=name, split=split, sampler=sampler, regime="target")
    atoms.info.update(teacher_sha256="a" * 64, teacher_head="omat_pbe",
                      prediction_sha256="b" * 64, prediction_head="default")
    model = "teacher" if sampler == "teacher" else "prediction"
    for suffix in ("sha256", "head"):
        atoms.info[f"sampler_{suffix}"] = atoms.info[f"{model}_{suffix}"]
    for prefix in ("REF", "STUDENT"):
        atoms.info[prefix + "_energy"] = -2.0 * len(atoms)
        atoms.info[prefix + "_stress"] = np.zeros((3, 3))
        atoms.arrays[prefix + "_forces"] = np.ones((len(atoms), 3))
    return atoms


def test_pool_components_not_frames_and_stress_in_ase_order():
    a = frame()
    b = frame("O", name="b")
    a.arrays["REF_forces"][:] = [[3, 0, 0], [0, 4, 0]]
    a.arrays["STUDENT_forces"][:] = [[3, 0, 0], [0, 1, 4]]  # norm of H error is 5
    b.arrays["REF_forces"][:] = b.arrays["STUDENT_forces"][:] = [[0, 0, 0]]
    a.info["STUDENT_energy"] += 0.004  # 2 meV/atom, second frame exact
    a.info["STUDENT_stress"] = np.array([[1, 6, 5], [6, 2, 4], [5, 4, 3]])
    report = score_frames(iter([a, b]))
    force = report["forces"]
    assert force["overall"]["rmse"] == pytest.approx(5 / 3)
    assert force["overall"]["teacher_rms"] == pytest.approx(5 / 3)
    assert force["overall"]["relative_rmse"] == pytest.approx(1)
    assert force["per_species"]["H"]["relative_rmse"] == pytest.approx(1.25)
    assert force["per_species"]["O"]["relative_rmse"] == 0
    tail = force["atom_error_norm"]
    assert tail["p99_9"] == pytest.approx(4.99)  # linear quantile of [0, 0, 5]
    assert tail["max"] == pytest.approx(5)
    assert tail["limit"] == pytest.approx(5 * 0.02 * 5 / 3)
    assert force["unit"] == "eV/Angstrom"
    assert report["energy"]["rmse_meV_per_atom"] == pytest.approx(np.sqrt(2))
    stress = report["stress"]
    assert stress["components"] == ["xx", "yy", "zz", "yz", "xz", "xy"]
    assert stress["rmse_by_component"] == pytest.approx(np.arange(1, 7) / np.sqrt(2))
    assert stress["rmse"] == pytest.approx(np.sqrt(91 / 12))
    assert stress["max_abs"] == 6
    assert report["frames"][0]["force_rmse_eV_per_A"] == pytest.approx(5 / np.sqrt(6))
    assert report["student_visited_label_bar"] == "fail"
    assert report["md_fidelity"] == "not_evaluated"


def test_tail_can_fail_with_good_component_rmse_and_energy():
    a = frame("H1000")
    a.arrays["STUDENT_forces"][:10, 1] += 0.15
    report = score_frames([a])
    assert report["forces"]["per_species"]["H"]["passed"]
    assert not report["forces"]["atom_error_norm"]["passed"]
    assert not report["label_metrics_passed"]
    assert score_frames([a], tail_multiplier=10)["student_visited_label_bar"] == "pass"


def test_species_can_fail_despite_good_global_force_and_tail():
    a = frame("OH100")
    a.arrays["REF_forces"][0] = [0.1, 0, 0]
    a.arrays["STUDENT_forces"][0] = [0.105, 0, 0]
    r = score_frames([a])
    assert r["forces"]["overall"]["passed"] and r["forces"]["atom_error_norm"]["passed"]
    assert not r["forces"]["per_species"]["O"]["passed"]
    assert r["student_visited_label_bar"] == "fail"


def test_energy_is_gated_stress_is_reported_only():
    a = frame()
    a.info["STUDENT_stress"][1, 1] = -20
    assert score_frames([a])["student_visited_label_bar"] == "pass"
    a.info["STUDENT_energy"] += 0.003
    assert score_frames([a])["student_visited_label_bar"] == "fail"


def test_molecular_force_and_torque_use_unwrapped_mass_weighted_arms():
    a = frame("HHe")
    a.set_cell([[10, 0, 0], [3, 9, 0], [2, -1, 8]])
    a.positions[:] = [[9.8, 0, 0], [0.2, 0, 0]]
    a.set_masses([1, 3])
    a.arrays["mol_id"] = np.array([7, 7])
    a.arrays["REF_forces"][:] = [[0, 1, 0], [0, 1, 0]]
    a.arrays["STUDENT_forces"][:] = [[0, 1, 0], [0, 1.4, 0]]
    for shift in (np.zeros(3), a.cell[1]):
        a.positions[0] += shift
        m = score_frames([a])["molecular"]
        assert m["net_force"]["teacher_rms"] == pytest.approx(2 / np.sqrt(3))
        assert m["net_force"]["rmse"] == pytest.approx(0.4 / np.sqrt(3))
        assert m["torque"]["teacher_rms"] == pytest.approx(0.2 / np.sqrt(3))
        assert m["torque"]["rmse"] == pytest.approx(0.04 / np.sqrt(3))
        assert not m["torque"]["passed"]


def test_molecular_terms_gate_independently_of_dominant_bond_forces():
    a = frame("H2")
    a.positions[1, 0] = 1
    a.arrays["mol_id"] = np.array([0, 0])
    a.arrays["REF_forces"][:] = [[100, 0, 0], [-100, 0, 0]]
    a.arrays["STUDENT_forces"][:] = [[100, 0, 0], [-100, 0.1, 0]]
    r = score_frames([a])
    assert r["forces"]["overall"]["passed"] and r["forces"]["atom_error_norm"]["passed"]
    assert r["molecular"]["torque"]["relative_rmse"] is None  # no division by a floor
    assert not r["molecular"]["net_force"]["passed"]
    assert r["student_visited_label_bar"] == "fail"
    json.dumps(r, allow_nan=False)


def test_invalid_or_incomplete_molecular_data_is_not_silently_dropped():
    a = frame()
    with pytest.raises(ValueError, match="mol_id"):
        score_frames([a], molecule_key="mol_id")
    for ids in ([1.2, 1.2], [-1, -1]):
        a.arrays["mol_id"] = np.array(ids)
        with pytest.raises(ValueError, match="mol_id"):
            score_frames([a])
    a.arrays["mol_id"] = np.array([1, 1])
    with pytest.raises(ValueError, match="molecul|mol_id"):
        score_frames([frame(name="without"), a])
    a.positions[0] = [5, 0, 0]  # exactly on the half-box boundary: two different torques
    with pytest.raises(ValueError, match="ambiguous|compact"):
        score_frames([a])


def test_teacher_or_unidentified_frames_do_not_certify_student_label_bar():
    a = frame(sampler="teacher", split="teacher-test")
    report = score_frames([a])
    assert report["label_metrics_passed"] and report["student_visited_label_bar"] == "not_evaluated"
    assert report["md_fidelity"] == "not_evaluated"
    a.info["split"] = "fixed-student-test"
    with pytest.raises(ValueError, match="sampler|teacher"):
        score_frames([a])
    for key in ("split", "sampler", "frame_id"):
        del a.info[key]
    assert score_frames([a])["student_visited_label_bar"] == "not_evaluated"


def test_reject_invalid_labels_and_duplicate_frames():
    a = frame()
    with pytest.raises(ValueError, match="duplicate"):
        score_frames([a, a])
    del a.info["REF_stress"]
    with pytest.raises(ValueError, match="REF_stress"):
        score_frames([a])
    a.info["REF_stress"] = np.zeros((3, 3))
    a.arrays["STUDENT_forces"][0, 0] = np.nan
    with pytest.raises(ValueError, match="STUDENT_forces"):
        score_frames([a])


def test_round_comparison_uses_absolute_rmse_and_distinct_test_identities():
    a = frame("H", name="student")
    b = frame("H", name="teacher", split="teacher-test", sampler="teacher")
    a.arrays["STUDENT_forces"] += 0.04
    b.arrays["REF_forces"] *= 100
    b.arrays["STUDENT_forces"][:] = b.arrays["REF_forces"] + 0.02
    s, t = score_frames([a]), score_frames([b])
    r = compare_round(s, t, 2)
    assert r["student_to_teacher_rmse_ratio"] == pytest.approx(2)
    assert r["exceeds_1_5x"] and r["round"] == 2
    b.info["frame_id"] = "student"
    with pytest.raises(ValueError, match="overlap"):
        compare_round(s, score_frames([b]), 2)
    a.info["split"] = "harvest"
    with pytest.raises(ValueError, match="fixed-student-test"):
        compare_round(score_frames([a]), t, 2)


def test_cli_joint_extxyz_outputs_rankable_frame_errors(tmp_path):
    easy = frame(name="s10", split="harvest")
    hard = frame(name="s30", split="harvest")
    hard.arrays["STUDENT_forces"][:, 2] += 3
    xyz, output = tmp_path / "scored.extxyz", tmp_path / "report.json"
    write(xyz, [easy, hard])
    main([str(xyz), "--output", str(output)])
    report = json.loads(output.read_text())
    ranked = sorted(report["frames"], key=lambda f: -f["force_rmse_eV_per_A"])
    assert [(f["frame_id"], f["index"]) for f in ranked] == [("s30", 1), ("s10", 0)]
    assert report["student_visited_label_bar"] == "not_evaluated"  # harvest isn't held out


def test_only_target_frames_can_contribute_to_a_passing_label_bar():
    a, b = frame(), frame(name="hot")
    b.info["regime"] = "hot"
    assert score_frames([a, b])["student_visited_label_bar"] == "not_evaluated"
    target = score_frames([b, a], regime="target")
    assert target["student_visited_label_bar"] == "pass"
    assert target["frames"][0]["index"] == 1
    with pytest.raises(ValueError, match="no frames"):
        score_frames([b], regime="target")


def test_round_comparison_rejects_different_candidate_models():
    a, b = frame(), frame(name="b", sampler="teacher", split="teacher-test")
    a.info["prediction_sha256"] = a.info["sampler_sha256"] = "1" * 64
    b.info["prediction_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="prediction_sha256"):
        compare_round(score_frames([a]), score_frames([b]), 1)


@pytest.mark.parametrize("key,value", [("sampler_sha256", "c" * 64), ("sampler_head", "other"),
                                      ("teacher_sha256", None), ("teacher_head", ""),
                                      ("prediction_sha256", None), ("sampler_head", None)])
def test_fixed_set_metrics_remain_available_without_evidence_of_own_rollout(key, value):
    a, t = frame(), frame(name="t", sampler="teacher", split="teacher-test")
    if value is None:
        del a.info[key]
    else:
        a.info[key] = value
    report = score_frames([a])
    assert report["label_metrics_passed"]
    assert report["student_visited_label_bar"] == "not_evaluated"
    assert report["label_bar_ineligible_reason"]
    with pytest.raises(ValueError):
        compare_round(report, score_frames([t]), 2)


def test_round_requires_target_conditions_and_own_teacher_trajectory():
    a, t = frame(), frame(name="t", sampler="teacher", split="teacher-test")
    a.info["regime"] = t.info["regime"] = "hot"
    with pytest.raises(ValueError, match="target"):
        compare_round(score_frames([a]), score_frames([t]), 2)
    a.info["regime"] = t.info["regime"] = "target"
    t.info["sampler_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="sampler_sha256"):
        compare_round(score_frames([a]), score_frames([t]), 2)


def test_shared_missing_model_identity_on_both_tests_is_not_evidence_of_a_match():
    a, t = frame(), frame(name="t", sampler="teacher", split="teacher-test")
    del a.info["teacher_sha256"], t.info["teacher_sha256"]
    with pytest.raises(ValueError, match="teacher_sha256"):
        compare_round(score_frames([a]), score_frames([t]), 2)
