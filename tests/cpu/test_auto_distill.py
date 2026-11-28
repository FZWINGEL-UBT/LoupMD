"""CPU unit tests for the Auto-Distiller pipeline components and direct .loup loading."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.spacegroup import crystal

import loupmd
import loupgym
from loupmd import bundles, models
from loupgym.cli import build_parser
from loupgym.distill.analyze import analyze_system, detect_molecules, propose_candidates
from loupgym.data import normalize_frame, prepare_datasets
from loupmd.errors import CheckpointTrustError, InputError, ModelElementError
from loupmd.export.bundle import write_bundle


class _MockEngine:
    def __init__(self, bundle, device_id=0):
        self.bundle = Path(bundle)
        self.device_id = device_id
        self.calls = []

    def set_system(self, numbers, masses, skin=0.5):
        self.calls.append(("set_system", tuple(int(z) for z in numbers), float(skin)))

    def evaluate(self, positions, cell, compute_stress=True):
        pos = np.asarray(positions, dtype=float)
        return 0.5, -0.1 * pos, (np.eye(3) * 1e-4 if compute_stress else None)


def _make_fake_loup(path: Path, zs=(8, 22, 38), r_max=5.0) -> Path:
    write_bundle(
        path,
        {"arch": "ilmlip", "Z": ",".join(str(z) for z in zs), "exporter_version": 1},
        {"r_max": np.array([r_max], dtype=np.float64)},
    )
    return path


def test_direct_loup_loading_trust_and_elements(tmp_path):
    bp = _make_fake_loup(tmp_path / "student.loup", zs=(8, 22, 38), r_max=5.2)
    assert models.is_loup_bundle(bp)
    digest = hashlib.sha256(bp.read_bytes()).hexdigest()

    # Wrong explicit sha256 must never be bypassed by binary bundle trust
    with pytest.raises(CheckpointTrustError):
        loupmd.load(str(bp), sha256="0" * 64)

    called_export = []
    def fail_ensure(*args, **kwargs):
        called_export.append(args)
        raise AssertionError("ensure_bundle must not be called for direct .loup bundles")

    ff = loupmd.load(str(bp), engine_factory=_MockEngine, ensure_bundle=fail_ensure)
    assert ff.elements == (8, 22, 38)
    assert ff.r_max_A == pytest.approx(5.2)
    assert ff.bundle == bp.resolve()
    assert ff.long_range is False

    sto = crystal(["Sr", "Ti", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221, cellpar=[3.905, 3.905, 3.905, 90, 90, 90])
    en, f, s = ff.evaluate(sto)
    assert called_export == []
    assert isinstance(en, float) and f.shape == (len(sto), 3)

    # Missing element in .loup raises ModelElementError
    nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    with pytest.raises(ModelElementError):
        ff.evaluate(nacl)
    with pytest.raises(ModelElementError):
        bundles.ensure_bundle(bp, [11, 17])


def test_detect_molecules_and_analyze_system():
    # Two water molecules in a box -> detected as 2 molecules
    w1 = molecule("H2O")
    w2 = molecule("H2O")
    w2.translate([4.0, 0.0, 0.0])
    box = w1 + w2
    box.set_cell([12.0, 12.0, 12.0])
    box.set_pbc((True, True, True))
    box.arrays["REF_forces"] = np.ones((len(box), 3)) * 0.2
    mid = detect_molecules(box)
    assert mid is not None and len(set(mid.tolist())) == 2

    prof_w = analyze_system([box])
    assert prof_w.system_kind == "molecular_liquid"
    assert prof_w.has_molecular_units is True
    assert prof_w.has_h_bonds is True
    cands_w = propose_candidates(prof_w, speed_preset="fast50")
    assert len(cands_w) >= 3 and cands_w[0].wct > 0.0

    # Periodic crystal SrTiO3 -> ionic_crystal, no molecular units
    sto = crystal(["Sr", "Ti", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221, cellpar=[3.905, 3.905, 3.905, 90, 90, 90]).repeat((2, 2, 2))
    sto.arrays["REF_forces"] = np.ones((len(sto), 3)) * 0.4
    assert detect_molecules(sto) is None
    prof_s = analyze_system([sto])
    assert prof_s.system_kind == "ionic_crystal"
    assert prof_s.has_molecular_units is False
    cands_s = propose_candidates(prof_s, speed_preset="fast50")
    assert all(c.r_max == c.r_max2 for c in cands_s[:2])


def test_normalize_frame_and_prepare_dft_only():
    mol = molecule("CH4")
    mol.set_pbc((False, False, False))
    mol.info["dft_e"] = -12.34
    mol.arrays["dft_f"] = np.arange(len(mol) * 3, dtype=float).reshape(-1, 3) * 0.01
    mol.info["stress"] = np.zeros(6)

    frames = []
    for _ in range(8):
        norm, ok = normalize_frame(mol, energy_key="dft_e", forces_key="dft_f")
        assert ok is True
        assert "REF_energy" in norm.info
        assert "REF_forces" in norm.arrays
        assert "REF_stress" not in norm.info  # stripped for pbc=False
        frames.append(norm)

    prep = prepare_datasets(dft_data=frames, energy_key="dft_e", forces_key="dft_f")
    assert prep.elements == (1, 6)
    assert prep.has_stress is False
    assert len(prep.train_frames) + len(prep.val_frames) + len(prep.test_frames) == 8

    with pytest.raises(InputError):
        prepare_datasets()


def test_cli_distill_parser():
    parser = build_parser()
    args = parser.parse_args(["distill", "--teacher", "omat0-small", "--structure", "s.cif", "--speed", "fast50", "--budget", "quick", "-o", "out"])
    assert args.command == "distill"
    assert args.teacher == "omat0-small"
    assert args.speed == "fast50"
    assert args.budget == "quick"
    assert callable(loupgym.auto_distill)
    assert not hasattr(loupmd, "auto_distill")
    assert not hasattr(loupmd, "DistillResult")


def test_fixed_d3bj_force_shift_continuity_fd_and_double_count_guard(tmp_path):
    from loupmd import dispersion as disp_mod
    from loupmd.errors import PlanRefused
    from loupgym.dispersion_d3 import eval_fixed_d3bj_f64, fit_fixed_d3bj, transform_labels_for_d3

    sto = crystal(
        ["Sr", "Ti", "O"],
        [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
    ).repeat((2, 2, 2))
    rng = np.random.default_rng(11)
    sto.positions += rng.normal(0.0, 0.05, size=sto.positions.shape)

    spec = fit_fixed_d3bj([sto], (8, 22, 38), r_cut=4.6, functional="pbe", force_shift=True, use_tail=True)
    ref = eval_fixed_d3bj_f64(sto, spec)
    assert ref["e_tail"] < 0.0

    # 1) Dimer scan across r_cut: C^1 continuity (both E and F vanish smoothly at r_cut)
    d_in = Atoms("OO", positions=[[0.0, 0.0, 0.0], [4.5999, 0.0, 0.0]], cell=[20.0, 20.0, 20.0], pbc=True)
    d_out = Atoms("OO", positions=[[0.0, 0.0, 0.0], [4.6001, 0.0, 0.0]], cell=[20.0, 20.0, 20.0], pbc=True)
    spec_nt = fit_fixed_d3bj([sto], (8, 22, 38), r_cut=4.6, functional="pbe", force_shift=True, use_tail=False)
    r_in = eval_fixed_d3bj_f64(d_in, spec_nt)
    r_out = eval_fixed_d3bj_f64(d_out, spec_nt)
    assert abs(r_in["energy"] - r_out["energy"]) < 1e-8
    assert float(np.max(np.abs(r_in["forces"] - r_out["forces"]))) < 1e-5

    # 2) Central finite-difference check on atomic force and isotropic volume tail + pair virial
    eps = 1e-5
    at_p, at_m = sto.copy(), sto.copy()
    at_p.positions[0, 0] += eps
    at_m.positions[0, 0] -= eps
    f_fd = -(eval_fixed_d3bj_f64(at_p, spec)["energy"] - eval_fixed_d3bj_f64(at_m, spec)["energy"]) / (2.0 * eps)
    assert abs(ref["forces"][0, 0] - f_fd) < 1e-7

    at_vp, at_vm = sto.copy(), sto.copy()
    at_vp.set_cell(sto.cell.array * (1.0 + eps), scale_atoms=True)
    at_vm.set_cell(sto.cell.array * (1.0 - eps), scale_atoms=True)
    tr_v_fd = (eval_fixed_d3bj_f64(at_vp, spec)["energy"] - eval_fixed_d3bj_f64(at_vm, spec)["energy"]) / (2.0 * eps)
    tr_v_ana = ref["virial"][0] + ref["virial"][3] + ref["virial"][5]
    assert abs(tr_v_ana - tr_v_fd) < 1e-5

    # 3) Double-counting guard on a bundle carrying has_builtin_disp=1
    d_meta, d_tensors = spec.to_bundle_entries()
    bp = tmp_path / "student_disp.loup"
    write_bundle(bp, {"arch": "ilmlip", "Z": "8,22,38", "exporter_version": 1, **d_meta}, {"r_max": np.array([4.6]), **d_tensors})
    ff = loupmd.load(str(bp), engine_factory=_MockEngine)
    assert ff.has_builtin_disp is True
    assert ff.family == "ilmlip-disp"
    with pytest.raises(PlanRefused):
        disp_mod.resolve("d3bj", functional=ff.functional, family=ff.family, model=ff.name)

    # 4) Label transform sanity check (labels_have_d3=True subtracts builtin+tail)
    sto.info["REF_energy"] = -100.0
    sto.arrays["REF_forces"] = np.zeros((len(sto), 3))
    sto.info["REF_stress"] = np.zeros((3, 3))
    tr = transform_labels_for_d3([sto], spec, labels_have_d3=True)[0]
    assert abs(tr.info["REF_energy"] - (-100.0 - ref["energy"])) < 1e-12
    assert np.allclose(tr.arrays["REF_forces"], -ref["forces"])

    # 5) Option C (CN-dependent C6) exact central FD force & virial check on a water box (unsaturated CN!)
    from loupgym.dispersion_d3 import eval_cn_d3bj_f64, fit_cn_d3bj

    w1, w2 = molecule("H2O"), molecule("H2O")
    w2.translate([2.9, 0.2, 0.1])
    wbox = w1 + w2
    wbox.set_cell([8.0, 8.0, 8.0])
    wbox.set_pbc((True, True, True))
    cn_spec = fit_cn_d3bj([wbox], (1, 8), r_cut=5.4, functional="pbe", force_shift=True, use_tail=True)
    cn_ref = eval_cn_d3bj_f64(wbox, cn_spec)
    wp, wm = wbox.copy(), wbox.copy()
    wp.positions[0, 0] += eps
    wm.positions[0, 0] -= eps
    f_cn_fd = -(eval_cn_d3bj_f64(wp, cn_spec)["energy"] - eval_cn_d3bj_f64(wm, cn_spec)["energy"]) / (2.0 * eps)
    assert abs(cn_ref["forces"][0, 0] - f_cn_fd) < 1e-6
    wvp, wvm = wbox.copy(), wbox.copy()
    wvp.set_cell(wbox.cell.array * (1.0 + eps), scale_atoms=True)
    wvm.set_cell(wbox.cell.array * (1.0 - eps), scale_atoms=True)
    v_cn_fd = (eval_cn_d3bj_f64(wvp, cn_spec)["energy"] - eval_cn_d3bj_f64(wvm, cn_spec)["energy"]) / (2.0 * eps)
    assert abs((cn_ref["virial"][0] + cn_ref["virial"][3] + cn_ref["virial"][5]) - v_cn_fd) < 1e-5
