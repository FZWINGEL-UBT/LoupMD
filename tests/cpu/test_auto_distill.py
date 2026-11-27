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
