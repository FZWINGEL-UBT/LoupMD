"""Stage 0 label metrics on joint REF_* / STUDENT_* ASE extxyz frames.

Both sets of labels must belong to the one geometry in the record; there is no
trajectory matching, energy-offset fitting, model evaluation, or MD test here.
Only held-out, target-condition, student-visited frames can pass the label bar.
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import product
from pathlib import Path

import numpy as np
from ase.geometry import find_mic, minkowski_reduce
from ase.io import iread
from ase.stress import full_3x3_to_voigt_6_stress


_FORCE_FRACTION = 0.02
_ENERGY_LIMIT_MEV = 1.0
_MODEL_KEYS = ("teacher_sha256", "teacher_head", "prediction_sha256", "prediction_head")
_MOL_KEYS = ("mol_id", "mol-id", "molecule_id", "mol")
_SPLITS = {"train", "valid", "teacher-test", "harvest", "fixed-student-test"}


class _Components:
    """Sufficient statistics of equally weighted Cartesian components."""

    def __init__(self):
        self.count = 0
        self.ref_ss = 0.0
        self.error_ss = 0.0

    def add(self, ref, error):
        self.count += ref.size
        self.ref_ss += float(np.vdot(ref, ref))
        self.error_ss += float(np.vdot(error, error))

    def result(self):
        rms = float(np.sqrt(self.ref_ss / self.count))
        rmse = float(np.sqrt(self.error_ss / self.count))
        limit = _FORCE_FRACTION * rms
        return {"components": self.count, "teacher_rms": rms, "rmse": rmse,
                "relative_rmse": rmse / rms if rms else None, "limit": limit,
                "passed": rmse <= limit}


def _label(mapping, key, shape):
    if key not in mapping:
        raise ValueError(f"missing {key}")
    a = np.asarray(mapping[key])
    if a.dtype.kind not in "fiu" or a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{key} must contain finite numeric values of shape {shape}")
    return np.asarray(a, dtype=np.float64)


def _molecule_key(atoms, required):
    if required is not None:
        if required not in atoms.arrays:
            raise ValueError(f"missing required molecule ID array {required}")
        return required
    keys = [key for key in _MOL_KEYS if key in atoms.arrays]
    if len(keys) > 1:
        raise ValueError(f"multiple molecule ID arrays {keys}; select --molecule-key")
    return keys[0] if keys else None


def _molecular_components(atoms, key, ref, error, net, torque):
    ids = np.asarray(atoms.arrays[key])
    if ids.dtype.kind not in "iu" or ids.shape != (len(atoms),) or np.any(ids < 0):
        raise ValueError(f"{key} must contain one nonnegative integer molecule ID per atom")
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(ids)])
    masses = atoms.get_masses()[order]
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("molecular COM requires finite positive masses")

    # With IDs but no bond topology or image flags, unwrap a compact molecule
    # relative to its first atom. Check ALL intramolecular pairs have consistent,
    # unique minimum images (an extended chain cannot silently get a false torque).
    pos = atoms.positions[order]
    cell, _ = minkowski_reduce(atoms.cell)
    displ, _ = find_mic(pos - np.repeat(pos[starts], sizes, axis=0), cell)
    # Voronoi-relevant lattice vectors lie among the {-1,0,1} combinations of
    # the Minkowski-reduced basis. Projection extremes test every pair in O(N).
    for indices in product((-1, 0, 1), repeat=3):
        if indices <= (0, 0, 0):  # one sign suffices
            continue
        vec = np.asarray(indices) @ cell
        projection = displ @ vec
        extent = np.maximum.reduceat(projection, starts) - np.minimum.reduceat(projection, starts)
        if np.any(extent >= (0.5 - 1e-12) * (vec @ vec)):
            raise ValueError("ambiguous periodic molecule images; compact molecules required")
    com = np.add.reduceat(masses[:, None] * displ, starts, axis=0) / np.add.reduceat(masses, starts)[:, None]
    arms = displ - np.repeat(com, sizes, axis=0)
    ref, error = ref[order], error[order]
    net.add(np.add.reduceat(ref, starts, axis=0), np.add.reduceat(error, starts, axis=0))
    torque.add(np.add.reduceat(np.cross(arms, ref), starts, axis=0),
               np.add.reduceat(np.cross(arms, error), starts, axis=0))


def _identity(info):
    split, sampler, frame_id = (info.get(k) for k in ("split", "sampler", "frame_id"))
    if split is None and sampler is None:
        return split, sampler, frame_id
    if split not in _SPLITS or sampler not in ("teacher", "student"):
        raise ValueError(f"invalid split/sampler: {split!r}/{sampler!r}")
    expected = {"teacher-test": "teacher", "fixed-student-test": "student", "harvest": "student"}.get(split)
    if expected and sampler != expected:
        raise ValueError(f"{split} requires {expected} sampler, not {sampler}")
    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ValueError(f"{split} frame missing a nonempty frame_id")
    return split, sampler, frame_id


def _own_test_problem(provenance, rows, split):
    """A held-out set can measure label errors even when it isn't this model's rollout."""
    sampler, model = ("student", "prediction") if split == "fixed-student-test" else ("teacher", "teacher")
    if provenance["split"] != split or provenance["sampler"] != sampler:
        return f"requires {split} frames sampled by {sampler}"
    if provenance["regimes"] != ["target"]:
        return "requires target-condition frames only"
    for key in _MODEL_KEYS:
        value = provenance[key]
        if not isinstance(value, str) or not value.strip():
            return f"missing model identity: {key}"
        if key.endswith("_sha256") and not re.fullmatch("[0-9a-fA-F]{64}", value):
            return f"invalid model identity: {key}"
    for row in rows:
        for suffix in ("sha256", "head"):
            if row[f"sampler_{suffix}"] != provenance[f"{model}_{suffix}"]:
                return f"frame {row['frame_id']}: sampler_{suffix} does not match {model}_{suffix}"
    return None


def score_frames(frames, *, molecule_key=None, tail_multiplier=5.0, regime=None):
    """Pool atoms/components for forces, but weight each frame equally for E/N and stress.

    The p99.9 and maximum use atom-vector error norms. The p99.9 limit, in
    eV/Angstrom, is tail_multiplier times the absolute overall *component* RMSE
    limit (no sqrt(3) rescaling). Stress uses the six unique symmetric tensor
    components in ASE order, not nine elements with double-counted shear.
    Undefined ratios (zero reference) are null; limits still apply in absolute units.
    """
    if not np.isfinite(tail_multiplier) or tail_multiplier <= 0:
        raise ValueError("tail_multiplier must be finite and positive")
    if regime not in (None, "target", "hot"):
        raise ValueError("regime must be target or hot")
    rows, atom_norms = [], []
    frame_ids, regimes = set(), set()
    overall, net, torque = _Components(), _Components(), _Components()
    species = {}
    energy_ss = 0.0
    stress_ss = np.zeros(6)
    stress_max = 0.0
    provenance = {}
    for index, atoms in enumerate(frames):
        info = atoms.info
        frame_regime = info.get("regime")
        if regime is not None and frame_regime != regime:
            continue
        split, sampler, frame_id = _identity(info)
        if frame_id is not None:
            if not isinstance(frame_id, str) or not frame_id.strip():
                raise ValueError(f"frame {index}: invalid frame_id")
            if frame_id in frame_ids:
                raise ValueError(f"duplicate frame_id: {frame_id}")
            frame_ids.add(frame_id)
        mol = _molecule_key(atoms, molecule_key)
        meta = {"split": split, "sampler": sampler, "molecule_key": mol,
                **{key: info.get(key) for key in _MODEL_KEYS}}
        if rows and meta != provenance:
            changed = ", ".join(k for k in meta if meta[k] != provenance[k])
            raise ValueError(f"frame {index}: inconsistent {changed}; score a homogeneous set")
        provenance = meta
        if frame_regime not in (None, "target", "hot"):
            raise ValueError(f"frame {index}: unrecognized regime {frame_regime!r}")
        regimes.add(frame_regime or "unknown")

        n = len(atoms)
        if (not n or not atoms.pbc.all() or not np.isfinite(atoms.positions).all()
                or not np.isfinite(atoms.cell).all() or abs(np.linalg.det(atoms.cell)) == 0):
            raise ValueError(f"frame {index}: need atoms with finite positions and a full periodic cell")
        ref = _label(atoms.arrays, "REF_forces", (n, 3))
        error = _label(atoms.arrays, "STUDENT_forces", (n, 3)) - ref
        e = float((_label(info, "STUDENT_energy", ()) - _label(info, "REF_energy", ())) / n * 1000)
        ds = full_3x3_to_voigt_6_stress(_label(info, "STUDENT_stress", (3, 3)) - _label(info, "REF_stress", (3, 3)))
        if mol is not None:
            _molecular_components(atoms, mol, ref, error, net, torque)
        per_frame = _Components()
        per_frame.add(ref, error)
        overall.count += per_frame.count
        overall.ref_ss += per_frame.ref_ss
        overall.error_ss += per_frame.error_ss
        elements = np.asarray(atoms.get_chemical_symbols())
        for symbol in np.unique(elements):
            mask = elements == symbol
            if symbol not in species:
                species[symbol] = _Components()
            species[symbol].add(ref[mask], error[mask])
        norm = np.linalg.norm(error, axis=1)
        atom_norms.append(norm)
        energy_ss += e * e
        stress_ss += ds * ds
        stress_max = max(stress_max, float(np.abs(ds).max()))
        f = per_frame.result()
        rows.append({"frame_id": frame_id, "index": index, "regime": frame_regime,
                     "trajectory_id": info.get("trajectory_id"), "atoms": n,
                     "step": int(info["step"]) if "step" in info else None,
                     "sampler_sha256": info.get("sampler_sha256"), "sampler_head": info.get("sampler_head"),
                     "force_rmse_eV_per_A": f["rmse"], "teacher_force_rms_eV_per_A": f["teacher_rms"],
                     "relative_force_rmse": f["relative_rmse"], "max_atom_error_eV_per_A": float(norm.max()),
                     "energy_error_meV_per_atom": e, "max_abs_stress_error_eV_per_A3": float(np.abs(ds).max())})
    if not rows:
        raise ValueError("no frames to score after regime selection")
    f = overall.result()
    by_species = {symbol: totals.result() for symbol, totals in sorted(species.items())}
    norms = np.concatenate(atom_norms)
    p999 = float(np.percentile(norms, 99.9, method="linear"))
    tail_limit = tail_multiplier * f["limit"]
    energy_rmse = float(np.sqrt(energy_ss / len(rows)))
    passed = (f["passed"] and all(m["passed"] for m in by_species.values())
              and p999 <= tail_limit and energy_rmse <= _ENERGY_LIMIT_MEV)
    molecular = None
    if net.count:
        molecular = {"molecules": net.count // 3, "net_force": net.result(), "torque": torque.result(),
                     "net_force_unit": "eV/Angstrom", "torque_unit": "eV"}
        passed = passed and molecular["net_force"]["passed"] and molecular["torque"]["passed"]
    provenance["regimes"] = sorted(regimes)
    problem = _own_test_problem(provenance, rows, "fixed-student-test")
    return {
        "scope": "label_metrics_only", "provenance": provenance,
        "frames": rows, "label_metrics_passed": passed,
        "student_visited_label_bar": ("pass" if passed else "fail") if problem is None else "not_evaluated",
        "label_bar_ineligible_reason": problem,
        "md_fidelity": "not_evaluated",
        "forces": {"unit": "eV/Angstrom", "relative_limit": _FORCE_FRACTION, "overall": f,
                   "per_species": by_species, "atom_error_norm": {
                       "p99_9": p999, "max": float(norms.max()), "limit": tail_limit,
                       "multiplier_of_component_limit": tail_multiplier, "passed": p999 <= tail_limit}},
        "energy": {"rmse_meV_per_atom": energy_rmse, "limit_meV_per_atom": _ENERGY_LIMIT_MEV,
                   "passed": energy_rmse <= _ENERGY_LIMIT_MEV},
        "stress": {"unit": "eV/Angstrom^3", "convention": "ASE tensile-positive; symmetric part",
                   "components": ["xx", "yy", "zz", "yz", "xz", "xy"], "max_abs": stress_max,
                   "rmse_by_component": np.sqrt(stress_ss / len(rows)).tolist(),
                   "rmse": float(np.sqrt(stress_ss.mean() / len(rows)))},
        "molecular": molecular,
    }


def compare_round(student, teacher, round_number):
    """Section 9, falsifier 1: compare the same predictor's absolute component force RMSE."""
    if round_number < 0:
        raise ValueError("round must be nonnegative")
    for report, split in ((student, "fixed-student-test"), (teacher, "teacher-test")):
        problem = _own_test_problem(report["provenance"], report["frames"], split)
        if problem is not None:
            raise ValueError(f"round comparison: {problem}")
    for key in _MODEL_KEYS:
        if student["provenance"][key] != teacher["provenance"][key]:
            raise ValueError(f"round comparison: different {key} on test sets")
    ids = {f["frame_id"] for f in student["frames"]}
    if any(f["frame_id"] in ids for f in teacher["frames"]):
        raise ValueError("student and teacher test frame IDs overlap")
    s = student["forces"]["overall"]["rmse"]
    t = teacher["forces"]["overall"]["rmse"]
    return {"round": round_number, "fixed_student_test_rmse_eV_per_A": s,
            "teacher_test_rmse_eV_per_A": t, "student_to_teacher_rmse_ratio": s / t if t else None,
            "exceeds_1_5x": s > 1.5 * t}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("frames", type=Path, help="one extxyz file with joint REF_* and STUDENT_* labels")
    p.add_argument("--output", type=Path, help="JSON report (default: stdout)")
    p.add_argument("--molecule-key", help="require this molecule ID array on every frame; otherwise detect " + ", ".join(_MOL_KEYS))
    p.add_argument("--regime", choices=("target", "hot"), help="score only matching frames; preserves source file indices")
    p.add_argument("--tail-multiplier", type=float, default=5.0, help="p99.9 cap in multiples of the overall component force limit")
    p.add_argument("--teacher-test", type=Path, help="joint teacher-test file to compare with fixed-student-test input")
    p.add_argument("--round", type=int, help="training round for the paired test-set comparison")
    args = p.parse_args(argv)
    if (args.teacher_test is None) != (args.round is None):
        p.error("use --teacher-test and --round together")
    if args.output and any(f and f.resolve() == args.output.resolve() for f in (args.frames, args.teacher_test)):
        p.error("output would overwrite input frames")

    def read_report(path):
        return score_frames(iread(path, index=":", format="extxyz"), regime=args.regime,
                            tail_multiplier=args.tail_multiplier, molecule_key=args.molecule_key)

    try:
        report = read_report(args.frames)
        report["source_file"] = str(args.frames)
        if args.teacher_test is not None:
            report["round_comparison"] = compare_round(report, read_report(args.teacher_test), args.round)
            report["round_comparison"]["teacher_test_file"] = str(args.teacher_test)
        text = json.dumps(report, indent=2, allow_nan=False)
        if args.output is None:
            print(text)
        else:
            args.output.write_text(text + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
