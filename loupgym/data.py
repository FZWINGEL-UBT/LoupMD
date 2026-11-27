"""Flexible data ingestion, normalization, teacher labeling/MD sampling, and DFT alignment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.stress import voigt_6_to_full_3x3_stress

from loupmd.errors import BackendError, InputError


@dataclass
class PreparedData:
    train_frames: list[Atoms]
    val_frames: list[Atoms]
    test_frames: list[Atoms]
    dft_train_frames: list[Atoms]
    elements: tuple[int, ...]
    has_stress: bool
    has_ei: bool
    e0_offset_summary: dict | None = None


def load_frames_input(source: str | Path | Atoms | Sequence[Atoms] | None) -> list[Atoms]:
    """Read ASE ``Atoms`` or structure/trajectory files into a flat list of frames."""
    if source is None:
        return []
    if isinstance(source, Atoms):
        return [source.copy()]
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise InputError(f"input data file {p} not found")
        loaded = ase_read(str(p), index=":")
        if isinstance(loaded, Atoms):
            return [loaded]
        return list(loaded)
    out: list[Atoms] = []
    for item in source:
        if isinstance(item, Atoms):
            out.append(item.copy())
        elif isinstance(item, (str, Path)):
            out.extend(load_frames_input(item))
        else:
            raise InputError(f"unsupported frame item type: {type(item).__name__}")
    return out


def _extract_stress(at: Atoms, raw_stress) -> np.ndarray | None:
    if raw_stress is None:
        return None
    arr = np.asarray(raw_stress, dtype=np.float64)
    if arr.shape == (6,):
        return voigt_6_to_full_3x3_stress(arr)
    if arr.shape == (3, 3):
        return arr.copy()
    if arr.size == 9:
        return arr.reshape(3, 3).copy()
    return None


def normalize_frame(
    atoms: Atoms,
    *,
    energy_key: str | None = None,
    forces_key: str | None = None,
    stress_key: str | None = None,
    weight: float = 1.0,
) -> tuple[Atoms, bool]:
    """Normalize frame into ``REF_energy``, ``REF_forces``, optional ``REF_stress`` without calculator baggage.

    Returns ``(normalized_atoms, is_fully_labeled)``.
    """
    calc_res = dict(getattr(getattr(atoms, "calc", None), "results", {}) or {})
    a = atoms.copy()
    a.calc = None

    e_val = None
    for k in (energy_key, "REF_energy", "energy", "dft_energy", "free_energy", "E"):
        if k and k in a.info:
            e_val = float(a.info[k])
            break
    if e_val is None and "energy" in calc_res:
        e_val = float(calc_res["energy"])

    f_arr = None
    for k in (forces_key, "REF_forces", "forces", "dft_forces", "F"):
        if k and k in a.arrays:
            f_arr = np.asarray(a.arrays[k], dtype=np.float64).copy()
            break
    if f_arr is None and "forces" in calc_res:
        f_arr = np.asarray(calc_res["forces"], dtype=np.float64).copy()

    s_arr = None
    for k in (stress_key, "REF_stress", "stress", "dft_stress"):
        if k and k in a.info:
            s_arr = _extract_stress(a, a.info[k])
            if s_arr is not None:
                break
    if s_arr is None and "stress" in calc_res:
        s_arr = _extract_stress(a, calc_res["stress"])

    if e_val is not None:
        a.info["REF_energy"] = float(e_val)
    if f_arr is not None:
        a.arrays["REF_forces"] = f_arr
    if s_arr is not None and all(bool(p) for p in a.pbc):
        a.info["REF_stress"] = s_arr
    elif "REF_stress" in a.info and not all(bool(p) for p in a.pbc):
        del a.info["REF_stress"]

    a.info["distill_weight"] = float(weight)
    return a, (e_val is not None and f_arr is not None)


def _pad_nonperiodic_for_engine(atoms: Atoms, r_max: float) -> tuple[Atoms, bool]:
    """Ensure a non-periodic molecule sits in an isolated periodic box ``>= 2*r_max + 8 A`` wide."""
    if all(bool(p) for p in atoms.pbc) and abs(float(np.linalg.det(atoms.cell.array))) > 1e-6:
        return atoms.copy(), False
    a = atoms.copy()
    pos = np.asarray(a.positions, dtype=np.float64)
    center = pos.mean(axis=0, keepdims=True) if len(pos) else np.zeros((1, 3))
    span = np.ptp(pos, axis=0) if len(pos) else np.zeros(3)
    pad = 2.0 * float(r_max) + 8.0
    box = np.maximum(span + pad, pad)
    a.set_cell(np.diag(box))
    a.set_pbc((True, True, True))
    a.set_positions(pos - center + 0.5 * box[None, :])
    return a, True


def _sampling_supercell(atoms: Atoms, r_max: float) -> tuple[Atoms, bool]:
    """Expand very small periodic unit cells for diverse MD sampling; pad isolated molecules."""
    boxed, was_nonperiodic = _pad_nonperiodic_for_engine(atoms, r_max)
    if was_nonperiodic:
        return boxed, True
    # Keep tiny cells replicated so thermostatting and phonons avoid gamma-only artifacts
    a = boxed
    lens = np.linalg.norm(a.cell.array, axis=1)
    rep = [1, 1, 1]
    if len(a) < 24:
        for ax in range(3):
            if lens[ax] < max(10.0, 2.0 * r_max + 0.5):
                rep[ax] = max(2, int(np.ceil((2.0 * r_max + 0.5) / max(1.0, lens[ax]))))
    if rep != [1, 1, 1]:
        a = a.repeat(rep)
    return a, False


def label_frames_with_teacher(
    frames: Sequence[Atoms],
    teacher_ff,
    *,
    per_atom: bool = True,
    weight: float = 1.0,
) -> list[Atoms]:
    """Evaluate all ``frames`` with ``teacher_ff`` GPU engine and write ``REF_*`` labels."""
    if not frames:
        return []
    r_cut = float(getattr(teacher_ff, "r_max_A", None) or 6.0)
    labeled: list[Atoms] = []
    use_per_atom = bool(per_atom) and not bool(getattr(teacher_ff, "long_range", False))

    for raw in frames:
        orig_pbc = tuple(bool(p) for p in raw.pbc)
        orig_periodic = all(orig_pbc) and abs(float(np.linalg.det(raw.cell.array))) > 1e-6
        boxed, _ = _pad_nonperiodic_for_engine(raw, r_cut)
        eng = teacher_ff.engine_for(boxed, skin=0.5)

        e_i = None
        if use_per_atom:
            try:
                en, frc, str_arr, e_i = eng.evaluate_per_atom(
                    boxed.positions, boxed.cell.array, compute_stress=orig_periodic
                )
            except (BackendError, NotImplementedError):
                use_per_atom = False
                en, frc, str_arr = eng.evaluate(boxed.positions, boxed.cell.array, compute_stress=orig_periodic)
        else:
            en, frc, str_arr = eng.evaluate(boxed.positions, boxed.cell.array, compute_stress=orig_periodic)

        out = raw.copy()
        out.calc = None
        out.info["REF_energy"] = float(en)
        out.arrays["REF_forces"] = np.asarray(frc, dtype=np.float64).copy()
        if orig_periodic and str_arr is not None:
            out.info["REF_stress"] = np.asarray(str_arr, dtype=np.float64).reshape(3, 3).copy()
        elif "REF_stress" in out.info:
            del out.info["REF_stress"]
        if e_i is not None:
            out.arrays["REF_ei"] = np.asarray(e_i, dtype=np.float64).copy()
        out.info["distill_weight"] = float(weight)
        labeled.append(out)
    return labeled


def sample_teacher_md(
    structures: Sequence[Atoms],
    teacher_ff,
    *,
    n_target_frames: int = 80,
    temps_k: Sequence[float] = (300.0, 600.0),
    steps_per_sample: int = 25,
    dt_fs: float = 0.5,
    rattle_std: float = 0.03,
    seed: int = 42,
    log_fn=None,
) -> list[Atoms]:
    """Run short multi-temperature Langevin MD + strained/rattled sampling on ``teacher_ff`` GPU engine."""
    if not structures:
        return []
    rng = np.random.default_rng(seed)
    r_cut = float(getattr(teacher_ff, "r_max_A", None) or 6.0)
    sampled_raw: list[Atoms] = []

    n_structs = len(structures)
    per_combo = max(2, int(np.ceil(n_target_frames / max(1, n_structs * len(temps_k)))))

    for s_idx, base_at in enumerate(structures):
        orig_periodic = all(bool(p) for p in base_at.pbc) and abs(float(np.linalg.det(base_at.cell.array))) > 1e-6
        samp, was_nonp = _sampling_supercell(base_at, r_cut)
        masses = samp.get_masses()

        # Include exact starting structure (plus 2 small rattles if only 1 input structure)
        sampled_raw.append(samp.copy() if not was_nonp else base_at.copy())
        if n_structs <= 2 and rattle_std > 0:
            for k in range(2):
                r_at = (samp if not was_nonp else base_at).copy()
                r_at.positions = r_at.positions + rng.normal(0.0, rattle_std * (k + 1), size=r_at.positions.shape)
                sampled_raw.append(r_at)

        # Mild volume strain diversity for periodic solids
        from .distill.analyze import detect_molecules

        vol_scales = (0.97, 1.03) if (orig_periodic and detect_molecules(base_at) is None and len(temps_k) >= 2) else (1.0,)

        for t_idx, temp_k in enumerate(temps_k):
            v_scale = vol_scales[t_idx % len(vol_scales)]
            cur = samp.copy()
            if abs(v_scale - 1.0) > 1e-6:
                cur.set_cell(cur.cell.array * v_scale, scale_atoms=True)
            eng = teacher_ff.engine_for(cur, skin=0.5)
            eng.set_state(cur.positions, np.zeros_like(cur.positions), cur.cell.array)

            # Brief equilibration before harvesting
            eng.run_langevin(steps_per_sample * 2, dt_fs=dt_fs, temp_k=float(temp_k), friction_per_fs=0.02, seed=seed + 101 * (s_idx + 1) + t_idx)
            for m in range(per_combo):
                eng.run_langevin(steps_per_sample, dt_fs=dt_fs, temp_k=float(temp_k), friction_per_fs=0.015, seed=seed + 1000 + 37 * m + t_idx)
                pos, _, _, _ = eng.get_state()
                snap = cur.copy()
                snap.set_positions(pos)
                if was_nonp:
                    snap.set_pbc((False, False, False))
                    snap.set_cell(base_at.cell.array.copy())
                    # Recenter inside original reference coords
                    snap.positions = snap.positions - snap.positions.mean(axis=0, keepdims=True)
                sampled_raw.append(snap)
                if len(sampled_raw) >= n_target_frames + 8:
                    break
            if len(sampled_raw) >= n_target_frames + 8:
                break

    # Deduplicate count and label all frames cleanly with teacher
    sampled_raw = sampled_raw[: max(n_target_frames, len(structures) + 4)]
    if log_fn is not None:
        log_fn(f"  Sampled {len(sampled_raw)} teacher-MD frames across T={list(temps_k)} K; labeling...")
    return label_frames_with_teacher(sampled_raw, teacher_ff, per_atom=True, weight=1.0)


def align_teacher_to_dft(
    teacher_frames: Sequence[Atoms],
    dft_frames: Sequence[Atoms],
    teacher_ff,
    elements: Sequence[int],
) -> tuple[list[Atoms], dict]:
    """Fit per-species E0 offset ``delta_e0 = lstsq(N_sp, E_dft - E_teacher)`` and shift teacher energies."""
    if not teacher_frames or not dft_frames:
        return list(teacher_frames), {"applied": False}

    # Evaluate teacher on up to 32 DFT frames cleanly without mutating caller's dft_frames
    probe_dft = list(dft_frames[: min(32, len(dft_frames))])
    t_on_dft = label_frames_with_teacher([a.copy() for a in probe_dft], teacher_ff, per_atom=False)

    z2i = {int(z): i for i, z in enumerate(elements)}
    S = len(elements)
    A = np.zeros((len(probe_dft), S), dtype=np.float64)
    y = np.zeros(len(probe_dft), dtype=np.float64)
    f_diffs = []

    for k, (df, tf) in enumerate(zip(probe_dft, t_on_dft)):
        y[k] = float(df.info["REF_energy"]) - float(tf.info["REF_energy"])
        for z in df.numbers:
            A[k, z2i[int(z)]] += 1.0
        f_diffs.append(((np.asarray(df.arrays["REF_forces"]) - np.asarray(tf.arrays["REF_forces"])) ** 2).ravel())

    delta_e0 = np.linalg.solve(A.T @ A + 1e-8 * np.eye(S), A.T @ y)
    aligned: list[Atoms] = []
    for tf in teacher_frames:
        out = tf.copy()
        shift = sum(float(delta_e0[z2i[int(z)]]) for z in out.numbers if int(z) in z2i)
        out.info["REF_energy"] = float(out.info["REF_energy"]) + shift
        aligned.append(out)

    f_gap_mev = float(np.sqrt(np.mean(np.concatenate(f_diffs))) * 1000.0) if f_diffs else 0.0
    summary = {
        "applied": True,
        "delta_e0_ev": [round(float(x), 6) for x in delta_e0],
        "teacher_vs_dft_f_rmse_mev": round(f_gap_mev, 2),
    }
    return aligned, summary


def _attach_mol_ids(frames: Sequence[Atoms]):
    from .distill.analyze import detect_molecules

    for at in frames:
        if "mol_id" not in at.arrays:
            mid = detect_molecules(at)
            if mid is not None:
                at.arrays["mol_id"] = mid


def _split_frames(frames: Sequence[Atoms], val_frac: float = 0.15, test_frac: float = 0.10, seed: int = 42):
    n = len(frames)
    if n < 3:
        return list(frames), list(frames[:1]), list(frames[:1])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac))) if n >= 8 else 1
    n_val = max(1, int(round(n * val_frac))) if n >= 5 else 1
    n_train = max(1, n - n_val - n_test)
    tr = [frames[i] for i in idx[:n_train]]
    va = [frames[i] for i in idx[n_train : n_train + n_val]]
    te = [frames[i] for i in idx[n_train + n_val :]]
    if not te:
        te = va[:]
    return tr, va, te


def prepare_datasets(
    *,
    teacher_ff=None,
    structure=None,
    md_data=None,
    dft_data=None,
    n_md_frames: int = 80,
    temps_k: Sequence[float] = (300.0, 600.0),
    no_teacher_aug: bool = False,
    dft_weight: float = 3.0,
    energy_key: str | None = None,
    forces_key: str | None = None,
    stress_key: str | None = None,
    seed: int = 42,
    log_fn=None,
) -> PreparedData:
    """Ingest, normalize, sample, label, and align any valid combination of inputs."""
    struct_frames = load_frames_input(structure)
    md_raw = load_frames_input(md_data)
    dft_raw = load_frames_input(dft_data)

    if teacher_ff is None and not dft_raw and not md_raw:
        raise InputError("Provide at least --teacher or labeled --dft-data / --md-data")
    if teacher_ff is not None and not struct_frames and not md_raw and not dft_raw:
        raise InputError("With --teacher, provide at least one of --structure, --md-data, or --dft-data")

    # Normalize DFT data (must already contain E & F)
    dft_frames: list[Atoms] = []
    for raw in dft_raw:
        at, ok = normalize_frame(raw, energy_key=energy_key, forces_key=forces_key, stress_key=stress_key, weight=dft_weight)
        if not ok:
            raise InputError("DFT data frame is missing energy or forces (checked REF_energy/energy and REF_forces/forces)")
        dft_frames.append(at)

    # Normalize MD data; if unlabeled and teacher provided, label it with teacher
    md_norm: list[Atoms] = []
    md_unlabeled: list[Atoms] = []
    for raw in md_raw:
        at, ok = normalize_frame(raw, energy_key=energy_key, forces_key=forces_key, stress_key=stress_key, weight=1.0)
        if ok and teacher_ff is None:
            md_norm.append(at)
        elif teacher_ff is not None:
            # With teacher provided, label MD geometries so targets are smooth & consistent
            md_unlabeled.append(at)
        else:
            raise InputError("MD data is missing labels and no --teacher was provided")

    if md_unlabeled:
        if log_fn is not None:
            log_fn(f"  Labeling {len(md_unlabeled)} MD frames with teacher...")
        md_norm.extend(label_frames_with_teacher(md_unlabeled, teacher_ff, per_atom=True, weight=1.0))

    # Subsample MD if huge? Keep up to max(n_md_frames * 2, len) or all if <= 600
    if len(md_norm) > max(400, n_md_frames * 3):
        stride = max(1, len(md_norm) // max(200, n_md_frames * 2))
        md_norm = md_norm[::stride]

    # Sample teacher MD if we have starting structure(s), or if DFT data benefits from augmentation
    sampled_teacher_frames: list[Atoms] = []
    if teacher_ff is not None and (struct_frames or (dft_frames and not md_norm and not no_teacher_aug)):
        seeds_for_md = struct_frames if struct_frames else dft_frames[: min(3, len(dft_frames))]
        n_samp = n_md_frames if struct_frames else max(24, min(60, len(dft_frames)))
        sampled_teacher_frames = sample_teacher_md(
            seeds_for_md, teacher_ff, n_target_frames=n_samp, temps_k=temps_k, seed=seed, log_fn=log_fn
        )

    teacher_pool = list(md_norm) + list(sampled_teacher_frames)

    all_elems: set[int] = set()
    for grp in (struct_frames, teacher_pool, dft_frames):
        for at in grp:
            all_elems.update(int(z) for z in at.numbers)
    elements = tuple(sorted(all_elems))

    e0_offset_summary = None
    if dft_frames and teacher_pool and teacher_ff is not None:
        teacher_pool, e0_offset_summary = align_teacher_to_dft(teacher_pool, dft_frames, teacher_ff, elements)
        if log_fn is not None and e0_offset_summary.get("applied"):
            log_fn(f"  Aligned teacher->DFT E0 offset={e0_offset_summary['delta_e0_ev']} (teacher vs DFT dF={e0_offset_summary['teacher_vs_dft_f_rmse_mev']} meV/A)")

    if dft_frames:
        dft_tr, dft_va, dft_te = _split_frames(dft_frames, val_frac=0.15, test_frac=0.15, seed=seed)
        if teacher_pool:
            # Always keep validation & test on ground-truth DFT!
            train_frames = list(dft_tr) + list(teacher_pool)
            val_frames = list(dft_va)
            test_frames = list(dft_te)
        else:
            train_frames, val_frames, test_frames = dft_tr, dft_va, dft_te
        dft_train_frames = dft_tr
    else:
        train_frames, val_frames, test_frames = _split_frames(teacher_pool, val_frac=0.15, test_frac=0.10, seed=seed)
        dft_train_frames = []

    _attach_mol_ids(train_frames)
    _attach_mol_ids(val_frames)
    _attach_mol_ids(test_frames)
    _attach_mol_ids(dft_train_frames)

    has_stress = all("REF_stress" in at.info for at in train_frames + val_frames)
    has_ei = bool(train_frames) and all("REF_ei" in at.arrays for at in train_frames)

    return PreparedData(
        train_frames=train_frames,
        val_frames=val_frames,
        test_frames=test_frames,
        dft_train_frames=dft_train_frames,
        elements=elements,
        has_stress=has_stress,
        has_ei=has_ei,
        e0_offset_summary=e0_offset_summary,
    )
