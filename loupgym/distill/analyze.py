"""System physics analyzer, candidate generator, and live-GPU speed selector."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, chemical_symbols, covalent_radii
from ase.neighborlist import primitive_neighbor_list

SPEED_PRESET_US: dict[str, float | None] = {
    "fast50": 0.050,
    "balanced80": 0.080,
    "accurate100": 0.100,
    "max": None,
}


@dataclass(frozen=True)
class SystemProfile:
    elements: tuple[int, ...]
    symbols: tuple[str, ...]
    n_atoms_mean: float
    n_atoms_max: int
    periodic: bool
    density_g_cm3: float
    has_molecular_units: bool
    n_molecules_mean: float
    has_h_bonds: bool
    system_kind: str
    first_shell_r_A: float
    second_shell_r_A: float
    recommended_r_max: float
    recommended_r_max2: float
    force_rms: float
    per_species_force_rms: dict[int, float]
    species_imbalance_ratio: float
    suggested_species_weights: tuple[float, ...] | None


@dataclass(frozen=True)
class StudentSpec:
    name: str
    r_max: float
    r_max2: float
    lmax: int
    lfeat: int
    taper: tuple[int, ...]
    n_base: int
    n_base2: int
    c0: int
    c1: int
    c2: int
    readout_depth: int
    readout_width: int
    pol_gram: bool = True
    pol_field: bool = True
    field_mlp_width: int = 32
    radial_warp: float = 0.0
    use_zbl_switch: bool = True
    film_radials: bool = True
    kappa: float = 16.0
    a0: float = 0.05
    z_c: float = 10.0
    max_order: int = 3
    sw_mp: bool = True
    mp_cap: float = 32.0
    ef_cap: float = 32.0
    wct: float = 0.0


def detect_molecules(atoms: Atoms, scale: float = 1.25) -> np.ndarray | None:
    """Detect covalently bonded molecular units and return per-atom ``mol_id`` if multi-unit, else None."""
    N = len(atoms)
    if N <= 1:
        return None
    zs = np.asarray(atoms.numbers, dtype=np.int64)
    rcov = np.asarray([float(covalent_radii[int(z)]) for z in zs], dtype=np.float64)
    r_cut = float(scale * (2.0 * np.max(rcov)) + 0.1)
    pbc = tuple(bool(x) for x in atoms.pbc)
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    i_idx, j_idx, d_arr = primitive_neighbor_list("ijd", pbc, cell, np.asarray(atoms.positions, dtype=np.float64), r_cut)

    parent = list(range(N))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for ii, jj, dij in zip(i_idx, j_idx, d_arr):
        if dij < scale * (rcov[ii] + rcov[jj]):
            union(int(ii), int(jj))

    roots = [find(i) for i in range(N)]
    comp_map: dict[int, int] = {}
    mol_id = np.empty(N, dtype=np.int32)
    comp_sizes: dict[int, int] = {}
    for i, r in enumerate(roots):
        cid = comp_map.get(r)
        if cid is None:
            cid = len(comp_map)
            comp_map[r] = cid
            comp_sizes[cid] = 0
        mol_id[i] = cid
        comp_sizes[cid] += 1

    n_comp = len(comp_map)
    if n_comp < 2:
        return None
    multi_atom_comps = sum(1 for sz in comp_sizes.values() if sz >= 2)
    if multi_atom_comps == 0:
        return None
    return mol_id


def _estimate_shells(frames: Sequence[Atoms], elements: Sequence[int]) -> tuple[float, float]:
    """Estimate 1st and 2nd neighbor shells from pair distances of sample frames."""
    dists = []
    for at in frames[: min(8, len(frames))]:
        pbc = tuple(bool(x) for x in at.pbc)
        cell = np.asarray(at.cell.array, dtype=np.float64)
        pos = np.asarray(at.positions, dtype=np.float64)
        _, _, d = primitive_neighbor_list("ijd", pbc, cell, pos, 6.5)
        if len(d) > 0:
            dists.append(d[d > 0.6])
    if not dists:
        max_rcov = max(float(covalent_radii[int(z)]) for z in elements)
        return (min(4.8, max(3.6, 2.8 * max_rcov)), 5.8)

    all_d = np.concatenate(dists)
    hist, edges = np.histogram(all_d, bins=60, range=(0.5, 6.5))
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Smooth lightly
    kernel = np.array([0.25, 0.5, 0.25])
    sm = np.convolve(hist.astype(np.float64), kernel, mode="same")
    # Locate first substantive peak above 1.2 A
    p1_mask = (centers >= 1.2) & (centers <= 3.8)
    p1_idx = int(np.argmax(sm[p1_mask])) + int(np.argmax(p1_mask))
    # First minimum after p1_idx
    first_shell = float(centers[p1_idx]) + 1.0
    for k in range(p1_idx + 2, len(centers) - 1):
        if sm[k] <= sm[k - 1] and sm[k] <= sm[k + 1] and centers[k] >= 2.2:
            first_shell = float(centers[k]) + 0.3
            break
    first_shell = float(np.clip(first_shell, 3.6, 5.8))
    second_shell = float(np.clip(max(first_shell + 1.2, 5.5), first_shell, 6.2))
    return first_shell, second_shell


def analyze_system(frames: Sequence[Atoms], elements: Sequence[int] | None = None) -> SystemProfile:
    """Analyze geometry, bonding, composition, and reference force statistics across ``frames``."""
    if not frames:
        raise ValueError("analyze_system requires at least one ASE Atoms frame")

    elem_set: set[int] = set(int(z) for z in (elements or ()))
    n_atoms_l: list[int] = []
    densities: list[float] = []
    mol_counts: list[int] = []
    any_mol = False
    periodic = all(bool(p) for p in frames[0].pbc)

    for at in frames[: min(32, len(frames))]:
        elem_set.update(int(z) for z in at.numbers)
        n_atoms_l.append(len(at))
        if all(bool(p) for p in at.pbc):
            vol = abs(float(np.linalg.det(at.cell.array)))
            if vol > 1e-6:
                mass_amu = sum(float(atomic_masses[int(z)]) for z in at.numbers)
                densities.append(mass_amu * 1.6605390666 / vol)
        if "mol_id" in at.arrays:
            any_mol = True
            mol_counts.append(int(np.max(at.arrays["mol_id"])) + 1)
        else:
            mid = detect_molecules(at)
            if mid is not None:
                any_mol = True
                mol_counts.append(int(np.max(mid)) + 1)
            else:
                mol_counts.append(1)

    elems = tuple(sorted(elem_set))
    symbols = tuple(chemical_symbols[z] for z in elems)
    density = float(np.mean(densities)) if densities else 0.0

    # Simple chemical/structural classification
    has_h = 1 in elems
    has_hbond_acc = any(z in elems for z in (7, 8, 9))
    has_h_bonds = bool(has_h and has_hbond_acc and any_mol)
    non_metals = {1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 34, 35, 36, 53, 54}
    has_metal = any(z not in non_metals for z in elems)

    if not periodic:
        system_kind = "isolated_molecule"
    elif any_mol and has_h_bonds and not has_metal:
        system_kind = "molecular_liquid"
    elif has_metal and any(z in (8, 9, 16, 17, 35, 53) for z in elems) and not any_mol:
        system_kind = "ionic_crystal"
    elif not any_mol:
        system_kind = "covalent_solid"
    else:
        system_kind = "mixed_condensed"

    r1, r2 = _estimate_shells(frames, elems)
    if system_kind == "molecular_liquid":
        rec_r1, rec_r2 = 4.2, 5.8
    elif system_kind == "ionic_crystal":
        rec_r1 = float(np.clip(max(r1, 5.2), 4.8, 5.8))
        rec_r2 = rec_r1
    elif system_kind == "isolated_molecule":
        rec_r1 = float(np.clip(max(r1, 4.5), 4.2, 5.2))
        rec_r2 = rec_r1
    else:
        rec_r1 = float(np.clip(r1, 4.2, 5.5))
        rec_r2 = float(np.clip(max(r2, rec_r1), rec_r1, 5.8))

    # Force scale & species imbalance
    sp_sq: dict[int, list[np.ndarray]] = {z: [] for z in elems}
    all_f_sq: list[np.ndarray] = []
    for at in frames:
        if "REF_forces" not in at.arrays:
            continue
        f = np.asarray(at.arrays["REF_forces"], dtype=np.float64)
        all_f_sq.append((f ** 2).ravel())
        zs = np.asarray(at.numbers, dtype=np.int64)
        for z in elems:
            mask = zs == z
            if np.any(mask):
                sp_sq[z].append((f[mask] ** 2).ravel())

    f_rms = float(np.sqrt(np.mean(np.concatenate(all_f_sq)))) if all_f_sq else 1.0
    per_sp_rms = {
        z: (float(np.sqrt(np.mean(np.concatenate(v)))) if v else f_rms)
        for z, v in sp_sq.items()
    }
    vals = [max(1e-6, v) for v in per_sp_rms.values()]
    imb = float(max(vals) / min(vals)) if len(vals) >= 2 else 1.0
    sp_weights = None
    if imb > 2.5 and f_rms > 0:
        raw = [float(np.clip(math.sqrt(f_rms / max(1e-6, per_sp_rms[z])), 0.5, 2.0)) for z in elems]
        mean_w = sum(raw) / len(raw)
        sp_weights = tuple(round(w / mean_w, 3) for w in raw)

    return SystemProfile(
        elements=elems,
        symbols=symbols,
        n_atoms_mean=float(np.mean(n_atoms_l)),
        n_atoms_max=int(np.max(n_atoms_l)),
        periodic=periodic,
        density_g_cm3=density,
        has_molecular_units=any_mol,
        n_molecules_mean=float(np.mean(mol_counts)),
        has_h_bonds=has_h_bonds,
        system_kind=system_kind,
        first_shell_r_A=r1,
        second_shell_r_A=r2,
        recommended_r_max=round(rec_r1, 2),
        recommended_r_max2=round(rec_r2, 2),
        force_rms=f_rms,
        per_species_force_rms=per_sp_rms,
        species_imbalance_ratio=imb,
        suggested_species_weights=sp_weights,
    )


def propose_candidates(
    profile: SystemProfile,
    speed_preset: str = "fast50",
    max_speed_us: float | None = None,
) -> list[StudentSpec]:
    """Generate physics-aware ``FastLRSKStudent`` candidates ordered from targeted to lean fallback."""
    target = SPEED_PRESET_US.get(speed_preset, 0.050) if max_speed_us is None else max_speed_us
    r1 = profile.recommended_r_max
    r2 = profile.recommended_r_max2
    wct = 18.0 if profile.has_molecular_units else 0.0
    heavy = (target is None) or (target >= 0.080)

    cands: list[StudentSpec] = []
    if profile.system_kind == "molecular_liquid":
        cands.extend([
            StudentSpec("wat_c220_r37_58_ro2x32", r_max=3.7, r_max2=5.8, lmax=2, lfeat=2, taper=(16, 16, 16), n_base=8, n_base2=8, c0=2, c1=2, c2=0, readout_depth=2, readout_width=32, radial_warp=3.0, wct=wct),
            StudentSpec("wat_c220_r37_58_ro1x64", r_max=3.7, r_max2=5.8, lmax=2, lfeat=2, taper=(16, 16, 16), n_base=8, n_base2=8, c0=2, c1=2, c2=0, readout_depth=1, readout_width=64, radial_warp=3.0, wct=wct),
            StudentSpec("wat_c220_r42_58_ro1x64", r_max=4.2, r_max2=5.8, lmax=2, lfeat=2, taper=(16, 16, 16), n_base=8, n_base2=8, c0=2, c1=2, c2=0, readout_depth=1, readout_width=64, radial_warp=3.0, wct=wct),
            StudentSpec("wat_c220_lean_ro1x32", r_max=3.7, r_max2=5.4, lmax=2, lfeat=2, taper=(16, 16, 16), n_base=8, n_base2=4, c0=2, c1=2, c2=0, readout_depth=1, readout_width=32, radial_warp=2.0, wct=wct),
        ])
        if heavy:
            cands.insert(0, StudentSpec("c441_w48", r_max=r1, r_max2=r2, lmax=3, lfeat=2, taper=(24, 16, 8, 8), n_base=8, n_base2=6, c0=4, c1=4, c2=1, readout_depth=2, readout_width=48, radial_warp=2.0, wct=wct))
    elif profile.system_kind == "ionic_crystal":
        cands.extend([
            StudentSpec("shared_c442_w48", r_max=r1, r_max2=r1, lmax=2, lfeat=2, taper=(16, 16, 16), n_base=8, n_base2=6, c0=4, c1=4, c2=2, readout_depth=2, readout_width=48, radial_warp=1.5),
            StudentSpec("shared_c440_l3_w48", r_max=min(r1, 5.2), r_max2=min(r1, 5.2), lmax=3, lfeat=2, taper=(16, 16, 12, 8), n_base=8, n_base2=6, c0=4, c1=4, c2=0, readout_depth=1, readout_width=48, radial_warp=1.5),
            StudentSpec("shared_c220_l2_w48", r_max=min(r1, 5.0), r_max2=min(r1, 5.0), lmax=2, lfeat=2, taper=(16, 16, 16), n_base=8, n_base2=6, c0=2, c1=2, c2=0, readout_depth=2, readout_width=48, radial_warp=1.5),
            StudentSpec("shared_c220_lean_w32", r_max=min(r1, 4.6), r_max2=min(r1, 4.6), lmax=2, lfeat=2, taper=(16, 8, 4), n_base=8, n_base2=4, c0=2, c1=2, c2=0, readout_depth=1, readout_width=32),
        ])
        if heavy:
            cands.insert(0, StudentSpec("shared_c442_hi", r_max=r1, r_max2=r1, lmax=3, lfeat=2, taper=(32, 16, 8, 8), n_base=8, n_base2=8, c0=4, c1=4, c2=2, readout_depth=2, readout_width=64, radial_warp=1.5))
    elif profile.system_kind == "isolated_molecule":
        cands.extend([
            StudentSpec("mol_l3_c440_w48", r_max=r1, r_max2=r1, lmax=3, lfeat=2, taper=(16, 16, 12, 8), n_base=8, n_base2=6, c0=4, c1=4, c2=0, readout_depth=2, readout_width=48, radial_warp=2.0, wct=wct),
            StudentSpec("mol_l3_c220_w32", r_max=r1, r_max2=r1, lmax=3, lfeat=2, taper=(16, 16, 12, 8), n_base=8, n_base2=6, c0=2, c1=2, c2=0, readout_depth=2, readout_width=32, radial_warp=2.0, wct=wct),
            StudentSpec("mol_l2_c440_w48", r_max=r1, r_max2=r1, lmax=2, lfeat=2, taper=(16, 16, 16), n_base=8, n_base2=4, c0=4, c1=4, c2=0, readout_depth=2, readout_width=48, radial_warp=1.5, wct=wct),
            StudentSpec("mol_l2_c220_w32", r_max=min(r1, 4.4), r_max2=min(r1, 4.4), lmax=2, lfeat=2, taper=(16, 8, 4), n_base=8, n_base2=4, c0=2, c1=2, c2=0, readout_depth=1, readout_width=32, wct=wct),
        ])
        if heavy:
            cands.insert(0, StudentSpec("mol_l3_c442_w64", r_max=r1, r_max2=r1, lmax=3, lfeat=2, taper=(32, 16, 8, 8), n_base=8, n_base2=8, c0=4, c1=4, c2=2, readout_depth=2, readout_width=64, radial_warp=2.0, wct=wct))
    else:
        cands.extend([
            StudentSpec("gen_l3_c220_w48", r_max=r1, r_max2=r2, lmax=3, lfeat=2, taper=(16, 16, 12, 8), n_base=8, n_base2=6, c0=2, c1=2, c2=0, readout_depth=2, readout_width=48, radial_warp=2.0, wct=wct),
            StudentSpec("gen_l2_c440_w48", r_max=r1, r_max2=r2, lmax=2, lfeat=2, taper=(16, 16, 16), n_base=8, n_base2=4, c0=4, c1=4, c2=0, readout_depth=2, readout_width=48, radial_warp=2.0, wct=wct),
            StudentSpec("gen_l2_c220_w48", r_max=min(r1, 4.5), r_max2=min(r2, 5.2), lmax=2, lfeat=2, taper=(16, 16, 16), n_base=8, n_base2=4, c0=2, c1=2, c2=0, readout_depth=2, readout_width=48, radial_warp=2.0, wct=wct),
            StudentSpec("gen_l2_c220_lean", r_max=min(r1, 4.2), r_max2=min(r2, 4.8), lmax=2, lfeat=2, taper=(16, 8, 4), n_base=8, n_base2=4, c0=2, c1=2, c2=0, readout_depth=1, readout_width=32, wct=wct),
        ])
        if heavy:
            cands.insert(0, StudentSpec("gen_l3_c442_w64", r_max=r1, r_max2=r2, lmax=3, lfeat=2, taper=(32, 16, 8, 8), n_base=8, n_base2=8, c0=4, c1=4, c2=2, readout_depth=2, readout_width=64, radial_warp=2.0, wct=wct))

    return cands


def _make_benchmark_supercell(atoms: Atoms, min_cut: float, target_atoms: int = 4400) -> Atoms:
    """Replicate ``atoms`` into a periodic ~3k-5k atom box suitable for steady-state speed benchmarking."""
    a = atoms.copy()
    if not all(bool(p) for p in a.pbc) or abs(float(np.linalg.det(a.cell.array))) < 1e-6:
        pos = a.positions - a.positions.mean(axis=0, keepdims=True)
        span = np.ptp(pos, axis=0) if len(pos) else np.zeros(3)
        pad = 2.0 * float(min_cut) + 4.0
        box = np.maximum(span + pad, pad)
        a.set_cell(np.diag(box))
        a.set_pbc((True, True, True))
        a.set_positions(pos + 0.5 * box[None, :])

    # Ensure box lengths are >= 2 * min_cut + 1.0 before measuring
    lengths = np.linalg.norm(a.cell.array, axis=1)
    rep = [max(1, int(math.ceil((2.0 * min_cut + 1.0) / max(1e-3, L)))) for L in lengths]
    big = a.repeat(rep)
    while len(big) < target_atoms // 2:
        # Repeat along shortest box vector
        lens = np.linalg.norm(big.cell.array, axis=1)
        ax = int(np.argmin(lens))
        r = [1, 1, 1]
        r[ax] = 2
        big = big.repeat(r)
        if len(big) > target_atoms * 2:
            break
    return big


def build_student_from_spec(spec: StudentSpec, profile: SystemProfile) -> "FastLRSKStudent":
    from .fast_lrsk import FastLRSKStudent

    return FastLRSKStudent(
        zs=profile.elements,
        r_max=spec.r_max,
        r_max2=spec.r_max2,
        lmax=spec.lmax,
        lfeat=spec.lfeat,
        taper=spec.taper,
        n_base=spec.n_base,
        n_base2=spec.n_base2,
        c0=spec.c0,
        c1=spec.c1,
        c2=spec.c2,
        readout_depth=spec.readout_depth,
        readout_width=spec.readout_width,
        pol_gram=spec.pol_gram,
        pol_field=spec.pol_field,
        field_mlp_width=spec.field_mlp_width,
        radial_warp=spec.radial_warp,
        use_zbl_switch=spec.use_zbl_switch,
        film_radials=spec.film_radials,
        kappa=spec.kappa,
        a0=spec.a0,
        z_c=spec.z_c,
        energy_scale=max(0.1, profile.force_rms),
        max_order=spec.max_order,
        sw_mp=spec.sw_mp,
        mp_cap=spec.mp_cap,
        ef_cap=spec.ef_cap,
    )


def benchmark_spec_speed(
    spec: StudentSpec,
    profile: SystemProfile,
    reference_atoms: Atoms,
    work_dir: Path,
    device: int = 0,
    steps_per_burst: int = 500,
    n_bursts: int = 3,
) -> float:
    """Export ``spec`` to a scratch ``.loup`` bundle and measure warmed steady-state GPU us/atom."""
    import gc
    import torch
    from loupmd import _core
    from loupmd.export.ilmlip import export_ilmlip_checkpoint
    from .fast_lrsk import save_student_checkpoint

    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_p = work_dir / f"_bench_{spec.name}.pt"
    bundle_p = work_dir / f"_bench_{spec.name}.loup"

    model = build_student_from_spec(spec, profile)
    with torch.no_grad():
        model.u_Z.zero_()
        model.W_lr_rad.zero_()
        if model.pol_field and getattr(model, "pol_u", None) is not None:
            model.pol_u.zero_()
    save_student_checkpoint(model, ckpt_p)
    export_ilmlip_checkpoint(ckpt_p, bundle_p, energy_scale=float(model.energy_scale))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    big = _make_benchmark_supercell(reference_atoms, max(spec.r_max, spec.r_max2))
    eng = _core.LoupEngine(bundle_p, device_id=device)
    try:
        eng.set_system(big.numbers, big.get_masses(), skin=0.5)
        eng.set_state(big.positions, np.zeros_like(big.positions), big.cell.array)
        # Warm-up (includes GEMM/kernel autotune + list build) with tiny dt so atoms stay at physical density
        eng.run_nve(steps_per_burst, dt_fs=1e-6)
        step_ms_samples = []
        for _ in range(n_bursts):
            st = eng.run_nve(steps_per_burst, dt_fs=1e-6)
            step_ms_samples.append(float(st.step_ms))
        med_ms = float(np.median(step_ms_samples))
        return (med_ms * 1000.0) / max(1, len(big))
    finally:
        del eng
        gc.collect()
        ckpt_p.unlink(missing_ok=True)
        bundle_p.unlink(missing_ok=True)


def _trim_spec(spec: StudentSpec, step: int) -> StudentSpec:
    if step == 1 and spec.c2 > 0:
        return replace(spec, name=spec.name + "_nc2", c2=0)
    if step == 2 and (spec.c0 > 2 or spec.c1 > 2):
        return replace(spec, name=spec.name + "_c22", c0=min(spec.c0, 2), c1=min(spec.c1, 2), c2=0)
    if step == 3 and spec.lmax > 2:
        return replace(spec, name=spec.name + "_l2", lmax=2, taper=spec.taper[:3])
    if step == 4 and (spec.r_max > 3.8 or spec.r_max2 > 5.2):
        return replace(
            spec,
            name=spec.name + "_rtight",
            lmax=2,
            taper=(16, 8, 4),
            c0=2,
            c1=2,
            c2=0,
            n_base2=4,
            readout_depth=1,
            readout_width=min(spec.readout_width, 32),
            r_max=min(spec.r_max, 3.8),
            r_max2=min(spec.r_max2, 5.0),
        )
    return replace(
        spec,
        name=spec.name + f"_trim{step}",
        lmax=2,
        taper=(16, 8, 4),
        c0=2,
        c1=2,
        c2=0,
        n_base2=4,
        readout_depth=1,
        readout_width=32,
        pol_field=(step < 5),
        pol_gram=False,
        r_max=min(spec.r_max, 3.6),
        r_max2=min(spec.r_max2, 4.8),
    )


def select_speed_feasible_candidates(
    candidates: Sequence[StudentSpec],
    profile: SystemProfile,
    reference_atoms: Atoms,
    speed_target_us: float | None,
    work_dir: Path,
    device: int = 0,
    log_fn=None,
) -> list[tuple[StudentSpec, float]]:
    """Benchmark candidates on the actual GPU and return those meeting ``speed_target_us``."""
    measured: list[tuple[StudentSpec, float]] = []
    seen_keys: set[tuple] = set()

    def _key(s: StudentSpec) -> tuple:
        return (s.r_max, s.r_max2, s.lmax, s.lfeat, s.taper, s.n_base, s.n_base2, s.c0, s.c1, s.c2, s.readout_depth, s.readout_width)

    for sp in candidates:
        k = _key(sp)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        us_at = benchmark_spec_speed(sp, profile, reference_atoms, work_dir, device=device)
        measured.append((sp, us_at))
        if log_fn is not None:
            log_fn(f"  [speed] {sp.name:24s} -> {us_at:.4f} us/at")

    if speed_target_us is None:
        return measured

    tol = speed_target_us * 1.05
    feasible = [(sp, us) for sp, us in measured if us <= tol]

    # Auto-trim progressively if fewer than 2 candidates clear the target budget
    if len(feasible) < 2 and measured:
        fastest_spec = min(measured, key=lambda x: x[1])[0]
        for step in (1, 2, 3, 4, 5):
            trimmed = _trim_spec(fastest_spec, step)
            tk = _key(trimmed)
            if tk in seen_keys:
                continue
            seen_keys.add(tk)
            us_at = benchmark_spec_speed(trimmed, profile, reference_atoms, work_dir, device=device)
            measured.append((trimmed, us_at))
            if log_fn is not None:
                log_fn(f"  [speed-trim] {trimmed.name:24s} -> {us_at:.4f} us/at")
            if us_at <= tol:
                feasible.append((trimmed, us_at))
                if len(feasible) >= 2:
                    break

    if feasible:
        feasible.sort(key=lambda x: x[1], reverse=True)  # Highest capacity within speed budget first
        return feasible
    measured.sort(key=lambda x: x[1])
    return measured[:2]
