"""End-to-end Auto-Distiller orchestrator: data prep -> analysis -> GPU speed check -> CUDA screen/train -> verify."""
from __future__ import annotations

import dataclasses
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from ase import Atoms

from loupmd import _core, models
from loupmd.export.ilmlip import export_ilmlip_checkpoint
from .analyze import (
    SPEED_PRESET_US,
    StudentSpec,
    SystemProfile,
    _make_benchmark_supercell,
    analyze_system,
    build_student_from_spec,
    propose_candidates,
    select_speed_feasible_candidates,
)
from .cuda_trainer import calibrate_e0_lstsq, eval_model_splits, train_student_cuda
from ..data import _pad_nonperiodic_for_engine, label_frames_with_teacher, prepare_datasets, align_teacher_to_dft
from .fast_lrsk import batch_graph_items, build_graph_items, save_student_checkpoint

BUDGET_PRESETS: dict[str, dict] = {
    "quick": {
        "n_md_frames": 48,
        "temps_k": (300.0, 500.0),
        "screen_epochs": 30,
        "top_k_screen": 2,
        "train_epochs": 100,
        "active_rounds": 0,
        "batch_size": 4,
    },
    "standard": {
        "n_md_frames": 96,
        "temps_k": (250.0, 450.0, 700.0),
        "screen_epochs": 45,
        "top_k_screen": 3,
        "train_epochs": 200,
        "active_rounds": 1,
        "batch_size": 4,
    },
    "thorough": {
        "n_md_frames": 160,
        "temps_k": (200.0, 400.0, 650.0, 900.0),
        "screen_epochs": 60,
        "top_k_screen": 4,
        "train_epochs": 320,
        "active_rounds": 1,
        "batch_size": 4,
    },
}


@dataclasses.dataclass
class DistillResult:
    bundle_path: Path
    checkpoint_path: Path
    report_path: Path
    spec: StudentSpec
    profile: SystemProfile
    test_f_mev: float
    test_e_mev: float
    speed_us_per_atom: float
    gate_passed: bool
    md_parity_passed: bool
    report: dict


def _verify_bundle_gate_and_md(
    model,
    bundle_path: Path,
    test_frames: Sequence[Atoms],
    profile: SystemProfile,
    device: int = 0,
) -> tuple[bool, bool, float, dict]:
    """Verify 3x FP32-vs-FP64 export gate, 50-step NVE MD parity, and warmed 3x500 speed."""
    from .fast_lrsk import FastLRSKStudent

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    eng = _core.LoupEngine(bundle_path, device_id=device)
    model_f64 = FastLRSKStudent(
        zs=profile.elements, r_max=model.r_max, r_max2=model.r_max2, lmax=model.lmax, lfeat=model.lfeat,
        taper=model.n_l, n_base=model.n_base, n_base2=model.n_base2, c0=model.c0, c1=model.c1, c2=model.c2,
        readout_width=int(model.readout_layers[0].out_features), readout_depth=len(model.readout_layers),
        radial_warp=model.radial_warp, use_zbl_switch=model.use_zbl_switch, kappa=model.kappa, a0=model.a0,
        energy_scale=model.energy_scale, z_c=model.z_c, max_order=model.max_order,
        film_radials=model.film_radials, pol_field=model.pol_field, field_mlp_width=model.field_mlp_width,
        sym=model.sym, pol_gram=model.pol_gram, sw_mp=model.sw_mp, mp_cap=model.mp_cap, ef_cap=model.ef_cap,
    )
    model_f64.load_state_dict({k: v.detach().cpu().double().clone() for k, v in model.state_dict().items()})
    model_f64 = model_f64.double().to(f"cuda:{device}").eval()

    model_f32 = FastLRSKStudent(
        zs=profile.elements, r_max=model.r_max, r_max2=model.r_max2, lmax=model.lmax, lfeat=model.lfeat,
        taper=model.n_l, n_base=model.n_base, n_base2=model.n_base2, c0=model.c0, c1=model.c1, c2=model.c2,
        readout_width=int(model.readout_layers[0].out_features), readout_depth=len(model.readout_layers),
        radial_warp=model.radial_warp, use_zbl_switch=model.use_zbl_switch, kappa=model.kappa, a0=model.a0,
        energy_scale=model.energy_scale, z_c=model.z_c, max_order=model.max_order,
        film_radials=model.film_radials, pol_field=model.pol_field, field_mlp_width=model.field_mlp_width,
        sym=model.sym, pol_gram=model.pol_gram, sw_mp=model.sw_mp, mp_cap=model.mp_cap, ef_cap=model.ef_cap,
    )
    model_f32.load_state_dict({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model_f32 = model_f32.float()
    model_f32.e0 = model_f32.e0.double()
    model_f32 = model_f32.to(f"cuda:{device}").eval()

    probe_frames = list(test_frames[: min(4, len(test_frames))])
    de_list, df_list, ds_list = [], [], []
    de32_list, df32_list, ds32_list = [], [], []
    r_cut = float(max(model.r_max, model.r_max2))

    try:
        for at in probe_frames:
            is_per = all(bool(p) for p in at.pbc) and abs(float(np.linalg.det(at.cell.array))) > 1e-6
            boxed, _ = _pad_nonperiodic_for_engine(at, r_cut + 2.0)
            eng.set_system(boxed.numbers, boxed.get_masses(), skin=0.5)
            en_loup, f_loup, s_loup = eng.evaluate(boxed.positions, boxed.cell.array, compute_stress=is_per)

            items_p, _, _ = build_graph_items([boxed], profile.elements, r_cut)
            b64 = batch_graph_items(items_p, device=f"cuda:{device}", dtype=torch.float64)
            out64 = model_f64(b64, training=False, compute_force=True, compute_stress=is_per)
            en64 = float(out64["energy"][0].item())
            f64 = out64["forces"].detach().cpu().numpy()
            s64 = out64["stress"][0].detach().cpu().numpy() if is_per else None

            b32 = batch_graph_items(items_p, device=f"cuda:{device}", dtype=torch.float32)
            out32 = model_f32(b32, training=False, compute_force=True, compute_stress=is_per)
            en32 = float(out32["energy"][0].item())
            f32 = out32["forces"].detach().cpu().numpy()
            s32 = out32["stress"][0].detach().cpu().numpy() if is_per else None

            N = len(boxed)
            de_list.append(abs(en_loup - en64) / N * 1000.0)
            de32_list.append(abs(en32 - en64) / N * 1000.0)
            df_list.append(float(np.max(np.abs(f_loup - f64))))
            df32_list.append(float(np.max(np.abs(f32 - f64))))
            if is_per and s_loup is not None and s64 is not None and s32 is not None:
                ds_list.append(float(np.max(np.abs(s_loup - s64))))
                ds32_list.append(float(np.max(np.abs(s32 - s64))))

        dE = max(de_list) if de_list else 0.0
        dE32 = max(de32_list) if de32_list else 0.0
        dF = max(df_list) if df_list else 0.0
        dF32 = max(df32_list) if df32_list else 0.0
        dS = max(ds_list) if ds_list else 0.0
        dS32 = max(ds32_list) if ds32_list else 0.0

        tol_E = max(3.0 * dE32, 0.01)
        tol_F = max(3.0 * dF32, 1e-5)
        tol_S = max(3.0 * dS32, 1e-7)
        gate_ok = (dE <= tol_E) and (dF <= tol_F) and (not ds_list or dS <= tol_S)

        # 50-step burst NVE parity on test_frames[0]
        at0 = test_frames[0]
        boxed0, _ = _pad_nonperiodic_for_engine(at0, r_cut + 2.0)
        eng.set_system(boxed0.numbers, boxed0.get_masses(), skin=0.5)
        eng.set_state(boxed0.positions, np.zeros_like(boxed0.positions), boxed0.cell.array)
        eng.run_nve(50, dt_fs=0.25)
        pos_end, _, f_end, e_end = eng.get_state()
        at_end = boxed0.copy()
        at_end.set_positions(pos_end)
        items_e, _, _ = build_graph_items([at_end], profile.elements, r_cut)
        b_end = batch_graph_items(items_e, device=f"cuda:{device}", dtype=torch.float64)
        out_end = model_f64(b_end, training=False, compute_force=True, compute_stress=False)
        dF_md = float(np.max(np.abs(f_end - out_end["forces"].detach().cpu().numpy())))
        dE_md = abs(float(e_end) - float(out_end["energy"][0].item())) / len(at_end) * 1000.0
        md_ok = np.isfinite(e_end) and np.all(np.isfinite(f_end)) and (dF_md <= max(2.0 * tol_F, 5e-4))

        # Warmed 3x500 speed measurement on supercell
        big = _make_benchmark_supercell(at0, r_cut)
        eng.set_system(big.numbers, big.get_masses(), skin=0.5)
        eng.set_state(big.positions, np.zeros_like(big.positions), big.cell.array)
        eng.run_nve(500, dt_fs=1e-6)
        bursts = [float(eng.run_nve(500, dt_fs=1e-6).step_ms) for _ in range(3)]
        med_ms = float(np.median(bursts))
        us_per_atom = (med_ms * 1000.0) / max(1, len(big))

        metrics = {
            "dE_mev_at": round(dE, 6),
            "tol_E_mev_at": round(tol_E, 6),
            "dF_max_ev_a": round(dF, 7),
            "tol_F_max_ev_a": round(tol_F, 7),
            "dS_max_ev_a3": round(dS, 8) if ds_list else None,
            "tol_S_max_ev_a3": round(tol_S, 8) if ds_list else None,
            "md_burst_dF_ev_a": round(dF_md, 7),
            "md_burst_dE_mev_at": round(dE_md, 6),
            "speed_n_atoms": len(big),
            "speed_ms_per_step": round(med_ms, 5),
            "speed_us_per_atom": round(us_per_atom, 5),
        }
        return bool(gate_ok), bool(md_ok), float(us_per_atom), metrics
    finally:
        del eng, model_f32
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def auto_distill(
    *,
    teacher: str | Path | None = None,
    head: str | None = None,
    checkpoint: str | Path | None = None,
    sha256: str | None = None,
    trusted: bool = False,
    structure: str | Path | Atoms | Sequence | None = None,
    md_data: str | Path | Sequence | None = None,
    dft_data: str | Path | Sequence | None = None,
    output: str | Path = "./distill_out",
    speed: str = "fast50",
    max_speed_us: float | None = None,
    budget: str = "standard",
    epochs: int | None = None,
    n_md_frames: int | None = None,
    temps_k: Sequence[float] | None = None,
    no_teacher_aug: bool = False,
    dft_weight: float = 3.0,
    dispersion: str | None = None,
    dispersion_functional: str = "pbe",
    energy_key: str | None = None,
    forces_key: str | None = None,
    stress_key: str | None = None,
    device: int = 0,
    seed: int = 42,
    verbose: bool = True,
) -> DistillResult:
    """Automatically analyze a system, select a speed-compliant student, train via pure CUDA, and verify."""
    t_start = time.perf_counter()
    if budget not in BUDGET_PRESETS:
        raise ValueError(f"Unknown budget {budget!r}; choose from {list(BUDGET_PRESETS)}")
    if speed not in SPEED_PRESET_US and max_speed_us is None:
        raise ValueError(f"Unknown speed preset {speed!r}; choose from {list(SPEED_PRESET_US)}")

    out_p = Path(output)
    if out_p.suffix == ".loup":
        bundle_out = out_p
        out_dir = out_p.parent
    else:
        out_dir = out_p
        bundle_out = out_dir / "student.loup"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = out_dir / "_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    bcfg = BUDGET_PRESETS[budget]
    target_speed = max_speed_us if max_speed_us is not None else SPEED_PRESET_US[speed]
    dev_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    def log(msg: str):
        if verbose:
            print(msg, flush=True)

    teacher_ff = None
    if teacher is not None:
        from loupmd.forcefield import ForceField
        log(f"[1/6] Loading teacher {teacher!r}...")
        resolved, digest, _ = models.trusted_checkpoint(
            str(teacher), head=head, checkpoint=checkpoint, sha256=sha256, trusted=trusted
        )
        teacher_ff = ForceField(resolved, digest, head=head, device=device)

    log("[2/6] Preparing & normalizing datasets...")
    prep = prepare_datasets(
        teacher_ff=teacher_ff,
        structure=structure,
        md_data=md_data,
        dft_data=dft_data,
        n_md_frames=int(n_md_frames or bcfg["n_md_frames"]),
        temps_k=tuple(temps_k or bcfg["temps_k"]),
        no_teacher_aug=no_teacher_aug,
        dft_weight=dft_weight,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        seed=seed,
        log_fn=log,
    )
    if teacher_ff is not None:
        teacher_ff.engine = None
        teacher_ff._system = None
        gc.collect()

    log("[3/6] Analyzing system chemistry, geometry, and force scales...")
    profile = analyze_system(prep.train_frames + prep.val_frames, elements=prep.elements)
    log(
        f"  System kind={profile.system_kind} elements={list(profile.symbols)} "
        f"periodic={profile.periodic} mol={profile.has_molecular_units} "
        f"r_cut=({profile.recommended_r_max}, {profile.recommended_r_max2}) "
        f"f_rms={profile.force_rms:.3f} imbalance={profile.species_imbalance_ratio:.2f}x"
    )

    log(f"[4/6] Benchmarking candidate architectures (target={target_speed} us/at)...")
    raw_cands = propose_candidates(profile, speed_preset=speed, max_speed_us=max_speed_us)
    feasible = select_speed_feasible_candidates(
        raw_cands, profile, prep.val_frames[0], target_speed, scratch_dir, device=device, log_fn=log
    )

    to_screen = feasible[: int(bcfg["top_k_screen"])]
    screen_results: list[tuple[StudentSpec, float, float]] = []
    if len(to_screen) > 1 and int(bcfg["screen_epochs"]) > 0:
        log(f"[5/6] Screening top-{len(to_screen)} candidates ({bcfg['screen_epochs']} CUDA epochs)...")
        for idx, (sp, us_at) in enumerate(to_screen):
            m = build_student_from_spec(sp, profile)
            res = train_student_cuda(
                m, prep.train_frames, prep.val_frames, profile.elements,
                epochs=int(bcfg["screen_epochs"]),
                batch_size=int(bcfg["batch_size"]),
                peak_lr=0.008, min_lr=2e-4,
                eval_every=max(5, int(bcfg["screen_epochs"]) // 3),
                wct=sp.wct,
                sp_w=profile.suggested_species_weights,
                device=dev_str, seed=seed + idx,
            )
            val_f = res["best_val_f_mev"]
            screen_results.append((sp, val_f, us_at))
            log(f"  [screen] {sp.name:24s} val_F={val_f:6.2f} meV/A ({us_at:.4f} us/at, {res['train_s']:.1f}s)")
            del m, res
            gc.collect()
            torch.cuda.empty_cache()
        screen_results.sort(key=lambda x: (round(x[1], 3), x[2]))
        winner_spec = screen_results[0][0]
    else:
        winner_spec = to_screen[0][0]
        screen_results.append((winner_spec, 0.0, to_screen[0][1]))

    n_train_ep = int(epochs or bcfg["train_epochs"])
    log(f"  Selected winner: {winner_spec.name}; training {n_train_ep} CUDA epochs...")
    winner_model = build_student_from_spec(winner_spec, profile)
    train_res = train_student_cuda(
        winner_model, prep.train_frames, prep.val_frames, profile.elements,
        epochs=n_train_ep,
        batch_size=int(bcfg["batch_size"]),
        peak_lr=0.008, min_lr=1e-4,
        eval_every=max(10, n_train_ep // 10),
        wct=winner_spec.wct,
        sp_w=profile.suggested_species_weights,
        device=dev_str, seed=seed,
        log_fn=log,
    )

    # Optional 1-round student-MD active learning against teacher
    if int(bcfg["active_rounds"]) >= 1 and teacher_ff is not None:
        al_ckpt = scratch_dir / "al_round.pt"
        al_loup = scratch_dir / "al_round.loup"
        save_student_checkpoint(winner_model, al_ckpt)
        export_ilmlip_checkpoint(al_ckpt, al_loup, energy_scale=float(winner_model.energy_scale))
        eng_s = _core.LoupEngine(al_loup, device_id=device)
        seed_at = prep.val_frames[0]
        boxed_s, was_np = _pad_nonperiodic_for_engine(seed_at, float(max(winner_spec.r_max, winner_spec.r_max2)) + 2.0)
        eng_s.set_system(boxed_s.numbers, boxed_s.get_masses(), skin=0.5)
        eng_s.set_state(boxed_s.positions, np.zeros_like(boxed_s.positions), boxed_s.cell.array)
        al_snaps: list[Atoms] = []
        for k in range(12):
            eng_s.run_langevin(40, dt_fs=0.5, temp_k=450.0, friction_per_fs=0.02, seed=seed + 500 + k)
            p_k, _, _, _ = eng_s.get_state()
            snap = boxed_s.copy()
            snap.set_positions(p_k)
            if was_np:
                snap.set_pbc((False, False, False))
                snap.set_cell(seed_at.cell.array.copy())
            al_snaps.append(snap)
        del eng_s
        gc.collect()

        al_labeled = label_frames_with_teacher(al_snaps, teacher_ff, per_atom=prep.has_ei, weight=2.0)
        teacher_ff.engine = None
        teacher_ff._system = None
        if prep.dft_train_frames:
            al_labeled, _ = align_teacher_to_dft(al_labeled, prep.dft_train_frames, teacher_ff, profile.elements)
        r_graph_w = float(max(winner_spec.r_max, winner_spec.r_max2))
        al_items, _, _ = build_graph_items(al_labeled, profile.elements, r_graph_w)
        al_sc = eval_model_splits(winner_model, al_items, profile.elements, device=dev_str)
        if al_sc["f_mev"] > max(20.0, 1.35 * train_res["best_val_f_mev"]):
            log(f"  [active-learning] Student-MD shift detected (F={al_sc['f_mev']:.2f} vs val={train_res['best_val_f_mev']:.2f} meV/A); polishing 40 epochs...")
            aug_train = list(prep.train_frames) + al_labeled
            train_student_cuda(
                winner_model, aug_train, prep.val_frames, profile.elements,
                epochs=max(40, n_train_ep // 4),
                batch_size=int(bcfg["batch_size"]),
                peak_lr=0.003, min_lr=1e-4, eval_every=10,
                wct=winner_spec.wct, sp_w=profile.suggested_species_weights,
                device=dev_str, seed=seed + 77,
            )

    # Calibrate E0 exclusively on DFT if DFT ground truth exists
    r_graph_final = float(max(winner_spec.r_max, winner_spec.r_max2))
    if prep.dft_train_frames:
        dft_items, _, _ = build_graph_items(prep.dft_train_frames, profile.elements, r_graph_final)
        calibrate_e0_lstsq(winner_model, dft_items, profile.elements, device=dev_str)

    te_items, _, te_stress = build_graph_items(prep.test_frames, profile.elements, r_graph_final)
    test_sc = eval_model_splits(winner_model, te_items, profile.elements, device=dev_str, has_stress=te_stress)

    disp_spec = None
    if dispersion and str(dispersion).lower() not in ("off", "none"):
        from ..dispersion_d3 import fit_fixed_d3bj

        use_tail = bool(profile.is_periodic and profile.system_kind != "isolated_molecule")
        disp_spec = fit_fixed_d3bj(
            prep.train_frames[: min(16, len(prep.train_frames))],
            profile.elements,
            r_cut=r_graph_final,
            functional=dispersion_functional,
            force_shift=True,
            use_tail=use_tail,
        )
        log(f"  [dispersion] Built-in force-shifted D3(BJ) ({dispersion_functional}, r_cut={r_graph_final:.2f} A, tail={use_tail})")

    ckpt_out = out_dir / "student.pt"
    save_student_checkpoint(winner_model, ckpt_out, extra_args={"distill_teacher": str(teacher) if teacher else None})
    export_ilmlip_checkpoint(ckpt_out, bundle_out, energy_scale=float(winner_model.energy_scale), dispersion=disp_spec)

    log(f"[6/6] Verifying exported bundle {bundle_out.name} (3x gate + 50-step MD parity + speed)...")
    gate_ok, md_ok, speed_us, ver_metrics = _verify_bundle_gate_and_md(
        winner_model, bundle_out, prep.test_frames, profile, device=device
    )

    total_s = time.perf_counter() - t_start
    report = {
        "teacher": str(teacher) if teacher else None,
        "budget": budget,
        "speed_preset": speed,
        "speed_target_us_per_atom": target_speed,
        "profile": dataclasses.asdict(profile),
        "winner_spec": dataclasses.asdict(winner_spec),
        "screen_results": [{"name": s.name, "val_f_mev": round(vf, 3), "us_per_atom": round(u, 5)} for s, vf, u in screen_results],
        "e0_alignment": prep.e0_offset_summary,
        "dataset_counts": {
            "train": len(prep.train_frames),
            "val": len(prep.val_frames),
            "test": len(prep.test_frames),
            "dft_train": len(prep.dft_train_frames),
        },
        "test_metrics": test_sc,
        "verification": {**ver_metrics, "gate_passed": gate_ok, "md_parity_passed": md_ok},
        "wall_time_s": round(total_s, 2),
        "bundle_path": str(bundle_out),
        "checkpoint_path": str(ckpt_out),
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    shutil.rmtree(scratch_dir, ignore_errors=True)
    log(
        f"Done in {total_s:.1f}s -> {bundle_out} | "
        f"test F={test_sc['f_mev']:.2f} meV/A E={test_sc['e_mev']:.3f} meV/at | "
        f"speed={speed_us:.4f} us/at | gate={'PASS' if gate_ok else 'FAIL'} md={'PASS' if md_ok else 'FAIL'}"
    )
    return DistillResult(
        bundle_path=bundle_out,
        checkpoint_path=ckpt_out,
        report_path=report_path,
        spec=winner_spec,
        profile=profile,
        test_f_mev=float(test_sc["f_mev"]),
        test_e_mev=float(test_sc["e_mev"]),
        speed_us_per_atom=float(speed_us),
        gate_passed=gate_ok,
        md_parity_passed=md_ok,
        report=report,
    )
