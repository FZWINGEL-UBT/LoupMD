"""Universal fine-tuning engine for all LoupMD foundation model families."""
from __future__ import annotations

import copy
import dataclasses
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms

from loupmd import _core, models
from loupmd.errors import InputError, ModelElementError
from ..data import _pad_nonperiodic_for_engine, load_frames_input, normalize_frame


@dataclasses.dataclass
class FinetuneResult:
    model_path: Path
    bundle_path: Path | None
    report_path: Path
    base_model: str
    model_class: str
    head: str | None
    strategy: str
    elements: tuple[int, ...]
    zero_shot_metrics: dict
    test_metrics: dict
    gate_passed: bool | None
    wall_time_s: float
    report: dict


def _resolve_head(model_obj, requested_head: str | None, resolved) -> tuple[list[str], str, int, str | None]:
    raw_heads = list(getattr(model_obj, "heads", None) or [])
    if not raw_heads:
        return ["Default"], "Default", 0, None
    if len(raw_heads) == 1:
        if requested_head is not None and requested_head != raw_heads[0]:
            raise InputError(f"model has head {raw_heads[0]!r}, not {requested_head!r}")
        return raw_heads, raw_heads[0], 0, None
    lbl_head = resolved.label.head if resolved.label is not None else None
    active = requested_head if requested_head is not None else lbl_head
    if active is None:
        raise InputError(
            f"{resolved.path.name} has multiple heads and no default",
            next_step=f"Pass --head / head= one of {', '.join(raw_heads)}.",
        )
    if active not in raw_heads:
        raise InputError(
            f"{resolved.path.name} has no head {active!r}",
            next_step=f"Available heads: {', '.join(raw_heads)}.",
        )
    return raw_heads, active, raw_heads.index(active), active


def _split_frames(frames: list[Atoms], valid_frames: list[Atoms], test_frames: list[Atoms], seed: int):
    rng = np.random.default_rng(seed)
    tr = list(frames)
    va = list(valid_frames)
    te = list(test_frames)
    if not va and not te:
        n = len(tr)
        if n < 3:
            return tr, tr[:1], tr[:1]
        perm = rng.permutation(n)
        n_te = max(1, int(round(0.10 * n))) if n >= 8 else 1
        n_va = max(1, int(round(0.15 * n))) if n >= 5 else 1
        n_tr = max(1, n - n_va - n_te)
        return [tr[i] for i in perm[:n_tr]], [tr[i] for i in perm[n_tr : n_tr + n_va]], [tr[i] for i in perm[n_tr + n_va :]]
    if not va:
        n = len(tr)
        if n >= 5:
            perm = rng.permutation(n)
            n_va = max(1, int(round(0.15 * n)))
            tr, va = [tr[i] for i in perm[n_va:]], [tr[i] for i in perm[:n_va]]
        else:
            va = tr[:1]
    if not te:
        te = list(va)
    return tr, va, te


def _build_atomic_data(
    frames: Sequence[Atoms],
    *,
    z_table,
    cutoff: float,
    heads: list[str],
    head_name: str,
    dtype: torch.dtype,
    charge: int | None,
    spin_multiplicity: int | None,
    has_joint_emb: bool,
) -> list:
    import torch
    from mace.data import AtomicData, KeySpecification, config_from_atoms
    from mace.tools import torch_tools

    keyspec = KeySpecification(
        info_keys={
            "energy": "REF_energy",
            "stress": "REF_stress",
            "total_charge": "charge",
            "total_spin": "spin",
            "external_field": "external_field",
        },
        arrays_keys={"forces": "REF_forces"},
    )
    items = []
    dt_str = "float64" if dtype == torch.float64 else "float32"
    with torch_tools.default_dtype(dt_str):
        for at in frames:
            a = at.copy()
            a.calc = None
            if charge is not None and "charge" not in a.info:
                a.info["charge"] = int(charge)
            elif has_joint_emb and "charge" not in a.info:
                a.info["charge"] = 0
            if spin_multiplicity is not None and "spin" not in a.info:
                a.info["spin"] = int(spin_multiplicity)
            elif has_joint_emb and "spin" not in a.info:
                a.info["spin"] = 1
            cfg = config_from_atoms(a, key_specification=keyspec, head_name=head_name)
            ad = AtomicData.from_config(cfg, z_table=z_table, cutoff=float(cutoff), heads=heads)
            ad._has_stress = bool(all(bool(p) for p in a.pbc) and "REF_stress" in a.info)
            ad._sample_weight = float(a.info.get("distill_weight", 1.0))
            items.append(ad)
    return items


def _set_model_dtype(model, dtype):
    import torch
    with torch.no_grad():
        model.to(dtype=dtype)
    return model


def _forward_model(model: torch.nn.Module, batch_dict: dict, *, training: bool, compute_stress: bool) -> dict:
    try:
        return model(batch_dict, training=training, compute_force=True, compute_stress=compute_stress)
    except TypeError:
        return model(batch_dict, training=training, compute_force=True)


def _eval_split(
    model: torch.nn.Module,
    items: Sequence,
    device: torch.device,
    batch_size: int = 4,
) -> dict:
    import torch
    from mace.tools.torch_geometric import Batch

    if not items:
        return {"f_mev": 0.0, "e_mev": 0.0, "s_gpa": None}
    model.eval()
    f_sq, e_mev = [], []
    s_gpa_sq = []
    for s0 in range(0, len(items), batch_size):
        sub = items[s0 : s0 + batch_size]
        can_stress = (type(model).__name__ != "PolarMACE") and all(bool(getattr(it, "_has_stress", False)) for it in sub)
        b = Batch.from_data_list(sub).to(device).to_dict()
        out = _forward_model(model, b, training=False, compute_stress=can_stress)
        pred_e = out["energy"].detach().cpu().double().numpy()
        pred_f = out["forces"].detach().cpu().double().numpy()
        pred_s = out.get("stress", None)
        if can_stress and pred_s is not None:
            ps_np = pred_s.detach().cpu().double().numpy()
        else:
            ps_np = None

        off = 0
        for g, it in enumerate(sub):
            n = int(it.positions.shape[0])
            rf = it.forces.detach().cpu().double().numpy()
            df = pred_f[off : off + n] - rf
            f_sq.append((df ** 2).ravel())
            re = float(it.energy.reshape(-1)[0].item())
            e_mev.append(((float(pred_e[g]) - re) / max(1, n)) * 1000.0)
            if can_stress and ps_np is not None:
                rs = it.stress.detach().cpu().double().numpy().reshape(3, 3)
                # eV/A^3 -> GPa (1 eV/A^3 = 160.2176634 GPa)
                s_gpa_sq.append((((ps_np[g] - rs) * 160.2176634) ** 2).ravel())
            off += n
        n_batch_atoms = sum(int(it.positions.shape[0]) for it in sub)
        del out, b
        if device.type == "cuda" and (type(model).__name__ == "PolarMACE" or n_batch_atoms >= 300):
            torch.cuda.empty_cache()

    f_rmse = float(np.sqrt(np.mean(np.concatenate(f_sq))) * 1000.0) if f_sq else 0.0
    e_rmse = float(np.sqrt(np.mean(np.square(e_mev)))) if e_mev else 0.0
    s_rmse = float(np.sqrt(np.mean(np.concatenate(s_gpa_sq)))) if s_gpa_sq else None
    return {"f_mev": f_rmse, "e_mev": e_rmse, "s_gpa": s_rmse}


def _fit_active_head_e0(
    model: torch.nn.Module,
    items: Sequence,
    all_z: list[int],
    active_z: Sequence[int],
    head_idx: int,
    device: torch.device,
    batch_size: int = 4,
) -> None:
    """Shift only active species in ``head_idx`` row of ``atomic_energies`` by least-squares residual."""
    import torch
    from mace.tools.torch_geometric import Batch

    if not items or not hasattr(model, "atomic_energies_fn"):
        return
    model.eval()
    cols = [all_z.index(int(z)) for z in active_z]
    z_to_col = {col: j for j, col in enumerate(cols)}
    S = len(cols)
    A = np.zeros((len(items), S), dtype=np.float64)
    y = np.zeros(len(items), dtype=np.float64)

    for s0 in range(0, len(items), batch_size):
        sub = items[s0 : s0 + batch_size]
        b = Batch.from_data_list(sub).to(device).to_dict()
        out = _forward_model(model, b, training=False, compute_stress=False)
        pred_e = out["energy"].detach().cpu().double().numpy()
        off = 0
        for g, it in enumerate(sub):
            k = s0 + g
            n = int(it.positions.shape[0])
            y[k] = float(it.energy.reshape(-1)[0].item()) - float(pred_e[g])
            z_i = it.node_attrs.argmax(dim=-1).cpu().numpy()
            for zi in z_i:
                j = z_to_col.get(int(zi))
                if j is not None:
                    A[k, j] += 1.0
            off += n
        n_batch_atoms = sum(int(it.positions.shape[0]) for it in sub)
        del out, b
        if device.type == "cuda" and (type(model).__name__ == "PolarMACE" or n_batch_atoms >= 300):
            torch.cuda.empty_cache()

    de0, *_ = np.linalg.lstsq(A, y, rcond=None)
    ae = model.atomic_energies_fn.atomic_energies
    with torch.no_grad():
        dt, dev = ae.dtype, ae.device
        de_t = torch.tensor(de0, dtype=torch.float64, device=dev)
        if ae.ndim == 1:
            for j, c in enumerate(cols):
                ae[c] = (ae[c].double() + de_t[j]).to(dt)
        else:
            for j, c in enumerate(cols):
                ae[head_idx, c] = (ae[head_idx, c].double() + de_t[j]).to(dt)


def _try_enable_cueq(model: torch.nn.Module, cls_name: str, strategy: str, use_cueq: bool, device: torch.device, dtype: torch.dtype):
    """Enable CuEq acceleration only if ``e3nn -> cueq -> e3nn`` roundtrip preserves structure."""
    import torch
    if not use_cueq or strategy == "lora" or device.type != "cuda":
        return model, False
    if cls_name not in ("ScaleShiftMACE", "PolarMACE"):
        return model, False
    try:
        from mace.cli import convert_cueq_e3nn, convert_e3nn_cueq
        from mace.tools import torch_tools

        orig_ae_shape = tuple(model.atomic_energies_fn.atomic_energies.shape)
        orig_heads = list(getattr(model, "heads", []) or [])
        dt_str = "float64" if dtype == torch.float64 else "float32"
        m_src = copy.deepcopy(model).cpu().to(dtype=dtype)
        with torch_tools.default_dtype(dt_str):
            m_cueq = convert_e3nn_cueq.run(m_src, return_model=True).to(dtype=dtype)
            # Smoke-check reverse conversion once up front so saving back to e3nn never surprises us after training
            m_back = convert_cueq_e3nn.run(copy.deepcopy(m_cueq).cpu(), return_model=True)
        if tuple(m_back.atomic_energies_fn.atomic_energies.shape) != orig_ae_shape:
            return model, False
        if list(getattr(m_back, "heads", []) or []) != orig_heads:
            return model, False
        del m_back
        return m_cueq, True
    except Exception:
        return model, False


def _convert_back_to_e3nn(model_cueq: torch.nn.Module) -> torch.nn.Module:
    import torch
    from mace.cli import convert_cueq_e3nn

    ae_saved = model_cueq.atomic_energies_fn.atomic_energies.detach().cpu().clone()
    out = convert_cueq_e3nn.run(model_cueq.cpu(), return_model=True)
    with torch.no_grad():
        if out.atomic_energies_fn.atomic_energies.shape == ae_saved.shape:
            out.atomic_energies_fn.atomic_energies.copy_(ae_saved.to(out.atomic_energies_fn.atomic_energies.dtype))
    return out


def _export_bundle_subprocess(
    model_path: Path,
    elements: Sequence[int],
    bundle_out: Path,
    *,
    head: str | None,
    charge: int | None,
    spin_multiplicity: int | None,
) -> None:
    cmd = [
        sys.executable, "-m", "loupmd.export",
        "--model", str(model_path),
        "--elements", ",".join(str(int(z)) for z in elements),
        "--output", str(bundle_out),
    ]
    if head is not None:
        cmd.extend(["--head", str(head)])
    if charge is not None:
        cmd.extend(["--charge", str(int(charge))])
    if spin_multiplicity is not None:
        cmd.extend(["--spin-multiplicity", str(int(spin_multiplicity))])
    env = dict(os.environ)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if res.returncode != 0 or not bundle_out.is_file():
        raise RuntimeError(
            f"Exporting {bundle_out.name} failed (rc={res.returncode}):\n{res.stdout}\n{res.stderr}"
        )


def _verify_loup_gate(
    model_path: Path,
    bundle_path: Path,
    test_frames: Sequence[Atoms],
    *,
    cls_name: str,
    head: str | None,
    charge: int | None,
    spin_multiplicity: int | None,
    r_max: float,
    ref_device: str,
    gpu_device: int,
) -> tuple[bool, dict]:
    """Compare exported ``.loup`` against fine-tuned MACE float64/float32 under the 3x rule."""
    import torch
    from loupmd.validate import references as _refs

    n_probe = 1 if (cls_name == "PolarMACE" and test_frames and len(test_frames[0]) > 200) else 3
    probe = list(test_frames[: min(n_probe, len(test_frames))])
    configs = []
    for i, at in enumerate(probe):
        is_per = all(bool(p) for p in at.pbc) and abs(float(np.linalg.det(at.cell.array))) > 1e-6
        boxed, _ = _pad_nonperiodic_for_engine(at, r_max + 2.0)
        configs.append((f"ft_{i}", boxed, is_per))

    ref_cfg_pairs = [(n, a) for n, a, _ in configs]
    ref64 = _refs._evaluate(model_path, "float64", ref_cfg_pairs, head, ref_device, charge, spin_multiplicity)
    ref32 = _refs._evaluate(model_path, "float32", ref_cfg_pairs, head, ref_device, charge, spin_multiplicity)
    if cls_name == "PolarMACE" and any(is_p for _, _, is_p in configs):
        s_fd = _refs._fd_stress(model_path, ref_cfg_pairs, ref_device)
        s_ref = s_fd
    else:
        s_ref = [s for _, _, s in ref64]

    if gpu_device >= 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

    eng = _core.LoupEngine(bundle_path, device_id=gpu_device)
    de_l, df_l, ds_l = [], [], []
    de32_l, df32_l, ds32_l = [], [], []
    try:
        for idx, (_, boxed, is_per) in enumerate(configs):
            eng.set_system(boxed.numbers, boxed.get_masses(), skin=0.5)
            en_l, f_l, st_l = eng.evaluate(boxed.positions, boxed.cell.array, compute_stress=is_per)
            en64, f64, _ = ref64[idx]
            en32, f32, s32 = ref32[idx]
            N = len(boxed)
            de_l.append(abs(en_l - en64) / N * 1000.0)
            de32_l.append(abs(en32 - en64) / N * 1000.0)
            df_l.append(float(np.max(np.abs(f_l - f64))))
            df32_l.append(float(np.max(np.abs(f32 - f64))))
            if is_per and st_l is not None:
                ds_l.append(float(np.max(np.abs(st_l - s_ref[idx]))))
                ds32_l.append(float(np.max(np.abs(s32 - s_ref[idx]))))

        dE = max(de_l) if de_l else 0.0
        dE32 = max(de32_l) if de32_l else 0.0
        dF = max(df_l) if df_l else 0.0
        dF32 = max(df32_l) if df32_l else 0.0
        dS = max(ds_l) if ds_l else 0.0
        dS32 = max(ds32_l) if ds32_l else 0.0

        tol_E = max(3.0 * dE32, 0.01)
        tol_F = max(3.0 * dF32, 1e-5)
        tol_S = max(3.0 * dS32, 1e-7)
        passed = (dE <= tol_E) and (dF <= tol_F) and (not ds_l or dS <= tol_S)
        return bool(passed), {
            "dE_mev_at": round(dE, 6),
            "tol_E_mev_at": round(tol_E, 6),
            "dF_max_ev_a": round(dF, 7),
            "tol_F_max_ev_a": round(tol_F, 7),
            "dS_max_ev_a3": round(dS, 8) if ds_l else None,
            "tol_S_max_ev_a3": round(tol_S, 8) if ds_l else None,
            "ref_device": ref_device,
        }
    finally:
        del eng
        gc.collect()


def finetune(
    *,
    model: str | Path,
    train_data: str | Path | Atoms | Sequence,
    valid_data: str | Path | Atoms | Sequence | None = None,
    test_data: str | Path | Atoms | Sequence | None = None,
    replay_data: str | Path | Atoms | Sequence | None = None,
    output: str | Path = "./finetune_out",
    head: str | None = None,
    checkpoint: str | Path | None = None,
    sha256: str | None = None,
    trusted: bool = False,
    download: bool = False,
    backend: str = "cuda",
    strategy: str = "auto",
    epochs: int = 30,
    lr: float = 5e-4,
    trunk_lr_mult: float = 0.25,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    anchor_weight: float = 1e-2,
    replay_weight: float = 0.25,
    energy_weight: float = 1.0,
    force_weight: float = 100.0,
    stress_weight: float | None = None,
    batch_size: int | None = None,
    grad_accum: int | None = None,
    dtype: str = "float32",
    use_cueq: bool = True,
    keep_e0: bool = False,
    charge: int | None = None,
    spin_multiplicity: int | None = None,
    energy_key: str | None = None,
    forces_key: str | None = None,
    stress_key: str | None = None,
    export_loup: bool = True,
    verify: bool = True,
    device: int = 0,
    seed: int = 42,
    verbose: bool = True,
) -> FinetuneResult:
    """Fine-tune any LoupMD foundation MACE checkpoint and export a verified ``.loup`` bundle."""
    if backend not in ("cuda", "pytorch"):
        raise InputError(f"unknown fine-tuning backend {backend!r} (expected 'cuda' or 'pytorch')")
    if backend == "cuda":
        if device < 0:
            raise InputError("backend='cuda' requires a CUDA GPU (device >= 0); pass backend='pytorch' for CPU")
        from .cuda_backend import finetune_cuda

        return finetune_cuda(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            replay_data=replay_data,
            output=output,
            head=head,
            checkpoint=checkpoint,
            sha256=sha256,
            trusted=trusted,
            download=download,
            strategy=strategy,
            epochs=epochs,
            lr=lr,
            trunk_lr_mult=trunk_lr_mult,
            anchor_weight=anchor_weight,
            replay_weight=replay_weight,
            energy_weight=energy_weight,
            force_weight=force_weight,
            batch_size=batch_size,
            grad_accum=grad_accum,
            keep_e0=keep_e0,
            charge=charge,
            spin_multiplicity=spin_multiplicity,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            verify=verify,
            device=device,
            seed=seed,
            verbose=verbose,
        )

    import torch
    from mace.tools import AtomicNumberTable
    from mace.tools.torch_geometric import Batch
    from .adapters import (
        AnchorRegularizer,
        configure_trainable_params,
        merge_lora_adapters,
        resolve_strategy,
    )

    t0 = time.perf_counter()
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    np.random.seed(seed)

    def log(msg: str):
        if verbose:
            print(msg, flush=True)

    out_p = Path(output)
    if out_p.suffix in (".model", ".loup"):
        out_dir = out_p.parent
    else:
        out_dir = out_p
    out_dir.mkdir(parents=True, exist_ok=True)
    model_out = out_p if out_p.suffix == ".model" else (out_dir / "finetuned.model")
    bundle_out = (out_p if out_p.suffix == ".loup" else (out_dir / "finetuned.loup")) if export_loup else None

    train_dt = torch.float64 if dtype == "float64" else torch.float32
    if device >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(device)
        dev = torch.device(f"cuda:{device}")
    else:
        dev = torch.device("cpu")

    log(f"[1/6] Loading base checkpoint {model!r}...")
    resolved, digest, _ = models.trusted_checkpoint(
        str(model), head=head, checkpoint=checkpoint, sha256=sha256, trusted=trusted, download=download
    )
    base_model = torch.load(resolved.path, map_location="cpu", weights_only=False)
    cls_name = type(base_model).__name__
    if cls_name not in ("ScaleShiftMACE", "MACE", "PolarMACE"):
        raise InputError(f"unsupported checkpoint class {cls_name!r}")

    heads, active_head, head_idx, export_head = _resolve_head(base_model, head, resolved)
    all_z = [int(z) for z in base_model.atomic_numbers.tolist()]
    r_max = float(base_model.r_max)
    has_joint_emb = hasattr(base_model, "joint_embedding")

    # Ingest & validate dataset frames
    raw_tr = load_frames_input(train_data)
    raw_va = load_frames_input(valid_data)
    raw_te = load_frames_input(test_data)
    raw_rp = load_frames_input(replay_data)
    if not raw_tr:
        raise InputError("--train data contains no frames")

    def _norm_list(raw: list[Atoms], w: float) -> list[Atoms]:
        out = []
        for a in raw:
            norm, ok = normalize_frame(a, energy_key=energy_key, forces_key=forces_key, stress_key=stress_key, weight=w)
            if not ok:
                raise InputError("Training frame missing energy or forces (checked REF_energy/energy and REF_forces/forces)")
            out.append(norm)
        return out

    tr_n = _norm_list(raw_tr, 1.0)
    va_n = _norm_list(raw_va, 1.0)
    te_n = _norm_list(raw_te, 1.0)
    rp_n = _norm_list(raw_rp, float(replay_weight))

    tr_frames, va_frames, te_frames = _split_frames(tr_n, va_n, te_n, seed=seed)
    tr_all = list(tr_frames) + list(rp_n)

    data_elems = sorted({int(z) for a in (tr_all + va_frames + te_frames) for z in a.numbers})
    missing_z = sorted(set(data_elems) - set(all_z))
    if missing_z:
        raise ModelElementError(
            f"dataset contains atomic numbers {missing_z} missing from {resolved.path.name}",
            why=f"supported elements: {all_z}",
        )

    eff_strategy = resolve_strategy(strategy, base_model, len(tr_frames))
    n_total_params = sum(p.numel() for p in base_model.parameters())
    max_atoms = max(len(a) for a in tr_all)
    is_large = (n_total_params > 40_000_000) or (cls_name == "PolarMACE" and len(getattr(base_model, "interactions", ())) >= 3)
    if batch_size is not None:
        bs = int(batch_size)
    elif is_large or cls_name == "PolarMACE" or max_atoms >= 150:
        bs = 1
    else:
        bs = max(1, min(4, len(tr_all), 256 // max(1, max_atoms)))
    accum = int(grad_accum) if grad_accum is not None else (4 if (is_large or bs == 1) and len(tr_all) >= 4 else 1)

    log(f"[2/6] class={cls_name} head={active_head!r} strategy={eff_strategy} params={n_total_params:,} bs={bs}x{accum} elements={data_elems}")

    z_table = AtomicNumberTable(all_z)
    tr_items = _build_atomic_data(
        tr_all, z_table=z_table, cutoff=r_max, heads=heads, head_name=active_head,
        dtype=train_dt, charge=charge, spin_multiplicity=spin_multiplicity, has_joint_emb=has_joint_emb,
    )
    tr_target_only = tr_items[: len(tr_frames)]
    va_items = _build_atomic_data(
        va_frames, z_table=z_table, cutoff=r_max, heads=heads, head_name=active_head,
        dtype=train_dt, charge=charge, spin_multiplicity=spin_multiplicity, has_joint_emb=has_joint_emb,
    )
    te_items = _build_atomic_data(
        te_frames, z_table=z_table, cutoff=r_max, heads=heads, head_name=active_head,
        dtype=train_dt, charge=charge, spin_multiplicity=spin_multiplicity, has_joint_emb=has_joint_emb,
    )

    train_model, used_cueq = _try_enable_cueq(base_model, cls_name, eff_strategy, use_cueq, dev, train_dt)
    del base_model
    gc.collect()
    train_model = _set_model_dtype(train_model, train_dt).to(dev)

    log(f"[3/6] Zero-shot eval (cueq={used_cueq}) & active-head E0 alignment...")
    zero_shot_val = _eval_split(train_model, va_items, dev, batch_size=max(1, min(4, bs)))
    zero_shot_test = _eval_split(train_model, te_items, dev, batch_size=max(1, min(4, bs)))
    if not keep_e0:
        _fit_active_head_e0(train_model, tr_target_only, all_z, data_elems, head_idx, dev, batch_size=max(1, min(4, bs)))
    after_e0_val = _eval_split(train_model, va_items, dev, batch_size=max(1, min(4, bs)))
    log(
        f"  zero-shot test: F={zero_shot_test['f_mev']:.2f} meV/A E={zero_shot_test['e_mev']:.3f} meV/at | "
        f"after E0 fit val E={after_e0_val['e_mev']:.3f} meV/at"
    )

    param_groups, trainable_list, n_trainable, n_total = configure_trainable_params(
        train_model, eff_strategy, lr, trunk_lr_mult=trunk_lr_mult, lora_rank=lora_rank, lora_alpha=lora_alpha
    )
    if not trainable_list:
        raise RuntimeError(f"Strategy {eff_strategy!r} selected 0 trainable parameters")
    anchor = AnchorRegularizer(trainable_list, weight=anchor_weight)
    opt = torch.optim.AdamW(param_groups, lr=float(lr), weight_decay=1e-6)

    has_dataset_stress = all(bool(getattr(it, "_has_stress", False)) for it in tr_items)
    if stress_weight is None:
        ws_eff = 0.0 if (cls_name == "PolarMACE" or not has_dataset_stress) else 10.0
    else:
        ws_eff = float(stress_weight) if has_dataset_stress else 0.0

    # Exponential Moving Average over trainable parameters only
    ema_shadow = {name: p.detach().clone() for name, p in trainable_list}
    ema_decay = 0.985
    opt_step = 0

    def _apply_ema():
        bak = {}
        with torch.no_grad():
            for name, p in trainable_list:
                bak[name] = p.detach().clone()
                p.copy_(ema_shadow[name])
        return bak

    def _restore_from_bak(bak):
        with torch.no_grad():
            for name, p in trainable_list:
                p.copy_(bak[name])

    best_f = float("inf")
    best_ep = 0
    best_trainable_cpu = {name: p.detach().cpu().clone() for name, p in trainable_list}
    eval_every = max(1, min(5, epochs // 5))
    warmup_ep = max(1, min(5, epochs // 8))
    tail_ep = max(1, int(math.floor(epochs * 0.80)))
    rng = np.random.default_rng(seed)
    history: list[dict] = []

    log(f"[4/6] Fine-tuning {n_trainable:,}/{n_total:,} params for {epochs} epochs (ws={ws_eff})...")
    for ep in range(1, epochs + 1):
        train_model.train()
        # Cosine schedule with linear warmup
        if ep <= warmup_ep:
            lr_mult = 0.2 + 0.8 * (ep / warmup_ep)
        else:
            frac = min(1.0, (ep - warmup_ep) / max(1, epochs - warmup_ep))
            lr_mult = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * frac))
        for grp in opt.param_groups:
            grp["lr"] = grp["base_lr"] * lr_mult

        we_cur = float(energy_weight) * (20.0 if (epochs >= 5 and ep > tail_ep) else 1.0)
        wf_cur = float(force_weight)
        perm = rng.permutation(len(tr_items))
        opt.zero_grad(set_to_none=True)
        ep_loss = 0.0
        n_batches = 0

        for step_idx, s0 in enumerate(range(0, len(perm), bs)):
            idxs = perm[s0 : s0 + bs]
            sub = [tr_items[int(i)] for i in idxs]
            b = Batch.from_data_list(sub).to(dev).to_dict()
            use_st = ws_eff > 0.0 and all(bool(getattr(it, "_has_stress", False)) for it in sub)
            out = _forward_model(train_model, b, training=True, compute_stress=use_st)

            n_at_t = torch.tensor([float(it.positions.shape[0]) for it in sub], dtype=train_dt, device=dev)
            gw_t = torch.tensor([float(getattr(it, "_sample_weight", 1.0)) for it in sub], dtype=train_dt, device=dev)
            gw_at = gw_t[b["batch"]]

            le = (gw_t * ((out["energy"].to(train_dt) - b["energy"].view(-1).to(train_dt)) / n_at_t) ** 2).mean()
            df = out["forces"].to(train_dt) - b["forces"].to(train_dt)
            lf = (gw_at.unsqueeze(-1) * (df ** 2)).mean()
            loss = we_cur * le + wf_cur * lf
            if use_st and out.get("stress", None) is not None:
                ls = (gw_t.view(-1, 1, 1) * (out["stress"].to(train_dt) - b["stress"].view(-1, 3, 3).to(train_dt)) ** 2).mean()
                loss = loss + ws_eff * ls
            loss = loss + anchor.loss(dev, train_dt)

            n_chunks = max(1, math.ceil(len(perm) / bs))
            scaled_loss = loss / float(accum)
            scaled_loss.backward()
            ep_loss += float(loss.detach().item())
            n_batches += 1

            is_step = ((step_idx + 1) % accum == 0) or (step_idx + 1 == n_chunks)
            if is_step:
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable_list], max_norm=10.0)
                opt.step()
                opt_step += 1
                opt.zero_grad(set_to_none=True)
                d_eff = min(ema_decay, (1.0 + opt_step) / (10.0 + opt_step))
                with torch.no_grad():
                    for name, p in trainable_list:
                        ema_shadow[name].mul_(d_eff).add_(p.detach(), alpha=1.0 - d_eff)

        if (ep % eval_every == 0) or (ep == epochs) or (ep == 1):
            bak = _apply_ema()
            sc = _eval_split(train_model, va_items, dev, batch_size=max(1, min(4, bs)))
            _restore_from_bak(bak)
            history.append({"epoch": ep, "loss": ep_loss / max(1, n_batches), **sc})
            log(f"  [FT ep {ep:3d}/{epochs}] loss={ep_loss / max(1, n_batches):.4f} val_F={sc['f_mev']:.2f} meV/A val_E={sc['e_mev']:.3f} meV/at")
            if sc["f_mev"] < best_f:
                best_f = sc["f_mev"]
                best_ep = ep
                best_trainable_cpu = {name: t.detach().cpu().clone() for name, t in ema_shadow.items()}

    # Restore best EMA weights + merge LoRA + convert back to clean float64 e3nn .model
    with torch.no_grad():
        for name, p in trainable_list:
            p.copy_(best_trainable_cpu[name].to(device=p.device, dtype=p.dtype))
    if eff_strategy == "lora":
        merge_lora_adapters(train_model)
    del opt, ema_shadow, best_trainable_cpu, anchor
    gc.collect()
    if device >= 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if used_cueq:
        train_model = _convert_back_to_e3nn(train_model)
    eval_on_cpu = is_large or (cls_name == "PolarMACE" and max_atoms >= 300) or (max_atoms >= 500)
    if eval_on_cpu:
        train_model = train_model.cpu()
        if device >= 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    eval_dev = torch.device("cpu") if eval_on_cpu else dev
    train_model64 = train_model.double().to(eval_dev)
    if not keep_e0:
        tr_eval_items = _build_atomic_data(
            tr_frames, z_table=z_table, cutoff=r_max, heads=heads, head_name=active_head,
            dtype=torch.float64, charge=charge, spin_multiplicity=spin_multiplicity, has_joint_emb=has_joint_emb,
        )
        _fit_active_head_e0(train_model64, tr_eval_items, all_z, data_elems, head_idx, eval_dev, batch_size=max(1, min(4, bs)))

    te_items64 = _build_atomic_data(
        te_frames, z_table=z_table, cutoff=r_max, heads=heads, head_name=active_head,
        dtype=torch.float64, charge=charge, spin_multiplicity=spin_multiplicity, has_joint_emb=has_joint_emb,
    )
    test_metrics = _eval_split(train_model64, te_items64, eval_dev, batch_size=max(1, min(4, bs)))

    save_model = train_model64.cpu().double()
    for p in save_model.parameters():
        p.requires_grad_(False)
    torch.save(save_model, model_out)
    log(f"[5/6] Saved fine-tuned model -> {model_out} | test F={test_metrics['f_mev']:.2f} meV/A E={test_metrics['e_mev']:.3f} meV/at")

    del train_model, train_model64, save_model
    gc.collect()
    if device >= 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

    gate_passed = None
    gate_metrics = None
    if export_loup and bundle_out is not None:
        log(f"[6/6] Exporting torch-free bundle -> {bundle_out}...")
        _export_bundle_subprocess(
            model_out, data_elems, bundle_out,
            head=export_head, charge=charge, spin_multiplicity=spin_multiplicity,
        )
        if verify:
            ref_dev = "cpu" if (eval_on_cpu or dev.type == "cpu" or not torch.cuda.is_available()) else "cuda"
            gate_passed, gate_metrics = _verify_loup_gate(
                model_out, bundle_out, te_frames,
                cls_name=cls_name, head=export_head, charge=charge, spin_multiplicity=spin_multiplicity,
                r_max=r_max, ref_device=ref_dev, gpu_device=device,
            )
            log(f"  3x Gate: {'PASS' if gate_passed else 'FAIL'} ({gate_metrics})")

    wall_s = time.perf_counter() - t0
    report = {
        "base_model": str(model),
        "resolved_checkpoint": str(resolved.path),
        "model_class": cls_name,
        "head": active_head,
        "strategy": eff_strategy,
        "used_cueq": used_cueq,
        "dtype": dtype,
        "elements": data_elems,
        "n_trainable": n_trainable,
        "n_total_params": n_total,
        "best_ep": best_ep,
        "dataset_counts": {"train": len(tr_frames), "replay": len(rp_n), "val": len(va_frames), "test": len(te_frames)},
        "zero_shot_val": zero_shot_val,
        "zero_shot_test": zero_shot_test,
        "test_metrics": test_metrics,
        "gate_passed": gate_passed,
        "gate_metrics": gate_metrics,
        "history": history,
        "wall_time_s": round(wall_s, 2),
        "model_path": str(model_out),
        "bundle_path": str(bundle_out) if bundle_out is not None else None,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return FinetuneResult(
        model_path=model_out,
        bundle_path=bundle_out,
        report_path=report_path,
        base_model=str(model),
        model_class=cls_name,
        head=active_head,
        strategy=eff_strategy,
        elements=tuple(data_elems),
        zero_shot_metrics=zero_shot_test,
        test_metrics=test_metrics,
        gate_passed=gate_passed,
        wall_time_s=round(wall_s, 2),
        report=report,
    )
