"""Pure-CUDA foundation MACE fine-tuning backend (``libloupmd_ft.so``) — zero PyTorch in training or inference."""
from __future__ import annotations

import ctypes
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from loupmd import _core, models
from loupmd.bundles import ensure_loup_bundle
from loupmd.errors import InputError, ModelElementError
from .._lib import find_native_lib
from ..data import _pad_nonperiodic_for_engine, load_frames_input, normalize_frame


def _find_ft_library() -> Path:
    return find_native_lib(
        "libloupmd_ft.so",
        env_var="LOUP_FT_LIB",
        make_target="mace-ft-lib",
        auto_build=False,
    )


_LIB_CACHE = None


def _load_lib():
    global _LIB_CACHE
    if _LIB_CACHE is not None:
        return _LIB_CACHE
    lib = ctypes.CDLL(str(_find_ft_library()))
    c_i = ctypes.c_int
    c_i64 = ctypes.c_int64
    c_f = ctypes.c_float
    c_d = ctypes.c_double
    c_p = ctypes.c_void_p
    c_cp = ctypes.c_char_p
    c_ip = ctypes.POINTER(c_i)
    c_i64p = ctypes.POINTER(c_i64)
    c_fp = ctypes.POINTER(c_f)
    c_dp = ctypes.POINTER(c_d)

    lib.mace_ft_create.argtypes = [c_cp, c_cp, c_i, c_cp, c_i]
    lib.mace_ft_create.restype = c_p
    lib.mace_ft_destroy.argtypes = [c_p]
    lib.mace_ft_destroy.restype = None
    lib.mace_ft_info.argtypes = [c_p, c_ip, c_ip, c_ip, c_ip, c_i64p, c_i64p, c_dp]
    lib.mace_ft_info.restype = c_i
    lib.mace_ft_set_e0.argtypes = [c_p, c_dp]
    lib.mace_ft_set_e0.restype = c_i
    lib.mace_ft_get_e0.argtypes = [c_p, c_dp]
    lib.mace_ft_get_e0.restype = c_i
    lib.mace_ft_eval_batch.argtypes = [
        c_p, c_i, c_i, c_i, c_ip, c_ip, c_ip, c_fp, c_ip, c_ip, c_fp, c_dp, c_dp, c_i, c_dp, c_fp
    ]
    lib.mace_ft_eval_batch.restype = c_i
    lib.mace_ft_zero_grad.argtypes = [c_p]
    lib.mace_ft_zero_grad.restype = c_i
    lib.mace_ft_backward_batch.argtypes = [
        c_p, c_i, c_i, c_i, c_ip, c_ip, c_ip, c_fp, c_ip, c_ip, c_fp, c_dp, c_dp, c_dp, c_fp, c_f, c_f, c_f, c_dp
    ]
    lib.mace_ft_backward_batch.restype = c_i
    lib.mace_ft_step.argtypes = [c_p, c_f, c_f, c_f, c_f, c_f]
    lib.mace_ft_step.restype = c_i
    lib.mace_ft_save.argtypes = [c_p, c_cp, c_i]
    lib.mace_ft_save.restype = c_i
    lib.mace_ft_param_meta.argtypes = [c_p, c_i, c_cp, c_i, c_i64p, c_ip]
    lib.mace_ft_param_meta.restype = c_i
    lib.mace_ft_param_get_elem.argtypes = [c_p, c_i, c_i64, c_fp, c_fp]
    lib.mace_ft_param_get_elem.restype = c_i
    lib.mace_ft_param_set_elem.argtypes = [c_p, c_i, c_i64, c_f]
    lib.mace_ft_param_set_elem.restype = c_i
    lib.mace_ft_last_error.argtypes = [c_p]
    lib.mace_ft_last_error.restype = c_cp

    _LIB_CACHE = lib
    return lib


def build_frame_graph(atoms: Atoms, z_to_sp: dict[int, int], r_max: float) -> dict:
    """Pre-compute directed edge list (vec = pos[rcv] - pos[snd] + shift) matching libloupmd.so."""
    a, _ = _pad_nonperiodic_for_engine(atoms, r_max + 1.0)
    sp = np.array([z_to_sp[int(z)] for z in a.numbers], dtype=np.int32)
    snd, rcv, D = neighbor_list("ijD", a, float(r_max))
    snd = np.asarray(snd, dtype=np.int32)
    rcv = np.asarray(rcv, dtype=np.int32)
    D = np.asarray(D, dtype=np.float64)
    if len(snd) > 0:
        dist = np.linalg.norm(D, axis=1)
        mask = dist < float(r_max) - 1e-6
        snd, rcv, D, dist = snd[mask], rcv[mask], D[mask], dist[mask]
        vl = np.empty((len(snd), 4), dtype=np.float32)
        vl[:, :3] = D.astype(np.float32)
        vl[:, 3] = dist.astype(np.float32)
    else:
        vl = np.zeros((0, 4), dtype=np.float32)
    e_ref = float(atoms.info.get("REF_energy", 0.0))
    f_ref = np.asarray(atoms.arrays.get("REF_forces", np.zeros((len(a), 3))), dtype=np.float32).reshape(-1, 3)
    w = float(atoms.info.get("distill_weight", 1.0))
    return {
        "N": len(a),
        "species": sp,
        "senders": snd,
        "receivers": rcv,
        "vec_len": vl,
        "pos64": np.ascontiguousarray(a.positions, dtype=np.float64),
        "cell": np.ascontiguousarray(a.cell.array, dtype=np.float64),
        "E_ref": e_ref,
        "F_ref": np.ascontiguousarray(f_ref, dtype=np.float32),
        "weight": w,
        "atoms": a,
    }


def collate_graphs(graphs: Sequence[dict]) -> dict:
    G = len(graphs)
    N = sum(g["N"] for g in graphs)
    E = sum(len(g["senders"]) for g in graphs)
    species = np.empty(N, dtype=np.int32)
    atom_graph = np.empty(N, dtype=np.int32)
    graph_natoms = np.empty(G, dtype=np.int32)
    graph_w = np.empty(G, dtype=np.float32)
    senders = np.empty(E, dtype=np.int32)
    receivers = np.empty(E, dtype=np.int32)
    vec_len = np.empty((E, 4), dtype=np.float32)
    pos64 = np.empty((N, 3), dtype=np.float64)
    cells = np.empty((G, 3, 3), dtype=np.float64)
    E_ref = np.empty(G, dtype=np.float64)
    F_ref = np.empty((N, 3), dtype=np.float32)

    a_off = 0
    e_off = 0
    for gi, g in enumerate(graphs):
        n = g["N"]
        e = len(g["senders"])
        species[a_off : a_off + n] = g["species"]
        atom_graph[a_off : a_off + n] = gi
        graph_natoms[gi] = n
        graph_w[gi] = g["weight"]
        pos64[a_off : a_off + n] = g["pos64"]
        cells[gi] = g["cell"]
        E_ref[gi] = g["E_ref"]
        F_ref[a_off : a_off + n] = g["F_ref"]
        if e > 0:
            senders[e_off : e_off + e] = g["senders"] + a_off
            receivers[e_off : e_off + e] = g["receivers"] + a_off
            vec_len[e_off : e_off + e] = g["vec_len"]
        a_off += n
        e_off += e

    return {
        "G": G,
        "N": N,
        "E": E,
        "species": np.ascontiguousarray(species),
        "atom_graph": np.ascontiguousarray(atom_graph),
        "graph_natoms": np.ascontiguousarray(graph_natoms),
        "graph_w": np.ascontiguousarray(graph_w),
        "senders": np.ascontiguousarray(senders),
        "receivers": np.ascontiguousarray(receivers),
        "vec_len": np.ascontiguousarray(vec_len.reshape(-1)),
        "pos64": np.ascontiguousarray(pos64.reshape(-1)),
        "cells": np.ascontiguousarray(cells.reshape(-1)),
        "E_ref": np.ascontiguousarray(E_ref),
        "F_ref": np.ascontiguousarray(F_ref.reshape(-1)),
    }


def _ptr_i(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def _ptr_f(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _ptr_d(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class CudaMaceTrainer:
    """Thin ctypes wrapper around ``libloupmd_ft.so`` (zero PyTorch)."""

    def __init__(self, bundle_path: str | Path, strategy: str = "full", device: int = 0):
        self._lib = _load_lib()
        err = ctypes.create_string_buffer(1024)
        self._h = self._lib.mace_ft_create(str(bundle_path).encode(), strategy.encode(), int(device), err, 1024)
        if not self._h:
            raise RuntimeError(f"mace_ft_create failed: {err.value.decode()}")
        S, nL, has_lr, nP = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        tr_el, tot_el = ctypes.c_int64(), ctypes.c_int64()
        rmax = ctypes.c_double()
        self._lib.mace_ft_info(
            self._h, ctypes.byref(S), ctypes.byref(nL), ctypes.byref(has_lr), ctypes.byref(nP),
            ctypes.byref(tr_el), ctypes.byref(tot_el), ctypes.byref(rmax),
        )
        self.S = S.value
        self.n_layers = nL.value
        self.has_lr = has_lr.value
        self.n_params = nP.value
        self.n_trainable = tr_el.value
        self.n_total = tot_el.value
        self.r_max = rmax.value

    def close(self):
        if getattr(self, "_h", None):
            self._lib.mace_ft_destroy(self._h)
            self._h = None

    def __del__(self):
        self.close()

    def _err(self) -> str:
        return self._lib.mace_ft_last_error(self._h).decode()

    def get_e0(self) -> np.ndarray:
        out = np.zeros(self.S, dtype=np.float64)
        if self._lib.mace_ft_get_e0(self._h, _ptr_d(out)) != 0:
            raise RuntimeError(self._err())
        return out

    def set_e0(self, e0: np.ndarray) -> None:
        arr = np.ascontiguousarray(e0, dtype=np.float64)
        if self._lib.mace_ft_set_e0(self._h, _ptr_d(arr)) != 0:
            raise RuntimeError(self._err())

    def eval_batch(self, batch: dict, use_ema: bool = False) -> tuple[np.ndarray, np.ndarray]:
        E_out = np.zeros(batch["G"], dtype=np.float64)
        F_out = np.zeros(batch["N"] * 3, dtype=np.float32)
        rc = self._lib.mace_ft_eval_batch(
            self._h, batch["G"], batch["N"], batch["E"],
            _ptr_i(batch["species"]), _ptr_i(batch["atom_graph"]), _ptr_i(batch["graph_natoms"]),
            _ptr_f(batch["graph_w"]), _ptr_i(batch["senders"]), _ptr_i(batch["receivers"]),
            _ptr_f(batch["vec_len"]), _ptr_d(batch["pos64"]), _ptr_d(batch["cells"]),
            int(use_ema), _ptr_d(E_out), _ptr_f(F_out),
        )
        if rc != 0:
            raise RuntimeError(f"mace_ft_eval_batch failed: {self._err()}")
        return E_out, F_out.reshape(batch["N"], 3)

    def zero_grad(self) -> None:
        if self._lib.mace_ft_zero_grad(self._h) != 0:
            raise RuntimeError(self._err())

    def backward_batch(self, batch: dict, w_e: float, w_f: float, inv_accum: float = 1.0) -> float:
        loss = ctypes.c_double()
        rc = self._lib.mace_ft_backward_batch(
            self._h, batch["G"], batch["N"], batch["E"],
            _ptr_i(batch["species"]), _ptr_i(batch["atom_graph"]), _ptr_i(batch["graph_natoms"]),
            _ptr_f(batch["graph_w"]), _ptr_i(batch["senders"]), _ptr_i(batch["receivers"]),
            _ptr_f(batch["vec_len"]), _ptr_d(batch["pos64"]), _ptr_d(batch["cells"]),
            _ptr_d(batch["E_ref"]), _ptr_f(batch["F_ref"]),
            float(w_e), float(w_f), float(inv_accum), ctypes.byref(loss),
        )
        if rc != 0:
            raise RuntimeError(f"mace_ft_backward_batch failed: {self._err()}")
        return float(loss.value)

    def step(self, lr_head: float, lr_trunk: float, max_grad_norm: float = 10.0,
             anchor_weight: float = 1e-2, ema_decay_cap: float = 0.985) -> None:
        rc = self._lib.mace_ft_step(
            self._h, float(lr_head), float(lr_trunk), float(max_grad_norm), float(anchor_weight), float(ema_decay_cap)
        )
        if rc != 0:
            raise RuntimeError(f"mace_ft_step failed: {self._err()}")

    def save(self, out_path: str | Path, use_ema: bool = True) -> None:
        rc = self._lib.mace_ft_save(self._h, str(out_path).encode(), int(use_ema))
        if rc != 0:
            raise RuntimeError(f"mace_ft_save failed: {self._err()}")

    def list_params(self) -> list[tuple[int, str, int, bool]]:
        out = []
        buf = ctypes.create_string_buffer(256)
        num = ctypes.c_int64()
        tr = ctypes.c_int()
        for i in range(self.n_params):
            self._lib.mace_ft_param_meta(self._h, i, buf, 256, ctypes.byref(num), ctypes.byref(tr))
            out.append((i, buf.value.decode(), int(num.value), bool(tr.value)))
        return out

    def get_param_elem(self, idx: int, elem: int) -> tuple[float, float]:
        v, g = ctypes.c_float(), ctypes.c_float()
        if self._lib.mace_ft_param_get_elem(self._h, idx, int(elem), ctypes.byref(v), ctypes.byref(g)) != 0:
            raise RuntimeError(self._err())
        return float(v.value), float(g.value)

    def set_param_elem(self, idx: int, elem: int, val: float) -> None:
        if self._lib.mace_ft_param_set_elem(self._h, idx, int(elem), float(val)) != 0:
            raise RuntimeError(self._err())


def _eval_graphs_cuda(tr: CudaMaceTrainer, graphs: Sequence[dict], batch_size: int = 4, use_ema: bool = False) -> dict:
    if not graphs:
        return {"f_mev": 0.0, "e_mev": 0.0, "s_gpa": None}
    f_sq, e_mev = [], []
    for s0 in range(0, len(graphs), batch_size):
        sub = graphs[s0 : s0 + batch_size]
        b = collate_graphs(sub)
        ep, fp = tr.eval_batch(b, use_ema=use_ema)
        off = 0
        for g_i, g in enumerate(sub):
            n = g["N"]
            df = fp[off : off + n] - g["F_ref"]
            f_sq.append((df.astype(np.float64) ** 2).ravel())
            e_mev.append(((float(ep[g_i]) - g["E_ref"]) / max(1, n)) * 1000.0)
            off += n
    return {
        "f_mev": float(np.sqrt(np.mean(np.concatenate(f_sq))) * 1000.0),
        "e_mev": float(np.sqrt(np.mean(np.square(e_mev)))),
        "s_gpa": None,
    }


def _fit_e0_cuda(tr: CudaMaceTrainer, graphs: Sequence[dict], batch_size: int = 4, use_ema: bool = False) -> None:
    if not graphs:
        return
    S = tr.S
    A = np.zeros((len(graphs), S), dtype=np.float64)
    y = np.zeros(len(graphs), dtype=np.float64)
    for s0 in range(0, len(graphs), batch_size):
        sub = graphs[s0 : s0 + batch_size]
        b = collate_graphs(sub)
        ep, _ = tr.eval_batch(b, use_ema=use_ema)
        for g_i, g in enumerate(sub):
            k = s0 + g_i
            y[k] = g["E_ref"] - float(ep[g_i])
            for sp in g["species"]:
                A[k, int(sp)] += 1.0
    de0, *_ = np.linalg.lstsq(A, y, rcond=None)
    tr.set_e0(tr.get_e0() + de0)


def finetune_cuda(
    *,
    model: str | Path,
    train_data,
    valid_data=None,
    test_data=None,
    replay_data=None,
    output: str | Path = "./finetune_out",
    head: str | None = None,
    checkpoint: str | Path | None = None,
    sha256: str | None = None,
    trusted: bool = False,
    download: bool = False,
    strategy: str = "auto",
    epochs: int = 30,
    lr: float = 5e-4,
    trunk_lr_mult: float = 0.25,
    anchor_weight: float = 1e-2,
    replay_weight: float = 0.25,
    energy_weight: float = 1.0,
    force_weight: float = 100.0,
    batch_size: int | None = None,
    grad_accum: int | None = None,
    keep_e0: bool = False,
    charge: int | None = None,
    spin_multiplicity: int | None = None,
    energy_key: str | None = None,
    forces_key: str | None = None,
    stress_key: str | None = None,
    verify: bool = True,
    device: int = 0,
    seed: int = 42,
    verbose: bool = True,
):
    """100% pure-CUDA foundation MACE fine-tuning (zero PyTorch in the training loop)."""
    from .core import FinetuneResult, _split_frames

    t0 = time.perf_counter()

    def log(msg: str):
        if verbose:
            print(msg, flush=True)

    out_p = Path(output)
    out_dir = out_p.parent if out_p.suffix in (".model", ".loup") else out_p
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_out = out_p if out_p.suffix == ".loup" else (out_dir / "finetuned.loup")

    # 1) Ingest & normalize frames
    raw_tr = load_frames_input(train_data)
    raw_va = load_frames_input(valid_data)
    raw_te = load_frames_input(test_data)
    raw_rp = load_frames_input(replay_data)
    if not raw_tr:
        raise InputError("--train data contains no frames")

    def _norm(raw: list[Atoms], w: float) -> list[Atoms]:
        out = []
        for a in raw:
            n, ok = normalize_frame(a, energy_key=energy_key, forces_key=forces_key, stress_key=stress_key, weight=w)
            if not ok:
                raise InputError("Training frame missing energy or forces")
            out.append(n)
        return out

    tr_n = _norm(raw_tr, 1.0)
    va_n = _norm(raw_va, 1.0)
    te_n = _norm(raw_te, 1.0)
    rp_n = _norm(raw_rp, float(replay_weight))
    tr_frames, va_frames, te_frames = _split_frames(tr_n, va_n, te_n, seed=seed)
    tr_all = list(tr_frames) + list(rp_n)
    data_elems = sorted({int(z) for a in (tr_all + va_frames + te_frames) for z in a.numbers})

    # 2) Obtain base .loup bundle (either directly passed .loup or cached/exported via ensure_loup_bundle)
    model_p = Path(str(model))
    active_head = head
    if model_p.is_file() and model_p.suffix == ".loup":
        base_bundle = model_p
    else:
        log(f"[1/5] Resolving base .loup bundle for {model!r} (elements={data_elems})...")
        resolved, _, _ = models.trusted_checkpoint(
            str(model), head=head, checkpoint=checkpoint, sha256=sha256, trusted=trusted, download=download
        )
        if active_head is None and resolved.label is not None:
            active_head = resolved.label.head
        base_bundle = ensure_loup_bundle(
            resolved.path, data_elems, head=active_head,
            charge=None if charge is None else int(charge),
            spin_multiplicity=None if spin_multiplicity is None else int(spin_multiplicity),
        )

    # Resolve strategy ("lora" maps to "top" in pure-CUDA direct weight space)
    eff_strategy = strategy
    if eff_strategy == "lora":
        eff_strategy = "top"
    if eff_strategy == "auto":
        if len(tr_frames) < 12:
            eff_strategy = "readout"
        else:
            # Probe param count / PolarLR
            probe_tr = CudaMaceTrainer(base_bundle, strategy="full", device=device)
            is_big = (probe_tr.n_total > 40_000_000) or (probe_tr.has_lr and probe_tr.n_layers >= 3)
            probe_tr.close()
            eff_strategy = "top" if is_big else "full"

    tr = CudaMaceTrainer(base_bundle, strategy=eff_strategy, device=device)
    try:
        max_atoms = max(len(a) for a in tr_all)
        is_large = (tr.n_total > 40_000_000) or (tr.has_lr and tr.n_layers >= 3)
        if batch_size is not None:
            bs = int(batch_size)
        elif is_large or tr.has_lr or max_atoms >= 150:
            bs = 1
        else:
            bs = max(1, min(4, len(tr_all), 256 // max(1, max_atoms)))
        accum = int(grad_accum) if grad_accum is not None else (4 if (is_large or bs == 1) and len(tr_all) >= 4 else 1)

        log(f"[2/5] Pure-CUDA trainer: backend=cuda has_lr={tr.has_lr} layers={tr.n_layers} strategy={eff_strategy} "
            f"trainable={tr.n_trainable:,}/{tr.n_total:,} bs={bs}x{accum}")

        z_to_sp = {z: i for i, z in enumerate(data_elems)}
        tr_graphs = [build_frame_graph(a, z_to_sp, tr.r_max) for a in tr_all]
        tr_target_graphs = tr_graphs[: len(tr_frames)]
        va_graphs = [build_frame_graph(a, z_to_sp, tr.r_max) for a in va_frames]
        te_graphs = [build_frame_graph(a, z_to_sp, tr.r_max) for a in te_frames]

        log("[3/5] Zero-shot eval & E0 alignment (pure CUDA)...")
        zero_shot_val = _eval_graphs_cuda(tr, va_graphs, batch_size=max(1, min(4, bs)), use_ema=False)
        zero_shot_test = _eval_graphs_cuda(tr, te_graphs, batch_size=max(1, min(4, bs)), use_ema=False)
        if not keep_e0:
            _fit_e0_cuda(tr, tr_target_graphs, batch_size=max(1, min(4, bs)), use_ema=False)
        after_e0_val = _eval_graphs_cuda(tr, va_graphs, batch_size=max(1, min(4, bs)), use_ema=False)
        log(f"  zero-shot test: F={zero_shot_test['f_mev']:.2f} meV/A E={zero_shot_test['e_mev']:.3f} meV/at | "
            f"after E0 val E={after_e0_val['e_mev']:.3f} meV/at")

        # 4) Pure-CUDA training loop
        rng = np.random.default_rng(seed)
        eval_every = max(1, min(5, epochs // 5))
        warmup_ep = max(1, min(5, epochs // 8))
        tail_ep = max(1, int(math.floor(epochs * 0.80)))
        best_f = float("inf")
        best_ep = 0
        best_path = out_dir / ".best_ema.loup"
        history: list[dict] = []
        tr.save(best_path, use_ema=True)

        t_train0 = time.perf_counter()
        log(f"[4/5] Pure-CUDA fine-tuning for {epochs} epochs...")
        for ep in range(1, epochs + 1):
            if ep <= warmup_ep:
                lr_mult = 0.2 + 0.8 * (ep / warmup_ep)
            else:
                frac = min(1.0, (ep - warmup_ep) / max(1, epochs - warmup_ep))
                lr_mult = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * frac))
            lr_head = float(lr) * lr_mult
            lr_trunk = (float(lr) if eff_strategy == "top" else float(lr) * float(trunk_lr_mult)) * lr_mult
            we_cur = float(energy_weight) * (20.0 if (epochs >= 5 and ep > tail_ep) else 1.0)
            wf_cur = float(force_weight)

            perm = rng.permutation(len(tr_graphs))
            n_chunks = max(1, math.ceil(len(perm) / bs))
            tr.zero_grad()
            ep_loss = 0.0
            n_batches = 0
            for step_idx, s0 in enumerate(range(0, len(perm), bs)):
                sub = [tr_graphs[int(i)] for i in perm[s0 : s0 + bs]]
                b = collate_graphs(sub)
                ep_loss += tr.backward_batch(b, we_cur, wf_cur, 1.0 / float(accum))
                n_batches += 1
                is_step = ((step_idx + 1) % accum == 0) or (step_idx + 1 == n_chunks)
                if is_step:
                    tr.step(lr_head, lr_trunk, max_grad_norm=10.0, anchor_weight=float(anchor_weight), ema_decay_cap=0.985)

            if (ep % eval_every == 0) or (ep == epochs) or (ep == 1):
                sc = _eval_graphs_cuda(tr, va_graphs, batch_size=max(1, min(4, bs)), use_ema=True)
                history.append({"epoch": ep, "loss": ep_loss / max(1, n_batches), **sc})
                log(f"  [CUDA ep {ep:3d}/{epochs}] loss={ep_loss / max(1, n_batches):.4f} "
                    f"val_F={sc['f_mev']:.2f} meV/A val_E={sc['e_mev']:.3f} meV/at")
                if sc["f_mev"] < best_f:
                    best_f = sc["f_mev"]
                    best_ep = ep
                    tr.save(best_path, use_ema=True)
        train_s = time.perf_counter() - t_train0
    finally:
        tr.close()

    # Load best EMA bundle for final E0 polish + test eval + libloupmd.so parity check
    tr_best = CudaMaceTrainer(best_path, strategy="readout", device=device)
    try:
        if not keep_e0:
            _fit_e0_cuda(tr_best, tr_target_graphs, batch_size=max(1, min(4, bs)), use_ema=False)
        test_metrics = _eval_graphs_cuda(tr_best, te_graphs, batch_size=max(1, min(4, bs)), use_ema=False)
        tr_best.save(bundle_out, use_ema=False)
        n_total = tr_best.n_total
        has_lr = tr_best.has_lr
        r_max_val = tr_best.r_max
    finally:
        tr_best.close()
    if best_path.exists():
        best_path.unlink()

    # Verify saved .loup loads and matches libloupmd.so inference engine!
    gate_passed = None
    gate_metrics = None
    if verify:
        tr_v = CudaMaceTrainer(bundle_out, strategy="readout", device=device)
        eng = _core.LoupEngine(bundle_out, device_id=device)
        try:
            probe = te_graphs[: min(3, len(te_graphs))]
            de_l, df_l = [], []
            for g in probe:
                b1 = collate_graphs([g])
                ep, fp = tr_v.eval_batch(b1, use_ema=False)
                at = g["atoms"]
                eng.set_system(at.numbers, at.get_masses(), skin=0.5)
                en_l, f_l, _ = eng.evaluate(at.positions, at.cell.array, compute_stress=False)
                de_l.append(abs(float(ep[0]) - float(en_l)) / max(1, len(at)) * 1000.0)
                df_l.append(float(np.max(np.abs(fp - f_l))))
            dE = max(de_l) if de_l else 0.0
            dF = max(df_l) if df_l else 0.0
            gate_passed = (dE < 0.05) and (dF < 1e-3)
            gate_metrics = {"dE_mev_at": round(dE, 6), "dF_max_ev_a": round(dF, 7), "engine": "libloupmd.so"}
            log(f"[5/5] Saved {bundle_out} | test F={test_metrics['f_mev']:.2f} meV/A E={test_metrics['e_mev']:.3f} meV/at | "
                f"Engine parity: {'PASS' if gate_passed else 'FAIL'} ({gate_metrics})")
        finally:
            del eng
            tr_v.close()

    wall_s = time.perf_counter() - t0
    report = {
        "backend": "cuda",
        "base_model": str(model),
        "base_bundle": str(base_bundle),
        "model_class": "PolarMACE" if has_lr else "ScaleShiftMACE",
        "head": active_head,
        "strategy": eff_strategy,
        "elements": data_elems,
        "n_total_params": n_total,
        "best_ep": best_ep,
        "train_time_s": round(train_s, 3),
        "wall_time_s": round(wall_s, 3),
        "zero_shot_val": zero_shot_val,
        "zero_shot_test": zero_shot_test,
        "test_metrics": test_metrics,
        "gate_passed": gate_passed,
        "gate_metrics": gate_metrics,
        "history": history,
        "bundle_path": str(bundle_out),
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return FinetuneResult(
        model_path=bundle_out,
        bundle_path=bundle_out,
        report_path=report_path,
        base_model=str(model),
        model_class="PolarMACE" if has_lr else "ScaleShiftMACE",
        head=active_head,
        strategy=eff_strategy,
        elements=tuple(data_elems),
        zero_shot_metrics=zero_shot_test,
        test_metrics=test_metrics,
        gate_passed=gate_passed,
        wall_time_s=round(wall_s, 3),
        report=report,
    )
