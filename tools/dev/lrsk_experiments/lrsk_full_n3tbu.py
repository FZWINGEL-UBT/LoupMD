#!/usr/bin/env python3
"""Full Pure-CUDA BO-LRSK n3tbu_v1 Trainer (`src/train/lrsk_full_n3tbu.py`).

100% GPU C++ training loop (`libloup_lrsk_full_train_v2.so`) with ZERO Python/PyTorch autograd
in the training loop, exact parity with `train_n3tbu_lrsk_v4_fast.py` (importance sampling from
GPU `frame_err` EMA, per-graph normalized force MSE, dynamic `w_g` weighting, GPU Block-Jacobi
SOAP + cautious-Adam + EMA, and DFT-only post-EMA E0 calibration).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / ".tmp_o"), str(_THIS_DIR), "/mnt/homes/zwingel/loups-phase1"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from ase.io import iread
from mace.data import AtomicData, KeySpecification, config_from_atoms
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric import Batch

from p2_lrsk_v4_fast import FastLRSKStudent, fit_min_norm_e0
from lrsk_full_trainer import CudaFullLRSKTrainer


def build_data(frames, z_table, r_max, e0_map):
    out = []
    ks = KeySpecification(info_keys={"energy": "energy_tgt"}, arrays_keys={"forces": "REF_forces"})
    for at in frames:
        b = at.copy()
        b.info["energy_tgt"] = float(at.info["REF_energy"]) - sum(float(e0_map[int(z)]) for z in b.numbers)
        ad = AtomicData.from_config(config_from_atoms(b, key_specification=ks), z_table=z_table, cutoff=r_max)
        out.append(ad)
    return out


def eval_split(model, frames, data_list, zs, device):
    model.eval()
    f_sq = 0.0
    f_cnt = 0
    e_errs = []
    sp_sq = {z: 0.0 for z in zs}
    sp_cnt = {z: 0 for z in zs}
    ct_sq = {}
    ct_cnt = {}
    bs = 16
    for s0 in range(0, len(frames), bs):
        sub_f = frames[s0 : s0 + bs]
        sub_d = data_list[s0 : s0 + bs]
        b = Batch.from_data_list(sub_d).to(device).to_dict()
        out = model(b, training=False, compute_force=True, compute_stress=False)
        eps = out["energy"].detach().cpu().numpy()
        fps = out["forces"].detach().cpu().numpy()
        ptr = b["ptr"].detach().cpu().numpy()
        for k, at in enumerate(sub_f):
            ep = float(eps[k])
            fp = fps[ptr[k] : ptr[k + 1]]
            er = float(at.info["REF_energy"])
            fr = at.arrays["REF_forces"]
            e_errs.append((ep - er) / len(at))
            df = fp - fr
            sq = float(np.sum(df * df))
            f_sq += sq
            f_cnt += df.size
            for z in zs:
                m = at.numbers == z
                if np.any(m):
                    sp_sq[z] += float(np.sum(df[m] ** 2))
                    sp_cnt[z] += int(np.sum(m)) * 3
            ct = str(at.info.get("config_type", "all"))
            ct_sq[ct] = ct_sq.get(ct, 0.0) + sq
            ct_cnt[ct] = ct_cnt.get(ct, 0) + df.size
    return {
        "f_mev": math.sqrt(f_sq / max(f_cnt, 1)) * 1000.0,
        "e_mev": float(np.sqrt(np.mean(np.square(e_errs)))) * 1000.0,
        "sp_mev": {int(z): round(math.sqrt(sp_sq[z] / max(sp_cnt[z], 1)) * 1000.0, 2) for z in zs},
        "ct_mev": {k: round(math.sqrt(ct_sq[k] / max(ct_cnt[k], 1)) * 1000.0, 2) for k in sorted(ct_sq)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--aug-xyz", type=Path, default=None)
    ap.add_argument("--dft-weight", type=float, default=2.0)
    ap.add_argument("--aug-std-scale", type=float, default=15.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--r-max", type=float, default=4.8)
    ap.add_argument("--r-max2", type=float, default=4.0)
    ap.add_argument("--lmax", type=int, default=3)
    ap.add_argument("--lfeat", type=int, default=2)
    ap.add_argument("--taper", default="16,16,12,8")
    ap.add_argument("--n-base", type=int, default=12)
    ap.add_argument("--n-base2", type=int, default=4)
    ap.add_argument("--radial-warp", type=float, default=1.5)
    ap.add_argument("--c0", type=int, default=4)
    ap.add_argument("--c1", type=int, default=4)
    ap.add_argument("--c2", type=int, default=0)
    ap.add_argument("--sym", action="store_true", default=True)
    ap.add_argument("--pol-field", action="store_true", default=True)
    ap.add_argument("--pol-gram", action="store_true", default=True)
    ap.add_argument("--sw-mp", action="store_true", default=True)
    ap.add_argument("--mp-cap", type=float, default=4.0)
    ap.add_argument("--ef-cap", type=float, default=4.0)
    ap.add_argument("--pair-mlp-width", type=int, default=0)
    ap.add_argument("--field-mlp-width", type=int, default=32)
    ap.add_argument("--readout-width", type=int, default=64)
    ap.add_argument("--readout-depth", type=int, default=1)
    ap.add_argument("--a0", type=float, default=0.6)
    ap.add_argument("--kappa", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--lr-peak", type=float, default=0.015)
    ap.add_argument("--lr-final", type=float, default=1e-4)
    ap.add_argument("--w-lr-reg", type=float, default=5e-4)
    ap.add_argument("--rad-smooth", type=float, default=1e-3)
    ap.add_argument("--wd-ro", type=float, default=1.5e-2)
    ap.add_argument("--weight-decay", type=float, default=2e-3)
    ap.add_argument("--soap-every", type=int, default=8)
    ap.add_argument("--soap-max-dim", type=int, default=128)
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda")

    dft_tr_frames = list(iread(args.data_dir / "train.xyz", ":", format="extxyz"))
    va_frames = list(iread(args.data_dir / "valid.xyz", ":", format="extxyz"))
    te_frames = list(iread(args.data_dir / "test.xyz", ":", format="extxyz"))
    zs = sorted({int(z) for a in dft_tr_frames for z in a.numbers})
    z_table = AtomicNumberTable(zs)
    phys_e0 = fit_min_norm_e0(dft_tr_frames, zs)
    e0_map = {z: float(e) for z, e in zip(zs, phys_e0)}
    rms_f = float(np.sqrt(np.mean(np.concatenate([a.arrays["REF_forces"].ravel() for a in dft_tr_frames]) ** 2)))

    tr_frames = list(dft_tr_frames)
    base_weights = [float(args.dft_weight)] * len(dft_tr_frames)
    n_aug = 0
    if args.aug_xyz is not None and args.aug_xyz.exists():
        aug_frames = list(iread(args.aug_xyz, ":", format="extxyz"))
        n_aug = len(aug_frames)
        for at in aug_frames:
            fstd = float(at.info.get("ens4_f_std_mev", 15.0))
            w_conf = 1.0 / (1.0 + (fstd / max(args.aug_std_scale, 1e-6)) ** 2)
            tr_frames.append(at)
            base_weights.append(w_conf)
    base_weights = np.array(base_weights, dtype=np.float64)
    n_atoms_arr = np.array([len(a) for a in tr_frames], dtype=np.float64)

    r_graph = max(args.r_max, args.r_max2 if args.r_max2 is not None else args.r_max)
    dft_tr_data = build_data(dft_tr_frames, z_table, r_graph, e0_map)
    tr_data = build_data(tr_frames, z_table, r_graph, e0_map)
    va_data = build_data(va_frames, z_table, r_graph, e0_map)
    te_data = build_data(te_frames, z_table, r_graph, e0_map)
    taper = [int(x) for x in args.taper.split(",")]

    model = FastLRSKStudent(
        zs=zs, r_max=args.r_max, r_max2=args.r_max2, lmax=args.lmax, lfeat=args.lfeat, taper=taper,
        n_base=args.n_base, n_base2=args.n_base2, radial_warp=args.radial_warp,
        c0=args.c0, c1=args.c1, c2=args.c2, pol_field=args.pol_field, field_mlp_width=args.field_mlp_width,
        readout_width=args.readout_width, readout_depth=args.readout_depth, sym=True,
        pol_gram=args.pol_gram, sw_mp=args.sw_mp, mp_cap=args.mp_cap, ef_cap=args.ef_cap,
        pair_mlp_width=0, film_radials=True, use_zbl_switch=True,
        kappa=args.kappa, a0=args.a0, energy_scale=rms_f,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\n============================================================\n"
        f">>> PURE-CUDA BO-LRSK N3TBU: {args.name} ({n_params:,} params, bs={args.batch_size}, 100% GPU C++)\n"
        f"    dft={len(dft_tr_frames)} + aug={n_aug} (dft_w={args.dft_weight}) | "
        f"r=({args.r_max},{args.r_max2}) l={args.lmax}/lf={args.lfeat} taper={taper} nb=({args.n_base},{args.n_base2}) warp={args.radial_warp}\n"
        f"    c=({args.c0},{args.c1},{args.c2}) pol={args.pol_field} gram={args.pol_gram} cap=({args.mp_cap},{args.ef_cap}) "
        f"ro={args.readout_depth}x{args.readout_width} wd_ro={args.wd_ro}",
        flush=True,
    )
    if args.init_from is not None and args.init_from.exists():
        ck0 = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd0 = ck0["state_dict"]
        cur = model.state_dict()
        loaded = 0
        for k, v in sd0.items():
            if k in cur and cur[k].shape == v.shape and k != "e0":
                cur[k].copy_(v.to(cur[k].dtype))
                loaded += 1
        print(f"  Warm-started {loaded}/{len(cur)} tensors from {args.init_from.name}", flush=True)
    else:
        model.calibrate_buffers(tr_data[: min(len(tr_data), 256)], bs=16)

    bs = args.batch_size
    soap_ev = args.soap_every if args.soap_every > 0 else max(4, min(16, 40 // max(bs, 4)))
    b_soap = max(0.95, 0.995 ** (bs / 4.0))
    adam_b1 = max(0.85, 0.90 ** math.sqrt(bs / 4.0))
    adam_b2 = max(0.95, 0.99 ** (bs / 4.0))
    d_ema_max = max(0.96, 0.996 ** (bs / 4.0))

    ft = CudaFullLRSKTrainer(model, dtype="f32", soap_max_dim=args.soap_max_dim)
    ft.set_optimizer(
        b_soap, b_soap, soap_ev, args.weight_decay, 1.0,
        adam_b1, adam_b2, d_ema_max, 1.0,
        args.w_lr_reg, args.rad_smooth, args.wd_ro, sp_w=None,
    )
    ft.upload_dataset(tr_data, mol_list=None, graph_weights=None, has_stress=False, device=device)

    N_tr = len(tr_data)
    spe = max(1, N_tr // bs)
    tail_ep = math.floor(args.epochs * 0.85)
    rng = np.random.default_rng(args.seed)

    def lr_at(t):
        pts = [(0.0, args.lr_peak * 0.1), (2.0, args.lr_peak), (tail_ep, args.lr_peak * 0.04), (float(args.epochs), args.lr_final)]
        for i in range(len(pts) - 1):
            if t <= pts[i + 1][0]:
                e0, v0 = pts[i]; e1, v1 = pts[i + 1]
                return v0 + (v1 - v0) * (t - e0) / max(e1 - e0, 1e-12)
        return pts[-1][1]

    p_base = base_weights / base_weights.sum()
    best_va = 1e9
    best_sd = None
    best_ep = 0

    t_tr = time.perf_counter()
    for ep in range(args.epochs):
        if ep >= 2:
            fe_np = ft.get_frame_err(N_tr).astype(np.float64)
            w = base_weights * np.power(np.maximum(fe_np, 1e-8), 1.5)
            probs = 0.6 * p_base + 0.4 * (w / w.sum())
            probs /= probs.sum()
        else:
            probs = p_base

        order = rng.choice(N_tr, size=spe * bs, replace=True, p=probs).astype(np.int32)
        # Precompute exact per-batch normalized dynamic graph weights w_g in numpy (< 0.1 ms!)
        idx_2d = order.reshape(spe, bs)
        nats_2d = n_atoms_arr[idx_2d]
        w_g = nats_2d / nats_2d.mean(axis=1, keepdims=True)
        w_conf = base_weights[idx_2d] / np.maximum(float(N_tr) * probs[idx_2d], 1e-12)
        w_g = w_g * np.sqrt(np.clip(w_conf, 0.25, 4.0))
        w_g = w_g / w_g.mean(axis=1, keepdims=True)

        t_arr = ep + np.arange(spe, dtype=np.float64) / spe
        lr_s = np.array([lr_at(float(t)) for t in t_arr], dtype=np.float64)
        we_s = np.where(t_arr < tail_ep, 2.0, 80.0).astype(np.float64)
        wf_s = np.full(spe, 100.0, dtype=np.float64)

        _, mean_lf = ft.run_epoch_weighted(order, w_g.ravel(), bs, lr_s, we_s, wf_s, update_ferr=True)

        if (ep + 1) % args.eval_every == 0 or (ep + 1) == args.epochs:
            ft.download_params_to_model(from_ema=True)
            with torch.no_grad():
                model.e0.copy_(torch.tensor(phys_e0, dtype=torch.float64, device=device))
            va = eval_split(model, va_frames, va_data, zs, device)
            mark = ""
            if va["f_mev"] < best_va:
                best_va = va["f_mev"]
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_ep = ep + 1
                mark = " *BEST*"
            tr_f = math.sqrt(max(mean_lf, 0.0)) * 1000.0
            print(
                f"    ep {ep+1:3d}/{args.epochs} ({time.perf_counter()-t_tr:5.1f}s) | trF={tr_f:5.2f} | "
                f"vaF={va['f_mev']:5.2f} (gap={va['f_mev']-tr_f:+5.2f}) sp={va['sp_mev']} ct={va['ct_mev']}{mark}",
                flush=True,
            )

    train_s = time.perf_counter() - t_tr
    if best_sd is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
    else:
        ft.download_params_to_model(from_ema=True)
    ft.close()
    torch.cuda.empty_cache()

    with torch.no_grad():
        model.e0.copy_(torch.tensor(phys_e0, dtype=torch.float64, device=device))

    # Calibrate E0 strictly on untouched DFT train frames!
    model.eval()
    A = np.zeros((len(dft_tr_frames), len(zs)), dtype=np.float64)
    y = np.zeros(len(dft_tr_frames), dtype=np.float64)
    z2i = {z: i for i, z in enumerate(zs)}
    with torch.no_grad():
        for s0 in range(0, len(dft_tr_frames), 16):
            sub_f = dft_tr_frames[s0 : s0 + 16]
            sub_d = dft_tr_data[s0 : s0 + 16]
            b = Batch.from_data_list(sub_d).to(device).to_dict()
            eps = model(b, training=False, compute_force=False, compute_stress=False)["energy"].detach().cpu().numpy()
            for j, at in enumerate(sub_f):
                k = s0 + j
                y[k] = float(at.info["REF_energy"]) - float(eps[j])
                for z in at.numbers:
                    A[k, z2i[int(z)]] += 1.0
    de0, *_ = np.linalg.lstsq(A, y, rcond=None)
    with torch.no_grad():
        model.e0.add_(torch.tensor(de0, dtype=torch.float64, device=device))

    tr_sc = eval_split(model, dft_tr_frames, dft_tr_data, zs, device)
    va_sc = eval_split(model, va_frames, va_data, zs, device)
    te_sc = eval_split(model, te_frames, te_data, zs, device)
    print(f"  FINAL (best_ep={best_ep}, train={train_s:.1f}s):", flush=True)
    for tag, sc in [("train_dft", tr_sc), ("valid", va_sc), ("test", te_sc)]:
        print(f"    [{tag:9s}] F={sc['f_mev']:5.2f} meV/A | E={sc['e_mev']:.3f} meV/at | sp={sc['sp_mev']} | ct={sc['ct_mev']}", flush=True)
    print(f"    Generalization gap: valid-train={va_sc['f_mev']-tr_sc['f_mev']:+.2f}, test-valid={te_sc['f_mev']-va_sc['f_mev']:+.2f} meV/A", flush=True)

    ckpt_path = args.out_dir / f"{args.name}.pt"
    save_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    save_args["lrsk"] = True
    save_args["pair_mlp_width"] = 0
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "energy_scale": rms_f, "args": save_args}, ckpt_path)
    (args.out_dir / f"{args.name}_summary.json").write_text(json.dumps({
        "name": args.name, "params": n_params, "best_ep": best_ep, "train_s": train_s,
        "train_dft": tr_sc, "valid": va_sc, "test": te_sc,
    }, indent=2) + "\n")
    print(f"  Saved -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
