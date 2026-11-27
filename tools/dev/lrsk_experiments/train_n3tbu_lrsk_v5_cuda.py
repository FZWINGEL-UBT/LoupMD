#!/usr/bin/env python3
"""Native-CUDA BO-LRSK v5 n3tbu_v1 Trainer (20x-40x faster, exact parity with train_n3tbu_lrsk_v4_fast.py!)."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, "/mnt/homes/zwingel/loups-phase1")
sys.path.insert(0, str(_HERE.parents[1]))

import numpy as np
import torch
from ase.io import iread
from mace.data import AtomicData, KeySpecification, config_from_atoms
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric import Batch

from p2_ilmlip_fast import fit_min_norm_e0, matrix_blocks_for_ilmlip_soap
from p2_lrsk_v4_fast import FastLRSKStudent, PackedGraphDataset
from lrsk_train_ops import FastCudaILMlipSOAP, PackedCudaLRSKDataset, cuda_forward_lrsk


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
    ap.add_argument("--r-max", type=float, default=5.0)
    ap.add_argument("--r-max2", type=float, default=4.0)
    ap.add_argument("--lmax", type=int, default=3)
    ap.add_argument("--lfeat", type=int, default=2)
    ap.add_argument("--taper", default="16,16,12,8")
    ap.add_argument("--n-base", type=int, default=12)
    ap.add_argument("--n-base2", type=int, default=6)
    ap.add_argument("--radial-warp", type=float, default=0.0)
    ap.add_argument("--c0", type=int, default=4)
    ap.add_argument("--c1", type=int, default=4)
    ap.add_argument("--c2", type=int, default=0)
    ap.add_argument("--sym", action="store_true", default=False)
    ap.add_argument("--pol-field", action="store_true", default=False)
    ap.add_argument("--pol-gram", action="store_true", default=False)
    ap.add_argument("--sw-mp", action="store_true", default=False)
    ap.add_argument("--mp-cap", type=float, default=4.0)
    ap.add_argument("--ef-cap", type=float, default=4.0)
    ap.add_argument("--pair-mlp-width", type=int, default=0)
    ap.add_argument("--field-mlp-width", type=int, default=32)
    ap.add_argument("--readout-width", type=int, default=64)
    ap.add_argument("--readout-depth", type=int, default=2)
    ap.add_argument("--a0", type=float, default=0.6)
    ap.add_argument("--kappa", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--lr-peak", type=float, default=0.015)
    ap.add_argument("--lr-final", type=float, default=1e-4)
    ap.add_argument("--w-lr-reg", type=float, default=5e-4)
    ap.add_argument("--rad-smooth", type=float, default=1e-3)
    ap.add_argument("--wd-ro", type=float, default=1.5e-2)
    ap.add_argument("--weight-decay", type=float, default=2e-3)
    ap.add_argument("--soap-every", type=int, default=0)
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

    dft_tr_data = build_data(dft_tr_frames, z_table, args.r_max, e0_map)
    tr_data = build_data(tr_frames, z_table, args.r_max, e0_map)
    va_data = build_data(va_frames, z_table, args.r_max, e0_map)
    te_data = build_data(te_frames, z_table, args.r_max, e0_map)
    taper = [int(x) for x in args.taper.split(",")]

    model = FastLRSKStudent(
        zs=zs, r_max=args.r_max, r_max2=args.r_max2, lmax=args.lmax, lfeat=args.lfeat, taper=taper,
        n_base=args.n_base, n_base2=args.n_base2, radial_warp=args.radial_warp,
        c0=args.c0, c1=args.c1, c2=args.c2, pol_field=args.pol_field, field_mlp_width=args.field_mlp_width,
        readout_width=args.readout_width, readout_depth=args.readout_depth, sym=args.sym,
        pol_gram=args.pol_gram, sw_mp=args.sw_mp, mp_cap=args.mp_cap, ef_cap=args.ef_cap,
        pair_mlp_width=args.pair_mlp_width,
        film_radials=True, use_zbl_switch=True, kappa=args.kappa, a0=args.a0, energy_scale=rms_f,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    use_cuda_ops = bool(model.sym and model.pair_mlp_width == 0 and model.n_base <= 16 and model.n_base2 <= 16)
    print(
        f"\n============================================================\n"
        f">>> CUDA LRSK v5 N3TBU: {args.name} ({n_params:,} params, bs={args.batch_size}, cuda_ops={use_cuda_ops}) | dft={len(dft_tr_frames)} + aug={n_aug} (dft_w={args.dft_weight})\n"
        f"    r=({args.r_max},{args.r_max2}) l={args.lmax}/lf={args.lfeat} taper={taper} nb=({args.n_base},{args.n_base2}) "
        f"c=({args.c0},{args.c1},{args.c2}) pol={args.pol_field} gram={args.pol_gram} cap=({args.mp_cap},{args.ef_cap}) bo_mlp={args.pair_mlp_width} "
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

    if use_cuda_ops:
        packed_tr = PackedCudaLRSKDataset(tr_data, None, model, device=device, has_stress=False)
    else:
        packed_tr = PackedGraphDataset(tr_data, None, device=device, has_stress=False)

    bs = args.batch_size
    soap_ev = args.soap_every if args.soap_every > 0 else max(6, min(16, 48 // max(bs, 4)))
    b_soap = max(0.95, 0.995 ** (bs / 4.0))
    adam_b1 = max(0.85, 0.90 ** math.sqrt(bs / 4.0))
    adam_b2 = max(0.95, 0.99 ** (bs / 4.0))
    d_ema_max = max(0.96, 0.996 ** (bs / 4.0))
    opt = FastCudaILMlipSOAP(
        model, matrix_blocks_for_ilmlip_soap,
        soap_betas=(b_soap, b_soap), soap_every=soap_ev,
        betas=(adam_b1, adam_b2), weight_decay=args.weight_decay,
    )

    params_list = [p for p in model.parameters() if p.requires_grad]
    ema = [p.data.clone() for p in params_list]
    N_tr = len(tr_data)
    spe = max(1, N_tr // bs)
    tail_ep = math.floor(args.epochs * 0.85)
    rng = np.random.default_rng(args.seed)

    frame_err_gpu = torch.ones(N_tr, dtype=torch.float32, device=device)
    frame_seen_gpu = torch.zeros(N_tr, dtype=torch.bool, device=device)
    base_w_gpu = torch.tensor(base_weights, dtype=torch.float32, device=device)

    ema_updates = 0
    best_va = 1e9
    best_ema = None
    best_ep = 0

    def lr_at(t):
        pts = [(0.0, args.lr_peak * 0.1), (2.0, args.lr_peak), (tail_ep, args.lr_peak * 0.04), (float(args.epochs), args.lr_final)]
        for i in range(len(pts) - 1):
            if t <= pts[i + 1][0]:
                e0, v0 = pts[i]; e1, v1 = pts[i + 1]
                return v0 + (v1 - v0) * (t - e0) / max(e1 - e0, 1e-12)
        return pts[-1][1]

    k4_w = (torch.arange(model.n_base2, dtype=torch.float32, device=device) ** 4)[None, None, :]
    k2_w = (torch.arange(model.n_base, dtype=torch.float32, device=device) ** 2)[None, :]
    ro_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and ("readout_layers" in name or "pol_W0" in name or "W_q" in name or "W_g" in name or "u_Z" in name or "pol_u" in name or "bo_" in name)
    ]

    p_base = base_weights / base_weights.sum()
    t_tr = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        if ep >= 2:
            fe_np = frame_err_gpu.detach().cpu().numpy().astype(np.float64)
            w = base_weights * np.power(np.maximum(fe_np, 1e-8), 1.5)
            probs = 0.6 * p_base + 0.4 * (w / w.sum())
            probs /= probs.sum()
        else:
            probs = p_base
        order = rng.choice(N_tr, size=spe * bs, replace=True, p=probs)
        probs_gpu = torch.tensor(probs, dtype=torch.float32, device=device)
        ep_lf_t = torch.zeros((), dtype=torch.float32, device=device)

        for st in range(spe):
            t = ep + st / spe
            lr = lr_at(t)
            we = 2.0 if t < tail_ep else 80.0
            idx = order[st * bs : (st + 1) * bs]
            idx_t = torch.tensor(idx, dtype=torch.long, device=device)
            d = packed_tr.make_batch(idx)

            for p in params_list:
                p.grad = None
            if use_cuda_ops:
                out = cuda_forward_lrsk(model, d, training=True, compute_force=True, compute_stress=False)
            else:
                out = model(d, training=True, compute_force=True, compute_stress=False)
            nats = d["n_atoms_f"]
            e_sq = ((out["energy"].float() - d["energy"]) / nats) ** 2
            f_sq_a = ((out["forces"] - d["forces"]) ** 2).mean(dim=-1)
            f_sq_g = torch.zeros(len(idx), dtype=f_sq_a.dtype, device=device)
            f_sq_g.index_add_(0, d["batch"], f_sq_a)
            f_sq_g = f_sq_g / nats

            with torch.no_grad():
                vals = f_sq_g.detach()
                seen = frame_seen_gpu[idx_t]
                old_v = frame_err_gpu[idx_t]
                frame_err_gpu[idx_t] = torch.where(seen, 0.5 * old_v + 0.5 * vals, vals)
                frame_seen_gpu[idx_t] = True

            w_g = nats / nats.mean()
            w_conf_t = base_w_gpu[idx_t] / torch.clamp(float(N_tr) * probs_gpu[idx_t], min=1e-12)
            w_g = w_g * torch.sqrt(torch.clamp(w_conf_t, 0.25, 4.0))
            w_g = w_g / w_g.mean()

            loss = we * (e_sq * w_g).mean() + 100.0 * (f_sq_g * w_g).mean()
            if args.w_lr_reg > 0.0:
                loss = loss + args.w_lr_reg * ((model.W_lr_rad ** 2) * k4_w).mean()
            if args.rad_smooth > 0.0:
                loss = loss + args.rad_smooth * (((model.C1_R ** 2) * k2_w).mean() + ((model.C1_B ** 2) * k2_w).mean())
            if args.wd_ro > 0.0 and ro_params:
                loss = loss + args.wd_ro * sum((p ** 2).mean() for p in ro_params)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_list, 1.0)
            opt.step(lr)

            ema_updates += 1
            d_ema = min(d_ema_max, (1.0 + ema_updates) / (10.0 + ema_updates))
            with torch.no_grad():
                torch._foreach_mul_(ema, d_ema)
                torch._foreach_add_(ema, params_list, alpha=1.0 - d_ema)
            ep_lf_t.add_((f_sq_g * w_g).mean().detach())

        if (ep + 1) % args.eval_every == 0 or (ep + 1) == args.epochs:
            saved = [p.data.clone() for p in params_list]
            with torch.no_grad():
                for e_t, p in zip(ema, params_list):
                    p.data.copy_(e_t)
                model.e0.copy_(torch.tensor(phys_e0, dtype=torch.float64, device=device))
            va = eval_split(model, va_frames, va_data, zs, device)
            mark = ""
            if va["f_mev"] < best_va:
                best_va = va["f_mev"]
                best_ema = [e_t.clone() for e_t in ema]
                best_ep = ep + 1
                mark = " *BEST*"
            tr_f = math.sqrt(float(ep_lf_t.item()) / spe) * 1000.0
            print(
                f"    ep {ep+1:3d}/{args.epochs} ({time.perf_counter()-t_tr:5.1f}s) | trF={tr_f:5.2f} | "
                f"vaF={va['f_mev']:5.2f} (gap={va['f_mev']-tr_f:+5.2f}) sp={va['sp_mev']} ct={va['ct_mev']}{mark}",
                flush=True,
            )
            with torch.no_grad():
                for s_t, p in zip(saved, params_list):
                    p.data.copy_(s_t)
                model.e0.zero_()

    with torch.no_grad():
        for e_t, p in zip(best_ema or ema, params_list):
            p.data.copy_(e_t)
        model.e0.copy_(torch.tensor(phys_e0, dtype=torch.float64, device=device))

    model.eval()
    A = np.zeros((len(dft_tr_frames), len(zs)), dtype=np.float64)
    y = np.zeros(len(dft_tr_frames), dtype=np.float64)
    z2i = {z: i for i, z in enumerate(zs)}
    with torch.no_grad():
        for k, (at, ad) in enumerate(zip(dft_tr_frames, dft_tr_data)):
            b = Batch.from_data_list([ad]).to(device).to_dict()
            ep_p = float(model(b, training=False, compute_force=False, compute_stress=False)["energy"][0].cpu())
            y[k] = float(at.info["REF_energy"]) - ep_p
            for z in at.numbers:
                A[k, z2i[int(z)]] += 1.0
    de0, *_ = np.linalg.lstsq(A, y, rcond=None)
    with torch.no_grad():
        model.e0.add_(torch.tensor(de0, dtype=torch.float64, device=device))

    train_s = time.perf_counter() - t_tr
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
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "energy_scale": rms_f, "args": save_args}, ckpt_path)
    (args.out_dir / f"{args.name}_summary.json").write_text(json.dumps({
        "name": args.name, "params": n_params, "best_ep": best_ep, "train_s": train_s,
        "train_dft": tr_sc, "valid": va_sc, "test": te_sc,
    }, indent=2))
    print(f"  Saved -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
