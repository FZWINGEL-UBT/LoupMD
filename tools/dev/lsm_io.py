#!/usr/bin/env python3
"""Binary Loup-S Model (.lsm) I/O helper for the standalone zero-Torch C++/CUDA trainer (loupstrain_cli).

Format LSM1 (little-endian):
  char    magic[4]       = "LSM1"
  int32   version        = 1
  int32   dtype          = 0 (float32) or 1 (float64)
  int32   S, C, H, NB, hid, K, K3, K2, p_cut, zbl_p
  double  r_max, prefactor, act, scale, shift, avg0, avg1, zbl_a_exp, zbl_a_pref
  double  zbl_c[4]
  int32   Z[S]
  double  E0[S]
  double  cov_r[S]       (covalent_radii[Z[s]] cast to float64 from model.pair_repulsion_fn)
  double  bessel_w[NB]
  int64   param_numel[26]
  float32/float64 param_data[sum(param_numel)]  (in PARAM_NAMES order)
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HEADER_FMT = "<4s12i13d"  # magic(4s) + 12 int32 + 13 float64 (9 scalars + zbl_c[4])
NP = 26


def dump_model(model_path: str, lsm_path: str) -> None:
    import torch
    from loupgym.fasttrain.cuda import PARAM_NAMES
    from loupgym.fasttrain.model import check_supported, check_poly_terms, silu_norm_constant

    m = torch.load(model_path, map_location="cpu", weights_only=False)
    check_supported(m)
    check_poly_terms(m)

    allp = dict(m.named_parameters())
    params = [allp[n].detach().cpu().contiguous() for n in PARAM_NAMES]
    dt = params[0].dtype
    if dt == torch.float32:
        dtype_id = 0
    elif dt == torch.float64:
        dtype_id = 1
    else:
        raise ValueError(f"Unsupported parameter dtype: {dt}")

    zs = [int(z) for z in m.atomic_numbers.detach().cpu().tolist()]
    S = len(zs)
    C = int(m.products[0].linear.irreps_out.dim)
    net = m.interactions[0].conv_tp_weights
    H = int(net.layer0.weight.shape[1])
    be = m.radial_embedding.bessel_fn
    bw = [float(x) for x in be.bessel_weights.detach().double().cpu().tolist()]
    NB = len(bw)
    hid = int(m.readouts[1].linear_1.weight.numel() // C)
    sc = m.products[0].symmetric_contractions.contractions[0]
    K3 = int(sc.weights_max.shape[1])
    K2 = int(sc.weights[0].shape[1])
    K1 = int(sc.weights[1].shape[1])
    K = K3 + K2 + K1
    p_cut = int(m.radial_embedding.cutoff_fn.p)
    r_max = float(m.radial_embedding.cutoff_fn.r_max)
    prefactor = float(be.prefactor)
    act = float(silu_norm_constant())
    scale = float(m.scale_shift.scale.detach().double().item())
    shift = float(m.scale_shift.shift.detach().double().item())
    avg0 = float(m.interactions[0].avg_num_neighbors)
    avg1 = float(m.interactions[1].avg_num_neighbors)

    zbl = m.pair_repulsion_fn
    zbl_p = int(zbl.p.item())
    zbl_c = [float(x) for x in zbl.c.detach().double().cpu().tolist()]
    assert len(zbl_c) == 4
    zbl_a_exp = float(zbl.a_exp.detach().double().item())
    zbl_a_pref = float(zbl.a_prefactor.detach().double().item())
    cov_table = zbl.covalent_radii.detach().double().cpu().tolist()
    cov_r = [float(cov_table[z]) for z in zs]
    e0 = [float(x) for x in m.atomic_energies_fn.atomic_energies.detach().double().cpu().reshape(-1).tolist()]
    assert len(e0) == S

    numels = [int(p.numel()) for p in params]
    with open(lsm_path, "wb") as fh:
        fh.write(struct.pack(
            HEADER_FMT,
            b"LSM1", 1, dtype_id,
            S, C, H, NB, hid, K, K3, K2, p_cut, zbl_p,
            r_max, prefactor, act, scale, shift, avg0, avg1, zbl_a_exp, zbl_a_pref,
            zbl_c[0], zbl_c[1], zbl_c[2], zbl_c[3],
        ))
        fh.write(struct.pack(f"<{S}i", *zs))
        fh.write(struct.pack(f"<{S}d", *e0))
        fh.write(struct.pack(f"<{S}d", *cov_r))
        fh.write(struct.pack(f"<{NB}d", *bw))
        fh.write(struct.pack(f"<{NP}q", *numels))
        for p in params:
            fh.write(p.numpy().tobytes())
    print(f"Wrote {lsm_path}: S={S} Z={zs} C={C} H={H} NB={NB} hid={hid} K={K}({K3},{K2},{K1}) "
          f"r_max={r_max} p_cut={p_cut} zbl_p={zbl_p} scale={scale:.6g} shift={shift:.6g} "
          f"avg=({avg0:.4f},{avg1:.4f}) total_params={sum(numels)}")


def apply_lsm(template_model: str, lsm_path: str, out_model: str) -> None:
    import numpy as np
    import torch
    from loupgym.fasttrain.cuda import PARAM_NAMES

    with open(lsm_path, "rb") as fh:
        hdr_sz = struct.calcsize(HEADER_FMT)
        hdr = struct.unpack(HEADER_FMT, fh.read(hdr_sz))
        magic, version, dtype_id = hdr[0], hdr[1], hdr[2]
        assert magic == b"LSM1" and version == 1, f"Bad LSM header: {magic!r} v{version}"
        S, C, H, NB, hid, K, K3, K2, p_cut, zbl_p = hdr[3:13]
        r_max, prefactor, act, scale, shift, avg0, avg1, zbl_a_exp, zbl_a_pref = hdr[13:22]
        zs = struct.unpack(f"<{S}i", fh.read(4 * S))
        e0 = struct.unpack(f"<{S}d", fh.read(8 * S))
        _cov_r = struct.unpack(f"<{S}d", fh.read(8 * S))
        _bw = struct.unpack(f"<{NB}d", fh.read(8 * NB))
        numels = struct.unpack(f"<{NP}q", fh.read(8 * NP))
        np_dt = np.float32 if dtype_id == 0 else np.float64
        t_dt = torch.float32 if dtype_id == 0 else torch.float64
        arrays = [np.frombuffer(fh.read(n * np.dtype(np_dt).itemsize), dtype=np_dt).copy() for n in numels]

    m = torch.load(template_model, map_location="cpu", weights_only=False).to(t_dt)
    allp = dict(m.named_parameters())
    with torch.no_grad():
        for name, arr in zip(PARAM_NAMES, arrays):
            p = allp[name]
            assert p.numel() == arr.size, f"Shape mismatch on {name}: {p.numel()} vs {arr.size}"
            p.copy_(torch.from_numpy(arr).view_as(p))
        m.scale_shift.scale.fill_(scale)
        m.scale_shift.shift.fill_(shift)
        m.interactions[0].avg_num_neighbors = avg0
        m.interactions[1].avg_num_neighbors = avg1
        if hasattr(m, "r_max"):
            if hasattr(m.r_max, "fill_"):
                m.r_max.fill_(r_max)
            else:
                m.r_max = r_max
        m.radial_embedding.cutoff_fn.r_max.fill_(r_max) if hasattr(m.radial_embedding.cutoff_fn.r_max, "fill_") else setattr(m.radial_embedding.cutoff_fn, "r_max", r_max)
        m.radial_embedding.bessel_fn.bessel_weights.copy_(
            torch.tensor(_bw, dtype=m.radial_embedding.bessel_fn.bessel_weights.dtype)
        )
        if hasattr(m.radial_embedding.bessel_fn.prefactor, "fill_"):
            m.radial_embedding.bessel_fn.prefactor.fill_(prefactor)
        else:
            m.radial_embedding.bessel_fn.prefactor = prefactor
        m.atomic_energies_fn.atomic_energies.copy_(
            torch.tensor(e0, dtype=m.atomic_energies_fn.atomic_energies.dtype).view_as(m.atomic_energies_fn.atomic_energies)
        )
    torch.save(m, out_model)
    print(f"Applied {lsm_path} ({sum(numels)} params, scale={scale:.6g}, avg={avg0:.4f}) -> {out_model}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert between mace-torch .model and Loup-S .lsm binary.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_d = sub.add_parser("dump", help="Dump mace-torch .model to .lsm")
    p_d.add_argument("model")
    p_d.add_argument("lsm")
    p_a = sub.add_parser("apply", help="Apply .lsm weights into a template .model")
    p_a.add_argument("template_model")
    p_a.add_argument("lsm")
    p_a.add_argument("out_model")
    args = ap.parse_args()
    if args.cmd == "dump":
        dump_model(args.model, args.lsm)
    else:
        apply_lsm(args.template_model, args.lsm, args.out_model)


if __name__ == "__main__":
    main()
