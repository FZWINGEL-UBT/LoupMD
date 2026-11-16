#!/usr/bin/env python3
"""LoupMD's 3x float32 accuracy gate (python/test_gate.py, oxide set: NaCl 2x2x2, SrTiO3 3x3x3, NbCl5 2x2x2, two
strained / rattled configurations each) applied to symmetrix main (b907f04) MACEFoundation, through its ASE calculator
(symmetrix.experimental.MACEFoundation), with LoupMD's default and generic paths evaluated alongside.

  python sx_gate.py --json-dir DIR [--points 1024,256] [--dtypes float64,float32,float32-mixed]

The MACE float64 / float32 references are test_gate.py's (same cache key, so its cached references are reused). Per
configuration: dE = |E - E64| / atom (meV), dF = max |F - F64| (eV/A), dS = max |S - S64| (eV/A^3); pass when dE <=
max(3 dE32, 0.01), dF <= max(3 dF32, 1e-5), dS <= max(3 dS32, 1e-7); the ratios are dF / dF32 and dS / dS32.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.path.insert(0, "/mnt/d/Margraf/LoupMD/python")

import numpy as np  # noqa: E402

import test_gate as tg  # noqa: E402
from loupmd import LoupEngine, ensure_loup_bundle  # noqa: E402

MODEL = "/home/fzwin/models/mace-omat-0-medium.model"
SPECIES = {"NaCl": (11, 17), "SrTiO3": (8, 22, 38), "NbCl5": (17, 41)}


def json_path(json_dir, family, points):
    return os.path.join(json_dir, f"omat0m-{'-'.join(map(str, SPECIES[family]))}-{points}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--pkg", default="/home/fzwin/src/symmetrix-eval/b907f04")
    ap.add_argument("--points", default="1024,256")
    ap.add_argument("--dtypes", default="float64,float32,float32-mixed")
    a = ap.parse_args()
    sys.path.insert(0, a.pkg)
    from symmetrix.experimental import MACEFoundation

    configs = tg.make_structures()
    cache = tg.ref_cache_path(MODEL, None, "oxide", "cuda", configs, (None, None))
    refs = tg.load_refs(cache, len(configs))
    if refs is None:
        print("computing MACE references (no cache)", flush=True)
        ref64 = tg.eval_mace(MODEL, "float64", configs)
        ref32 = tg.eval_mace(MODEL, "float32", configs)
        refs = (ref64, ref32, [s for _, _, s in ref64])
        tg.save_refs(cache, *refs)
    else:
        print(f"MACE references from {cache}", flush=True)
    ref64, ref32, sref = refs
    import torch

    torch.cuda.empty_cache()

    candidates = [("LoupMD default", "loup", None, None), ("LoupMD generic", "loup-generic", None, None)]
    for points in [int(p) for p in a.points.split(",")]:
        for dtype in a.dtypes.split(","):
            candidates.append((f"symmetrix {dtype} {points}", "sx", dtype, points))

    print(f"\n{'config':16s} {'candidate':30s} | {'dE meV/at':>10s} | {'dF eV/A':>9s} {'ratio':>6s} | "
          f"{'dS eV/A3':>9s} {'ratio':>6s} | pass", flush=True)
    summary = []
    for label, impl, dtype, points in candidates:
        calcs = {}
        worst_f = worst_s = 0.0
        all_ok = True
        for (name, at), (e64, f64, s64), (e32, f32, s32), s_ref in zip(configs, ref64, ref32, sref):
            family = name.split("_")[0]
            zs = SPECIES[family]
            if family not in calcs:
                if impl == "sx":
                    calcs[family] = MACEFoundation(json_path(a.json_dir, family, points), "cuda", dtype, 1.0)
                else:
                    calcs[family] = LoupEngine(ensure_loup_bundle(MODEL, list(zs), generic=(impl == "loup-generic")),
                                               device_id=0)
            c = calcs[family]
            if impl == "sx":
                x = at.copy()
                x.calc = c
                e = float(x.get_potential_energy())
                f = np.asarray(x.get_forces(), dtype=np.float64)
                s = np.asarray(x.get_stress(voigt=False), dtype=np.float64)
            else:
                c.set_system(at.get_atomic_numbers(), at.get_masses(), skin=0.5)
                e, f, s = c.evaluate(at.positions, at.cell.array, compute_stress=True)
            nat = len(at)
            de = abs(e - e64) * 1000.0 / nat
            de32 = abs(e32 - e64) * 1000.0 / nat
            df = float(np.abs(f - f64).max())
            df32 = float(np.abs(f32 - f64).max())
            ds = float(np.abs(s - s_ref).max())
            ds32 = float(np.abs(s32 - s64).max())
            rf = df / max(df32, 1e-5 / 3.0)
            rs = ds / max(ds32, 1e-7 / 3.0)
            ok = de <= max(3.0 * de32, 0.01) and df <= max(3.0 * df32, 1e-5) and ds <= max(3.0 * ds32, 1e-7)
            all_ok = all_ok and ok
            worst_f, worst_s = max(worst_f, rf), max(worst_s, rs)
            print(f"{name:16s} {label:30s} | {de:10.5f} | {df:9.2e} {rf:6.2f} | {ds:9.2e} {rs:6.2f} | "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
        summary.append((label, worst_f, worst_s, all_ok))
        del calcs
        torch.cuda.empty_cache()

    print("\n| candidate | worst dF ratio | worst dS ratio | 3x gate |")
    print("|---|---|---|---|")
    for label, wf, ws, ok in summary:
        print(f"| {label} | {wf:.2f} | {ws:.2f} | {'pass' if ok else 'FAIL'} |")


if __name__ == "__main__":
    main()
