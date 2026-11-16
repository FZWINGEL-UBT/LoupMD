#!/usr/bin/env python3
"""Force-call timing: symmetrix main (b907f04) MACEFoundation against LoupMD, MACE-OMAT-0 medium, SrTiO3 n x n x n
from a 300 K LoupMD Langevin snapshot.

  python sx_speed.py prep    --scratch DIR [--sizes 6,10,14]
  python sx_speed.py run     --scratch DIR --label L --impl loup|loup-generic|sx [--dtype D --json FILE] --out F.jsonl
  python sx_speed.py summary --scratch DIR --out F.jsonl

One force call = positions in (host -> device), energy, forces and virial / stress out (device -> host), as each
library's Python entry point does it: LoupEngine.evaluate(x, cell, compute_stress=True) and
_mace_foundation.MACEFoundation.compute(x, numbers, cell, pbc). Every timed call gets fresh Gaussian noise (sigma
0.003 A) on the snapshot, far below either skin / 2 (LoupMD 0.25 A, symmetrix 0.5 A), so no call rebuilds a
neighbour list and no call can be served from a cache; 20 warm-up calls (list build, graph capture, tuning) are not
timed. LoupMD rows also report its device-only force pass (profile_force) and its MD step (Langevin 300 K, 1 fs,
median of 3 x 256 steps after 100), for context: symmetrix has no MD loop of its own outside LAMMPS.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.path.insert(0, "/mnt/d/Margraf/LoupMD/python")

import numpy as np  # noqa: E402

MODEL = "/home/fzwin/models/mace-omat-0-medium.model"
SIGMA = 0.003
WARMUP = 20
CALLS = 100


def srtio3(n):
    from ase import Atoms

    a = 3.905
    return Atoms(symbols=["Sr", "Ti", "O", "O", "O"],
                 scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
                 cell=[a, a, a], pbc=True).repeat((n, n, n))


def snapshot_path(scratch):
    return os.path.join(scratch, "srtio3_300K.npz")


def prep(a):
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    from loupmd import LoupEngine, ensure_loup_bundle

    out = {}
    eng = None
    for n in a.sizes:
        at = srtio3(n)
        MaxwellBoltzmannDistribution(at, temperature_K=300.0, rng=np.random.default_rng(42))
        if eng is None:
            eng = LoupEngine(ensure_loup_bundle(MODEL, at.get_atomic_numbers().tolist()), device_id=0)
        eng.set_system(at.get_atomic_numbers(), at.get_masses(), skin=0.5)
        eng.set_state(at.positions, at.get_velocities(), at.cell.array)
        st = eng.run_langevin(500, dt_fs=1.0, temp_k=300.0, friction_per_fs=0.01, seed=1)
        pos, vel, _, _ = eng.get_state()
        at.positions = pos
        at.wrap()
        out[f"pos{n}"] = at.positions.copy()
        out[f"vel{n}"] = vel
        out[f"cell{n}"] = at.cell.array.copy()
        out[f"Z{n}"] = at.get_atomic_numbers()
        out[f"m{n}"] = at.get_masses()
        print(f"SrTiO3 {n}x{n}x{n}: {len(at)} atoms, 500 Langevin steps, T {st.final_temperature_k:.0f} K", flush=True)
    np.savez(snapshot_path(a.scratch), sizes=np.array(a.sizes), **out)


def timed(fn, base, rng):
    for _ in range(WARMUP):
        fn(base + rng.normal(0.0, SIGMA, base.shape))
    ts = []
    for _ in range(CALLS):
        x = base + rng.normal(0.0, SIGMA, base.shape)
        t0 = time.perf_counter()
        fn(x)
        ts.append(time.perf_counter() - t0)
    return np.array(ts) * 1e3


def run(a):
    snap = np.load(snapshot_path(a.scratch))
    sizes = [int(n) for n in snap["sizes"]]
    rng = np.random.default_rng(7)
    rows = []
    if a.impl in ("loup", "loup-generic"):
        from loupmd import LoupEngine, ensure_loup_bundle

        lib = os.environ.get("LOUP_LIB") or "default"
        t0 = time.time()
        eng = LoupEngine(ensure_loup_bundle(MODEL, [8, 22, 38], generic=(a.impl == "loup-generic")), device_id=0)
        load_s = time.time() - t0
        for n in sizes:
            base, cell, Z, m, vel = (snap[f"pos{n}"], snap[f"cell{n}"], snap[f"Z{n}"], snap[f"m{n}"],
                                     snap[f"vel{n}"])
            eng.set_system(Z, m, skin=0.5)

            def fn(x):
                return eng.evaluate(x, cell, compute_stress=True)

            e0, f0, _ = fn(base)
            ts = timed(fn, base, rng)
            eng.set_state(base, vel, cell)
            prof_us = eng.profile_force(50, 200)
            eng.set_state(base, vel, cell)
            eng.run_langevin(100, dt_fs=1.0, temp_k=300.0, friction_per_fs=0.01, seed=1)
            md = statistics.median(eng.run_langevin(256, dt_fs=1.0, temp_k=300.0, friction_per_fs=0.01,
                                                    seed=b + 2).step_ms for b in range(3))
            np.save(os.path.join(a.scratch, f"F_{a.label}_{n}.npy"), f0)
            rows.append(dict(label=a.label, impl=a.impl, lib=lib, n_atoms=len(Z), load_s=load_s, energy=e0,
                             call_ms=float(np.median(ts)), call_p10=float(np.percentile(ts, 10)),
                             call_p90=float(np.percentile(ts, 90)), force_pass_ms=prof_us / 1e3, md_step_ms=md))
        del eng
    else:
        sys.path.insert(0, a.pkg)
        from symmetrix._mace_foundation import MACEFoundation

        t0 = time.time()
        ev = MACEFoundation(a.json, "cuda", a.dtype, 1.0)
        load_s = time.time() - t0
        pbc = np.array([True, True, True])
        for n in sizes:
            base, cell = snap[f"pos{n}"], np.ascontiguousarray(snap[f"cell{n}"])
            Z = np.ascontiguousarray(snap[f"Z{n}"], dtype=np.int32)

            def fn(x):
                return ev.compute(x, Z, cell, pbc, False)

            e0, f0, _, _ = fn(base)
            ts = timed(fn, base, rng)
            np.save(os.path.join(a.scratch, f"F_{a.label}_{n}.npy"), f0)
            rows.append(dict(label=a.label, impl=a.impl, dtype=a.dtype, json=os.path.basename(a.json),
                             n_atoms=len(Z), load_s=load_s, energy=float(e0), call_ms=float(np.median(ts)),
                             call_p10=float(np.percentile(ts, 10)), call_p90=float(np.percentile(ts, 90))))
        del ev
    with open(a.out, "a") as f:
        for r in rows:
            r["pass"] = a.pass_tag
            f.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)


def summary(a):
    rows = [json.loads(line) for line in open(a.out) if line.strip()]
    labels = list(dict.fromkeys(r["label"] for r in rows))
    sizes = sorted({r["n_atoms"] for r in rows})
    print("| implementation | atoms | force call ms (pass 1 / pass 2) | mean ms | us/atom | vs LoupMD default "
          "| max abs dF vs LoupMD default (eV/A) | dE vs LoupMD default (meV/atom) |")
    print("|---|---|---|---|---|---|---|---|")
    ref = {}
    for n in sizes:
        for lab in labels:
            rs = [r for r in rows if r["label"] == lab and r["n_atoms"] == n]
            if not rs:
                continue
            mean = statistics.mean(r["call_ms"] for r in rs)
            if lab == "loup":
                ref[n] = (mean, rs[0]["energy"])
            side = int(round((n / 5) ** (1 / 3)))
            f_lab = np.load(os.path.join(a.scratch, f"F_{lab}_{side}.npy"))
            f_ref = np.load(os.path.join(a.scratch, f"F_loup_{side}.npy"))
            df = float(np.abs(f_lab - f_ref).max())
            de = (rs[0]["energy"] - ref[n][1]) * 1e3 / n if n in ref else float("nan")
            ratio = mean / ref[n][0] if n in ref else float("nan")
            passes = " / ".join(f"{r['call_ms']:.2f}" for r in rs)
            print(f"| {lab} | {n} | {passes} | {mean:.2f} | {mean * 1e3 / n:.2f} | {ratio:.2f} | {df:.2e} | "
                  f"{de:+.4f} |")
    print()
    print("| LoupMD | atoms | device-only force pass ms | MD step ms (Langevin) | pass |")
    print("|---|---|---|---|---|")
    for r in rows:
        if r["impl"].startswith("loup"):
            print(f"| {r['label']} | {r['n_atoms']} | {r['force_pass_ms']:.2f} | {r['md_step_ms']:.2f} | {r['pass']} |")
    print()
    print("| symmetrix | model load s |")
    print("|---|---|")
    for lab in labels:
        rs = [r for r in rows if r["label"] == lab and r["impl"] == "sx"]
        if rs:
            print(f"| {lab} | {statistics.mean(r['load_s'] for r in rs):.1f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prep", "run", "summary"))
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--sizes", default="6,10,14")
    ap.add_argument("--label")
    ap.add_argument("--impl", choices=("loup", "loup-generic", "sx"))
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--json")
    ap.add_argument("--pkg", default="/home/fzwin/src/symmetrix-eval/b907f04")
    ap.add_argument("--out")
    ap.add_argument("--pass-tag", default="1")
    a = ap.parse_args()
    a.sizes = [int(s) for s in a.sizes.split(",")]
    {"prep": prep, "run": run, "summary": summary}[a.mode](a)


if __name__ == "__main__":
    main()
