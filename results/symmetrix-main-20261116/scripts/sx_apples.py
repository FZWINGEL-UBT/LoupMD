#!/usr/bin/env python3
"""Second round, force call against force call at equal skin, and a kernel profile of one force call.

  python sx_apples.py time   --impl loup|loup-generic|sx [--dtype D --json F] --skin S --label L --out F.jsonl
  python sx_apples.py prof   --impl ... --skin S --size 10 --calls 20      (run under nsys, see job_apples.sh)
  python sx_apples.py report --scratch DIR --out F.jsonl --labels L1,L2,...

Both libraries get the same 300 K SrTiO3 snapshots (sx_speed.py prep), the same noise sequence (fresh 0.003 A
Gaussian noise per call, seed 7), the same neighbour-list skin and the same outputs (energy, forces, virial / stress),
through their Python entry points (LoupEngine.evaluate, _mace_foundation.MACEFoundation.compute). 20 warm-up calls,
then 100 timed calls; the median is reported.

prof: after the warm-up, cudaProfilerStart, `calls` force calls separated by 3 ms host sleeps (so the trace splits into
calls at gaps over 1.5 ms), cudaProfilerStop. report: per implementation the device span of a call (first operation
start to last operation end, memcpys included), the kernel time per call, and every kernel's time per call.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.path.insert(0, "/mnt/d/Margraf/LoupMD/python")

import numpy as np  # noqa: E402

MODEL = "/home/fzwin/models/mace-omat-0-medium.model"
SIGMA = 0.003
WARMUP = 20
CALLS = 100
GAP_NS = 1_500_000


def load_snapshot(scratch):
    return np.load(os.path.join(scratch, "srtio3_300K.npz"))


def make_fn(a, snap, n):
    """The force call of implementation a.impl on SrTiO3 n^3 at skin a.skin, and the snapshot positions."""
    base, cell = snap[f"pos{n}"], np.ascontiguousarray(snap[f"cell{n}"])
    Z, m = snap[f"Z{n}"], snap[f"m{n}"]
    if a.impl in ("loup", "loup-generic"):
        if not hasattr(a, "_eng"):
            from loupmd import LoupEngine, ensure_loup_bundle

            a._eng = LoupEngine(ensure_loup_bundle(MODEL, [8, 22, 38], generic=(a.impl == "loup-generic")), 0)
        eng = a._eng
        eng.set_system(Z, m, skin=a.skin)
        return (lambda x: eng.evaluate(x, cell, compute_stress=True)), base
    if not hasattr(a, "_ev"):
        sys.path.insert(0, a.pkg)
        from symmetrix._mace_foundation import MACEFoundation

        a._ev = MACEFoundation(a.json, "cuda", a.dtype, a.skin)
    ev = a._ev
    Z32 = np.ascontiguousarray(Z, dtype=np.int32)
    pbc = np.array([True, True, True])
    return (lambda x: ev.compute(x, Z32, cell, pbc, False)), base


def time_mode(a):
    snap = load_snapshot(a.scratch)
    rng = np.random.default_rng(7)
    rows = []
    for n in [int(s) for s in a.sizes.split(",")]:
        fn, base = make_fn(a, snap, n)
        fn(base)
        for _ in range(WARMUP):
            fn(base + rng.normal(0.0, SIGMA, base.shape))
        ts = []
        for _ in range(CALLS):
            x = base + rng.normal(0.0, SIGMA, base.shape)
            t0 = time.perf_counter()
            fn(x)
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts) * 1e3
        rows.append(dict(label=a.label, impl=a.impl, dtype=a.dtype if a.impl == "sx" else None, skin=a.skin,
                         n_atoms=len(base), call_ms=float(np.median(ts)), call_p10=float(np.percentile(ts, 10)),
                         call_p90=float(np.percentile(ts, 90)), pass_tag=a.pass_tag))
    with open(a.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)


def prof_mode(a):
    snap = load_snapshot(a.scratch)
    rng = np.random.default_rng(7)
    fn, base = make_fn(a, snap, a.size)
    fn(base)
    for _ in range(WARMUP):
        fn(base + rng.normal(0.0, SIGMA, base.shape))
    xs = [base + rng.normal(0.0, SIGMA, base.shape) for _ in range(a.calls)]
    cudart = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
    if cudart.cudaProfilerStart() != 0:
        raise SystemExit("cudaProfilerStart failed")
    for x in xs:
        time.sleep(0.003)
        fn(x)
    cudart.cudaProfilerStop()
    print(f"profiled {a.calls} force calls ({a.impl}, skin {a.skin}, {len(base)} atoms)", flush=True)


# Phase of each kernel, by implementation (regexes on the demangled name, first match wins).
PHASES = {
    "sx": [
        ("neighbour edges (count, scan, fill)", r"k_count|k_fill|DeviceScan|cub::"),
        ("layer 0 conv forward (Phi0)", r"k_Phi0<"),
        ("layer 0 conv backward (Phi0_reverse)", r"k_Phi0_reverse"),
        ("layer 1 density (scale1)", r"k_scale1"),
        ("layer 1 conv forward (Phi1)", r"k_Phi1<"),
        ("layer 1 conv backward (Phi1_reverse)", r"k_Phi1_reverse"),
        ("products forward (k_M)", r"k_M<"),
        ("products backward (k_M_reverse)", r"k_M_reverse"),
        ("gates, biases, amax", r"k_gate|k_amax|k_h1_bias"),
        ("readouts, ZBL, outputs", r"k_readouts|k_zbl|k_outputs"),
        ("edge -> atom forces", r"k_forces"),
        ("GEMMs", r"k_gemm|k_sgemm|gemm|cutlass|xmma|Kernel"),
    ],
    "loup": [
        ("neighbour list / skin / edges", r"skin|neighbor|nl_|edge|frac_kernel|atom_order|count_and|copy_ints|"
                                          r"gather_rows|sort|scan|Scan|radix|Radix"),
        ("layer 0 conv forward", r"first_fwd|gconv_fwd_L0|fk_fwd|interaction0"),
        ("layer 0 conv backward", r"first_bwd|gconv_bwd_L0|fk_bwd"),
        ("layer 1 conv forward", r"spline_forward|gconv_fwd_L1|feat_pass"),
        ("layer 1 conv backward", r"wprime|gconv_bwd_L1"),
        ("products", r"prod|symm|contract|sc0|sc_|tree"),
        ("GEMMs", r"sgemm|gemm|cutlass|xmma|magma|Kernel2|tf32|cublas|ampere|sm80|sm90|sm120"),
        ("elementwise (gates, bias, bcast, energy)", r"gate|bias|bcast|silu|energy|scale|add8|axpy|d2f|dmm|fold"),
        ("forces", r"assemble_forces|force"),
    ],
}


def phase_of(impl, name):
    for phase, rx in PHASES["sx" if impl == "sx" else "loup"]:
        if re.search(rx, name):
            return phase
    return "other"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def short(name, width=110):
    name = re.sub(r"\(anonymous namespace\)::", "", name)
    return name if len(name) <= width else name[:width - 3] + "..."


def report(a):
    rows = [json.loads(line) for line in open(a.out) if line.strip()]
    print("## Force call at equal skin (ms, median of 100; pass 1 / pass 2)\n")
    labels = list(dict.fromkeys(r["label"] for r in rows))
    sizes = sorted({r["n_atoms"] for r in rows})
    skins = sorted({r["skin"] for r in rows})
    for skin in skins:
        print(f"skin {skin} A\n")
        print("| implementation | " + " | ".join(f"{n} atoms" for n in sizes) + " |")
        print("|---|" + "---|" * len(sizes))
        ref = {}
        for lab in labels:
            cells = []
            for n in sizes:
                rs = [r for r in rows if r["label"] == lab and r["n_atoms"] == n and r["skin"] == skin]
                if not rs:
                    cells.append("")
                    continue
                mean = statistics.mean(r["call_ms"] for r in rs)
                if lab == a.ref:
                    ref[n] = mean
                ratio = f" ({mean / ref[n]:.2f}x)" if n in ref and lab != a.ref else ""
                cells.append(" / ".join(f"{r['call_ms']:.2f}" for r in rs) + ratio)
            print(f"| {lab} | " + " | ".join(cells) + " |")
        print()

    print("## Kernel profile of one force call (nsys, SrTiO3 5,000 atoms)\n")
    summary = []
    for lab in a.labels.split(","):
        prefix = os.path.join(a.scratch, f"prof_{lab}")
        impl = "sx" if lab.startswith("sx") else "loup"
        trace = read_csv(prefix + "_cuda_gpu_trace.csv")
        ops = sorted((int(r["Start (ns)"]), int(r["Start (ns)"]) + int(r["Duration (ns)"]), r["Name"]) for r in trace)
        calls, cur, end = [], [], None
        for op in ops:
            if cur and op[0] - end > GAP_NS:
                calls.append(cur)
                cur = []
            cur.append(op)
            end = op[1] if end is None or not cur[:-1] else max(end, op[1])
        if cur:
            calls.append(cur)
        spans = [max(o[1] for o in c) - min(o[0] for o in c) for c in calls]
        nc = len(calls)
        per = defaultdict(float)
        cnt = defaultdict(int)
        mem = 0.0
        for c in calls:
            for s, e, name in c:
                if name.startswith("[CUDA"):
                    mem += e - s
                else:
                    per[name] += e - s
                    cnt[name] += 1
        ksum = sum(per.values()) / nc / 1e3
        summary.append((lab, nc, statistics.median(spans) / 1e3, ksum, mem / nc / 1e3))
        phases = defaultdict(float)
        for name, t in per.items():
            phases[phase_of(impl, name)] += t / nc / 1e3
        print(f"### {lab}: {nc} calls, device span {statistics.median(spans) / 1e3:.0f} us, kernels {ksum:.0f} us, "
              f"memcpy / memset {mem / nc / 1e3:.0f} us per call\n")
        print("| phase | us per call |")
        print("|---|---|")
        for phase, t in sorted(phases.items(), key=lambda kv: -kv[1]):
            print(f"| {phase} | {t:.0f} |")
        print()
        print("| kernel | launches per call | us per call |")
        print("|---|---|---|")
        for name, t in sorted(per.items(), key=lambda kv: -kv[1]):
            if t / nc / 1e3 >= 5:
                print(f"| `{short(name)}` | {cnt[name] / nc:.0f} | {t / nc / 1e3:.0f} |")
        print()
    print("| implementation | calls | device span per call (us) | kernel time per call (us) | memcpy / memset (us) |")
    print("|---|---|---|---|---|")
    for lab, nc, span, ksum, mem in summary:
        print(f"| {lab} | {nc} | {span:.0f} | {ksum:.0f} | {mem:.0f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("time", "prof", "report"))
    ap.add_argument("--scratch", default="/home/fzwin/src/symmetrix-eval/scratch")
    ap.add_argument("--impl", choices=("loup", "loup-generic", "sx"))
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--json")
    ap.add_argument("--pkg", default="/home/fzwin/src/symmetrix-eval/b907f04")
    ap.add_argument("--skin", type=float, default=0.5)
    ap.add_argument("--sizes", default="6,10,14")
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--label")
    ap.add_argument("--labels")
    ap.add_argument("--ref", default="loup")
    ap.add_argument("--out")
    ap.add_argument("--pass-tag", default="1")
    a = ap.parse_args()
    {"time": time_mode, "prof": prof_mode, "report": report}[a.mode](a)


if __name__ == "__main__":
    main()
