"""Summaries of job_profiles.sh's outputs as Markdown (stdout): python summarize.py RESULTS_DIR
  - per family: MD step time, kernel time per step, the time per kernel category, the top kernels;
  - per ncu family: each profiled kernel's duration, throughputs (% of peak), warps active, registers, hit rates and
    its three largest warp-stall reasons (warps per issued instruction, from the raw page)."""
import csv
import gzip
import glob
import os
import re
import sys
from collections import defaultdict

R = sys.argv[1]
CATS = [  # (category, regex on the demangled name); first match wins
    ("conv forward (interaction >= 1)", r"gconv_fwd|spline_forward_inter1"),
    ("conv backward (interaction >= 1)", r"gconv_bwd|wprime_backward"),
    ("layer 0 / interaction 0", r"first_fwd|first_bwd|tabled_outer_sum0|interaction0|i0_"),
    ("symmetric contraction / product", r"gprod|sc0_|sc1_|sc_fused|symmetric"),
    ("nonlinear gates", r"gate_fwd|gate_bwd"),
    ("GEMM", r"gemm|Gemm|cutlass|magma|fm_tf32|loup_gemm_tc|sm80_|sm90_|ampere|cublas|splitk"),
    ("POLAR k-space", r"sf_kernel|feat_pass|grad_pass|feat_coef|bwd_coef|^energy_kernel|lr_virial|k_reduce|frac_kernel|qtot"),
    ("POLAR dense / elementwise", r"inv_kernel|inv_bwd|bcast_mul|ro_fwd|ro_bwd|fukui|fk_fwd|fk_bwd|pot_fwd|pot_bwd|spin_grad|add8|axpy|fill_kernel|add_bias|ln_silu|sum_rows"),
    ("neighbour list / edges", r"neighbor|neighbour|count_|fill_|sort_|scatter_active|compute_and_count|nl_|rebuild|bin_|cell_"),
    ("forces / integrator / guards", r"assemble_forces|langevin|integr|guard|half_|kick|drift|zero|memset"),
]
RAW = {  # column -> header in the table
    "gpu__time_duration.sum": "us",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM %",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "L2 %",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "DRAM %",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "warps %",
    "launch__registers_per_thread": "regs",
    "lts__t_sector_hit_rate.pct": "L2 hit %",
    "l1tex__t_sector_hit_rate.pct": "L1 hit %",
}
STALL = re.compile(r"^smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$")


def demangle(name):
    return re.sub(r"\(.*$", "", name).replace("void ", "").replace("loup::", "").replace("<unnamed>::", "").strip()


def cat_of(name):
    for c, rx in CATS:
        if re.search(rx, name):
            return c
    return "other"


def num(s):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return float("nan")


print("# Kernel time per MD step (nsys)\n")
steps = {}
if os.path.exists(f"{R}/nsys/steps.txt"):
    print("| Model | System | Atoms | Steps | Rebuilds | ms / step |\n|---|---|---:|---:|---:|---:|")
    for line in open(f"{R}/nsys/steps.txt"):
        m = re.search(r"PROF (\S+) (\S+) (\d+) atoms: (\d+) steps, (\d+) rebuilds, ([\d.]+) ms/step", line)
        if m:
            print("| " + " | ".join(m.groups()) + " |")
    print()
# cross-family view: every case's category shares in one table
SHORT = ["conv backward (interaction >= 1)", "conv forward (interaction >= 1)", "GEMM", "layer 0 / interaction 0",
         "symmetric contraction / product", "nonlinear gates", "POLAR k-space", "POLAR dense / elementwise"]
cross = []
for path in sorted(glob.glob(f"{R}/nsys/*_cuda_gpu_kern_sum.csv")):
    label = os.path.basename(path).replace("_cuda_gpu_kern_sum.csv", "")
    n_steps = int(open(f"{R}/nsys/{label}.steps").read().split("=")[1])
    by_cat, launches = defaultdict(float), 0.0
    for r in csv.DictReader(open(path)):
        by_cat[cat_of(demangle(r["Name"]))] += float(r["Total Time (ns)"]) / 1e3 / n_steps
        launches += int(r["Instances"]) / n_steps
    cross.append((label, sum(by_cat.values()), launches, by_cat))
print("## All cases: share of kernel time per category\n")
print("| Case | kernel us / step | launches / step | conv bwd | conv fwd | GEMM | layer 0 | product | gates | "
      "POLAR k | POLAR dense | rest |")
print("|---|---:|---:|" + "---:|" * (len(SHORT) + 1))
for label, tot, launches, by_cat in cross:
    rest = tot - sum(by_cat[c] for c in SHORT)
    print(f"| {label} | {tot:,.0f} | {launches:.0f} | " + " | ".join(
        f"{100 * by_cat[c] / tot:.1f}%" if by_cat[c] >= 0.5 else "-" for c in SHORT) + f" | {100 * rest / tot:.1f}% |")
print()
for path in sorted(glob.glob(f"{R}/nsys/*_cuda_gpu_kern_sum.csv")):
    label = os.path.basename(path).replace("_cuda_gpu_kern_sum.csv", "")
    n_steps = int(open(f"{R}/nsys/{label}.steps").read().split("=")[1])
    rows, calls = defaultdict(float), defaultdict(float)
    for r in csv.DictReader(open(path)):
        n = demangle(r["Name"])
        rows[n] += float(r["Total Time (ns)"]) / 1e3 / n_steps
        calls[n] += int(r["Instances"]) / n_steps
    tot = sum(rows.values())
    by_cat = defaultdict(float)
    for n, v in rows.items():
        by_cat[cat_of(n)] += v
    print(f"## {label}\n")
    print(f"Kernel time {tot:,.0f} us per step (summed over streams), {sum(calls.values()):.0f} launches per step.\n")
    print("| Category | us / step | share |\n|---|---:|---:|")
    for c, v in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        if v >= 0.5:
            print(f"| {c} | {v:,.0f} | {100 * v / tot:.1f}% |")
    print("\n| Kernel | us / step | share | launches / step |\n|---|---:|---:|---:|")
    for n, v in sorted(rows.items(), key=lambda kv: -kv[1])[:14]:
        print(f"| `{n[:80]}` | {v:,.1f} | {100 * v / tot:.1f}% | {calls[n]:.1f} |")
    print()

print("# ncu, one force call (top kernels, sections + raw metrics)\n")
for path in sorted(glob.glob(f"{R}/ncu/*_raw.csv") + glob.glob(f"{R}/ncu/*_raw.csv.gz")):
    label = re.sub(r"_raw\.csv(\.gz)?$", "", os.path.basename(path))
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
        lines = [l for l in fh if l.startswith('"')]
    if len(lines) < 3:
        continue
    rd = list(csv.reader(lines))
    head, units, data = rd[0], rd[1], rd[2:]
    col = {h: i for i, h in enumerate(head)}
    stall_cols = {m.group(1): i for h, i in col.items() if (m := STALL.match(h))}
    t_unit = units[col["gpu__time_duration.sum"]]
    t_scale = {"nsecond": 1e-3, "ns": 1e-3, "usecond": 1.0, "us": 1.0, "msecond": 1e3, "ms": 1e3}.get(t_unit, 1e-3)
    agg = defaultdict(lambda: defaultdict(float))
    cnt = defaultdict(int)
    for row in data:
        n = demangle(row[col["Kernel Name"]])
        cnt[n] += 1
        t = num(row[col["gpu__time_duration.sum"]])
        agg[n]["_t"] += t
        for k in RAW:
            if k in col:
                agg[n][k] += num(row[col[k]]) * (1.0 if k == "gpu__time_duration.sum" else t)  # time-weighted
        for s, i in stall_cols.items():
            agg[n]["stall_" + s] += num(row[i]) * t
    print(f"## {label}\n")
    print("| Kernel | launches | " + " | ".join(RAW.values()) + " | top stalls (warps per issue) |")
    print("|---|---:|" + "---:|" * len(RAW) + "---|")
    for n, a in sorted(agg.items(), key=lambda kv: -kv[1]["_t"])[:16]:
        t = a["_t"] or 1.0
        vals = []
        for k in RAW:
            v = a[k] if k == "gpu__time_duration.sum" else a[k] / t
            vals.append(f"{v * t_scale:,.1f}" if k == "gpu__time_duration.sum" else f"{v:.0f}")
        st = sorted(((s[6:], v / t) for s, v in a.items() if s.startswith("stall_")), key=lambda kv: -kv[1])[:3]
        print(f"| `{n[:64]}` | {cnt[n]} | " + " | ".join(vals) + " | " + ", ".join(f"{k} {v:.2f}" for k, v in st) + " |")
    print()
