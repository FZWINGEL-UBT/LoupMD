"""Compare two profile directories (job_profiles.sh outputs): python compare.py OLD_DIR NEW_DIR
Markdown on stdout: per case the MD step time and summed kernel time of both, and the kernel-time change per category."""
import csv
import os
import re
import sys
from collections import defaultdict

OLD, NEW = sys.argv[1], sys.argv[2]
# (model file, system) -> labels in the order job_profiles.sh runs them (OMAT-0 medium: specialised, then generic)
LABELS = {
    ("mace-omat-0-medium.model", "srtio3"): ["omat0m_spec_srtio3", "omat0m_gen_srtio3"],
    ("2023-12-03-mace-128-L1_epoch-199.model", "srtio3"): ["mp0m_srtio3"],
    ("2023-12-03-mace-128-L1_epoch-199.model", "nbcl5"): ["mp0m_nbcl5"],
    ("MACE_MPtrj_2022.9.model", "srtio3"): ["mptrj_srtio3"],
    ("MACE_MPtrj_2022.9.model", "nbcl5"): ["mptrj_nbcl5"],
    ("mace-omat-0-small.model", "srtio3"): ["omat0s_srtio3"],
    ("mace-mh-1.model", "srtio3"): ["mh1_srtio3"],
    ("mace-mh-1.model", "nacl216"): ["mh1_nacl216"],
    ("MACE-OFF23_small.model", "water216"): ["off23s_water216"],
    ("MACE-OFF23_medium.model", "water216"): ["off23m_water216"],
    ("MACE-OFF24_medium.model", "water"): ["off24m_water"],
    ("MACE-OFF23_large.model", "water216"): ["off23l_water216"],
    ("MACE-omol-0-extra-large-1024.model", "water216"): ["omol0xl_water216"],
    ("MACE-POLAR-1-S.model", "nacl216"): ["polar1s_nacl216"],
    ("MACE-POLAR-1-S.model", "nbcl5"): ["polar1s_nbcl5"],
    ("MACE-POLAR-1-M.model", "nacl216"): ["polar1m_nacl216"],
    ("MACE-POLAR-1-M.model", "nbcl5"): ["polar1m_nbcl5"],
    ("MACE-POLAR-1-L.model", "nacl216"): ["polar1l_nacl216"],
}
CATS = [
    ("conv bwd", r"gconv_bwd|wprime_backward"),
    ("conv fwd", r"gconv_fwd|spline_forward_inter1"),
    ("layer 0", r"first_fwd|first_bwd|tabled_outer_sum0|interaction0|i0_"),
    ("product", r"gprod|sc0_|sc1_|sc_fused|symmetric"),
    ("gates", r"gate_fwd|gate_bwd"),
    ("GEMM", r"gemm|Gemm|cutlass|magma|fm_tf32|loup_gemm_tc|sm80_|sm90_|ampere|cublas|splitk"),
]


def steps(d):
    out, seen = {}, defaultdict(int)
    path = f"{d}/nsys/steps.txt"
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = re.search(r"PROF (\S+) (\S+) (\d+) atoms: (\d+) steps, (\d+) rebuilds, ([\d.]+) ms/step", line)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        labels = LABELS.get(key, [])
        i = seen[key]
        seen[key] += 1
        if i < len(labels):
            out[labels[i]] = (int(m.group(3)), float(m.group(6)))
    return out


def kernels(d, label):
    path = f"{d}/nsys/{label}_cuda_gpu_kern_sum.csv"
    if not os.path.exists(path):
        return None
    n = int(open(f"{d}/nsys/{label}.steps").read().split("=")[1])
    by = defaultdict(float)
    for r in csv.DictReader(open(path, newline="")):
        name = re.sub(r"\(.*$", "", r["Name"])
        cat = next((c for c, rx in CATS if re.search(rx, name)), "rest")
        by[cat] += float(r["Total Time (ns)"]) / 1e3 / n
    return by


so, sn = steps(OLD), steps(NEW)
print("| Case | Atoms | ms / step before | ms / step now | now / before | kernel us / step before | now |")
print("|---|---:|---:|---:|---:|---:|---:|")
order = [l for ls in LABELS.values() for l in ls]
for label in order:
    if label not in sn:
        continue
    atoms, t_new = sn[label]
    ko, kn = kernels(OLD, label), kernels(NEW, label)
    t_old = so.get(label, (None, None))[1]
    ratio = f"{t_new / t_old:.3f}" if t_old else "-"
    k_old = f"{sum(ko.values()):,.0f}" if ko else "-"
    k_new = f"{sum(kn.values()):,.0f}" if kn else "-"
    print(f"| {label} | {atoms:,} | {t_old if t_old else '-'} | {t_new} | {ratio} | {k_old} | {k_new} |")
print()
print("Kernel time change per category (us per step, now minus before; summed over streams):\n")
print("| Case | " + " | ".join(c for c, _ in CATS) + " | rest |")
print("|---|" + "---:|" * (len(CATS) + 1))
for label in order:
    ko, kn = kernels(OLD, label), kernels(NEW, label)
    if not ko or not kn:
        continue
    cells = []
    for c in [c for c, _ in CATS] + ["rest"]:
        dlt = kn.get(c, 0.0) - ko.get(c, 0.0)
        cells.append(f"{dlt:+,.0f}" if abs(dlt) >= 0.5 else "0")
    print(f"| {label} | " + " | ".join(cells) + " |")
