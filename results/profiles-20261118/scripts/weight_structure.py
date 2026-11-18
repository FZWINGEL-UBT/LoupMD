"""Structure of every cached generic bundle's projection weights (CPU): for each layer and each W_lin_l / W_p_l /
W_res_l / W_2_l matrix, the live output columns and input rows (exactly nonzero, and above 1e-6 of the layer's largest
|W_lin_0| entry), so exact or near-exact column / row sparsity shows up per family.
python weight_structure.py > weight_structure.md"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/mnt/d/Margraf/LoupMD/python")
from bundle import read_bundle  # noqa: E402

seen = set()
paths = sorted(glob.glob(os.path.expanduser("~/.cache/loupmd/**/*.loup"), recursive=True))
print("# Projection weight structure per family (exported float32 bundles)\n")
print("Columns = output channels, rows = inputs (path x channel for W_lin). `nz` exactly nonzero; `big` above 1e-6 of "
      "the layer's max |W_lin_0| (the scale of the live weights). A matrix with few live columns makes its GEMM thin "
      "and, for W_lin_l with l >= 1, leaves most channels' l >= 1 product inputs zero.\n")
for path in paths:
    fam = re.sub(r"(_\d+)+_[0-9a-f]{16}\.loup$", "", os.path.basename(path))
    if fam in seen:
        continue
    try:
        meta, t = read_bundle(path)
    except Exception:  # noqa: BLE001
        continue
    if "L0.kind" not in meta:
        continue
    seen.add(fam)
    print(f"## {fam} (`{os.path.basename(path)}`, head {meta.get('head', '-')})\n")
    print("| matrix | shape | nz cols | big cols | nz rows | big rows | max abs |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for t_ in range(int(meta.get("n_layers", 4))):
        p = f"L{t_}."
        if p + "kind" not in meta:
            continue
        ref = np.abs(t[p + "W_lin_0"]).max() if p + "W_lin_0" in t else 1.0
        for name in sorted(k for k in t if k.startswith(p) and re.match(r"W_(lin|p|res|2)_\d$", k[len(p):])):
            w = np.asarray(t[name], dtype=np.float64)
            if w.ndim == 3:  # per-species [S, in, out]
                w = np.abs(w).max(axis=0)
            nz, big = w != 0, np.abs(w) > 1e-6 * ref
            print(f"| `{name}` | {tuple(int(s) for s in w.shape)} | {int(nz.any(0).sum())} | {int(big.any(0).sum())} | "
                  f"{int(nz.any(1).sum())} | {int(big.any(1).sum())} | {np.abs(w).max():.2e} |")
    print()
