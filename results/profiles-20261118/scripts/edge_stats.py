"""Neighbour statistics of the benchmark systems (CPU, ASE): directed edges within the cutoff, edges per atom, the
share of edges per distance band, species-pair shares and how many receivers' edges fall in one radial interval.
python edge_stats.py > edge_stats.md  (structures from python/bench_family.py, rattled by 0.05 A like the profiles'
force calls; the MD profiles start from the same cells at 300 K)."""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/mnt/d/Margraf/LoupMD/python")
import bench_family as bf  # noqa: E402
from ase.neighborlist import neighbor_list  # noqa: E402

BANDS = [(0.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 5.5), (5.5, 6.0)]
print("# Neighbour statistics of the benchmark systems\n")
print("Cutoffs: 6.0 A (OMAT-0, MP-0, MPtrj, MH-1, POLAR-1) and 5.0 A (OFF23 / OFF24 / OMOL-0). Skin 0.5 A is not "
      "included: these are the edges inside the cutoff (the active edges), not the list's capacity.\n")
for name in ("srtio3", "nbcl5", "nacl216", "water216", "water"):
    at = bf.system(name)
    at.positions += np.random.default_rng(7).normal(0.0, 0.05, at.positions.shape)
    Z = at.get_atomic_numbers()
    print(f"## {name}: {len(at)} atoms, species {sorted(Counter(at.get_chemical_symbols()).items())}\n")
    print("| cutoff | directed edges | edges / atom (mean, min, max) | " +
          " | ".join(f"{a:.1f}-{b:.1f} A" for a, b in BANDS) + " |")
    print("|---|---:|---|" + "---:|" * len(BANDS))
    for rc in (5.0, 6.0):
        i, j, d = neighbor_list("ijd", at, rc)
        per = np.bincount(i, minlength=len(at))
        shares = [100.0 * np.count_nonzero((d >= a) & (d < b)) / len(d) for a, b in BANDS]
        print(f"| {rc:.1f} | {len(d):,} | {per.mean():.1f} ({per.min()}, {per.max()}) | " +
              " | ".join(f"{s:.1f}%" for s in shares) + " |")
    i, j, d = neighbor_list("ijd", at, 6.0)
    pairs = Counter((at[a].symbol, at[b].symbol) for a, b in zip(i, j))
    tot = sum(pairs.values())
    print("\nDirected species pairs at 6.0 A (receiver, sender): " +
          ", ".join(f"{a}-{b} {100 * c / tot:.1f}%" for (a, b), c in sorted(pairs.items(), key=lambda kv: -kv[1])))
    same = sum(c for (a, b), c in pairs.items() if a == b)
    print(f"\nSame-species edges: {100 * same / tot:.1f}%. Edges at d >= 2.75 A: "
          f"{100 * np.count_nonzero(d >= 2.75) / len(d):.1f}%; d >= 5.0 A: {100 * np.count_nonzero(d >= 5.0) / len(d):.1f}%.\n")
