#!/usr/bin/env python3
"""symmetrix_extract_mace's steps (symmetrix/source/symmetrix/cli/extract_mace.py at b907f04) run from the manual
build of sx_build.sh: extract_mace_data with the CLI's arguments, written as the CLI writes it (json, indent 4).
The CLI itself is not used because its importlib.metadata.version("symmetrix") needs a pip-installed symmetrix.

  python sx_extract.py --pkg DIR --model MODEL --species 8,22,38 --points 1024 --output FILE.json
"""
import argparse
import json
import sys
import time
from importlib.metadata import version

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True, help="directory holding the symmetrix package of sx_build.sh")
ap.add_argument("--model", required=True)
ap.add_argument("--species", required=True, help="comma-separated atomic numbers")
ap.add_argument("--points", type=int, required=True, help="--num-spline-points")
ap.add_argument("--output", required=True)
a = ap.parse_args()

sys.path.insert(0, a.pkg)
from symmetrix.extract_mace_data import extract_mace_data  # noqa: E402

species = [int(z) for z in a.species.split(",")]
t0 = time.time()
out = extract_mace_data(a.model, species, None, num_spline_points=a.points, r_min=0.5, compression_margin=None)
out["extracted_with"] = {"symmetrix": "b907f04 (manual build, sx_build.sh)", "mace-torch": version("mace-torch"),
                         "torch": version("torch")}
t1 = time.time()
with open(a.output, "w") as f:
    json.dump(out, f, indent=4)
print(f"{a.output}: species {species}, {a.points} spline points, extract {t1 - t0:.1f} s, "
      f"write {time.time() - t1:.1f} s", flush=True)
