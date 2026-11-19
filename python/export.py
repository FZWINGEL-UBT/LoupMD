#!/usr/bin/env python3
"""Export a MACE model into a .loup bundle (the generic path; the specialised OMAT-0-medium path was retired in
JIT_DESIGN.md stage 5, so every model, OMAT-0 medium included, exports through export_generic.py).

  python export.py --model M --elements 8,22,38 --output out.loup [--head H]
                   [--charge Q --spin-multiplicity M]   (models with a total charge / spin embedding: MACE-OMOL-0)

--generic is accepted and ignored (it selected the generic path before every model took it).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--elements", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--head", default=None)
    ap.add_argument("--generic", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--charge", type=int, default=None)
    ap.add_argument("--spin-multiplicity", type=int, default=None)
    a = ap.parse_args()
    import export_generic

    elems = [int(x) for x in a.elements.split(",") if x.strip()]
    export_generic.export_model(a.model, elems, Path(a.output), a.head, a.charge, a.spin_multiplicity)


if __name__ == "__main__":
    main()
