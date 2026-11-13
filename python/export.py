#!/usr/bin/env python3
"""Export a MACE model into a .loup bundle: the specialised OMAT-0-medium path when the model has exactly that
architecture (and --generic is not given), else the generic path.

  python export.py --model M --elements 8,22,38 --output out.loup [--head H] [--generic]
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
    ap.add_argument("--generic", action="store_true")
    a = ap.parse_args()
    import torch

    elems = [int(x) for x in a.elements.split(",") if x.strip()]
    specialised = False
    if not a.generic:
        import export_omat0

        m = torch.load(a.model, map_location="cpu", weights_only=False)
        try:
            export_omat0.check_architecture(m)
            specialised = a.head is None or a.head in (list(getattr(m, "heads", None) or ["default"]))
        except ValueError:
            specialised = False
        del m
    if specialised:
        import export_omat0

        export_omat0.export_model(a.model, elems, Path(a.output))
    else:
        import export_generic

        export_generic.export_model(a.model, elems, Path(a.output), a.head)


if __name__ == "__main__":
    main()
