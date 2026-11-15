"""Export a MACE model into a .loup bundle: the specialised OMAT-0-medium path when the model has exactly that
architecture (and --generic is not given), else the generic path.

  python -m loupmd.export --model M --elements 8,22,38 --output out.loup [--head H] [--generic]
                          [--charge Q --spin-multiplicity M]   (models with a total charge / spin embedding:
                                                                MACE-OMOL-0)

The exporters need torch, mace-torch and cuequivariance (the package's ``export`` extra) and, until they are
vendored (PLAN.md S2-T03), a FastMACE checkout. Nothing here is imported by ``import loupmd``: the bundle cache
(``loupmd.bundles``) runs this module in a process of its own.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--elements", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--head", default=None)
    ap.add_argument("--generic", action="store_true")
    ap.add_argument("--charge", type=int, default=None)
    ap.add_argument("--spin-multiplicity", type=int, default=None)
    a = ap.parse_args()
    import torch

    elems = [int(x) for x in a.elements.split(",") if x.strip()]
    specialised = False
    if not a.generic:
        from . import omat0 as export_omat0

        m = torch.load(a.model, map_location="cpu", weights_only=False)
        try:
            export_omat0.check_architecture(m)
            specialised = a.head is None or a.head in (list(getattr(m, "heads", None) or ["default"]))
        except ValueError:
            specialised = False
        del m
    if specialised:
        from . import omat0 as export_omat0

        if a.charge is not None or a.spin_multiplicity is not None:
            raise SystemExit("the model has no total-charge / spin embedding: --charge / --spin-multiplicity do not apply")
        export_omat0.export_model(a.model, elems, Path(a.output))
    else:
        from . import generic as export_generic

        from ..errors import UnsupportedModel
        from ..kernels import UNSUPPORTED_MARKER, UNSUPPORTED_STATUS

        try:
            export_generic.export_model(a.model, elems, Path(a.output), a.head, a.charge, a.spin_multiplicity)
        except UnsupportedModel as error:      # for the caller (bundles.ensure_bundle): the marker, then the status
            print(str(error), file=sys.stderr)
            print(f"{UNSUPPORTED_MARKER} {' '.join(error.signatures)}", file=sys.stderr)
            raise SystemExit(UNSUPPORTED_STATUS) from None

