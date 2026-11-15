"""``loupmd validate``: the accuracy gate on a model and a structure set or your own structures."""
from __future__ import annotations

import sys

from .. import validate as _validate
from ..config import GateConfig
from ..validate import sets as _sets


def add_parser(sub):
    parser = sub.add_parser(
        "validate", help="check LoupMD against MACE float64 (the 3x rule)",
        description="Compares LoupMD's energy, forces and stress with MACE float64 on strained and rattled "
                    "structures. A quantity passes when its deviation is at most three times MACE float32's own "
                    "deviation, or below 0.01 meV/atom, 1e-5 eV/A, 1e-7 eV/A^3. Needs the export extra (torch, "
                    "mace-torch). Exit code 1 when the gate fails.")
    parser.add_argument("--model", required=True, metavar="MODEL", help="a label (see `loupmd models`), a checkpoint "
                                                                       "path or a URL")
    parser.add_argument("--head", help="the head of a multi-head model")
    parser.add_argument("--checkpoint", metavar="PATH", help="the file of a label, when it is not in LOUPMD_MODELS")
    parser.add_argument("--sha256", metavar="HEX", help="the stated hash of a file that is not in the label table")
    parser.add_argument("--trusted", action="store_true", help="load a file outside the label table without a hash")
    parser.add_argument("--download", action="store_true", help="download a label's checkpoint if it is missing")
    parser.add_argument("--set", choices=_sets.SET_NAMES, default="oxide",
                        help="oxide: NaCl, SrTiO3, NbCl5 (default); organic: urea, naphthalene, paracetamol, water; "
                             "crystals: NaCl, NaNbO3, NbCl5 R2SCAN cells")
    parser.add_argument("--structure", action="append", metavar="FILE",
                        help="gate rattled copies of your own structure instead of a set (repeatable)")
    parser.add_argument("--copies", type=int, default=2, metavar="N", help="rattled copies per structure (default 2)")
    parser.add_argument("--rattle", type=float, default=0.05, metavar="A", help="their displacement width in A "
                                                                               "(default 0.05)")
    parser.add_argument("--ref-device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="where MACE computes the references; cpu for models whose float64 pass does not fit the "
                             "GPU (default: the GPU if there is one)")
    parser.add_argument("--no-ref-cache", action="store_true", help="compute the references again and overwrite the "
                                                                    "cached ones")
    parser.add_argument("--include-unstable", action="store_true",
                        help="also gate configurations on which the model itself is not float32-reproducible")
    parser.add_argument("--generic", action="store_true", help="the generic kernels for a model that has specialised "
                                                               "ones")
    parser.add_argument("--charge", type=int, help="total charge (models with a charge embedding)")
    parser.add_argument("--spin-multiplicity", type=int, help="2S + 1 (models with a spin embedding)")
    parser.add_argument("--device", type=int, default=0, metavar="N", help="the GPU index (default 0)")
    parser.add_argument("--json", metavar="FILE", help="write the record (loupmd.gate/1) to FILE, or - for stdout")
    parser.set_defaults(run=run)


def run(args) -> int:
    out = sys.stderr if args.json == "-" else sys.stdout
    record = _validate.run_gate(
        args.model, head=args.head, checkpoint=args.checkpoint, sha256=args.sha256, trusted=args.trusted,
        download=args.download, set_name=args.set, structures=args.structure,
        gate_config=GateConfig(rattle_A=args.rattle, copies=args.copies, reference_device=args.ref_device),
        generic=True if args.generic else None, charge=args.charge, spin_multiplicity=args.spin_multiplicity,
        device=args.device, use_ref_cache=not args.no_ref_cache, include_unstable=args.include_unstable)
    print(_validate.format_gate(record), file=out)
    print("gate PASSED" if record["passed"] else "gate FAILED", file=out)
    if args.json:
        _validate.write_record(record, args.json)
    return 0 if record["passed"] else 1
