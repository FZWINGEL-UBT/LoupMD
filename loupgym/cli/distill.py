"""``loupmd distill``: automatic teacher-to-FastLR-SK student distillation into a ``.loup`` bundle."""
from __future__ import annotations

_SPEED_CHOICES = ("fast50", "balanced80", "accurate100", "max")
_BUDGET_CHOICES = ("quick", "standard", "thorough")


def add_parser(sub):
    parser = sub.add_parser(
        "distill",
        help="distill a fast pure-CUDA student .loup bundle automatically",
        description=(
            "Automatically analyzes the system, benchmarks candidate FastLR-SK architectures on the local GPU, "
            "screens and trains the best model with pure CUDA, exports a torch-free student.loup bundle, "
            "and verifies accuracy and MD parity."
        ),
    )
    parser.add_argument("--teacher", metavar="MODEL", help="teacher label (e.g. omat0-small) or .model path")
    parser.add_argument("--head", help="head name for multi-head teachers")
    parser.add_argument("--checkpoint", metavar="PATH", help="explicit checkpoint file for a teacher label")
    parser.add_argument("--sha256", metavar="HEX", help="expected SHA-256 for an external teacher file")
    parser.add_argument("--trusted", action="store_true", help="trust an external teacher checkpoint without hash")

    parser.add_argument("--structure", action="append", metavar="FILE",
                        help="starting structure(s) (.xyz/.cif/.traj, repeatable)")
    parser.add_argument("--md-data", action="append", metavar="FILE",
                        help="MD frames or trajectory (.xyz/.extxyz/.traj, repeatable)")
    parser.add_argument("--dft-data", action="append", metavar="FILE",
                        help="DFT reference frames with energy/forces (.extxyz/.xyz, repeatable)")

    parser.add_argument("-o", "--output", default="./distill_out", metavar="PATH",
                        help="output directory or .loup file path (default ./distill_out)")
    parser.add_argument("--speed", choices=_SPEED_CHOICES, default="fast50",
                        help="target MD speed preset (default fast50 <= 0.050 us/atom)")
    parser.add_argument("--max-speed", type=float, default=None, metavar="US_PER_ATOM",
                        help="explicit target us/atom budget (overrides --speed)")
    parser.add_argument("--budget", choices=_BUDGET_CHOICES, default="standard",
                        help="sampling/training budget preset (quick, standard, thorough)")
    parser.add_argument("--epochs", type=int, default=None, help="override final training epoch count")
    parser.add_argument("--n-md-frames", type=int, default=None, help="override teacher-MD sampling frame count")
    parser.add_argument("--no-teacher-aug", action="store_true",
                        help="disable automatic teacher-MD augmentation when DFT frames are supplied alone")
    parser.add_argument("--dft-weight", type=float, default=3.0,
                        help="relative weight for DFT frames when mixing with teacher frames (default 3.0)")
    parser.add_argument("--energy-key", default=None, help="custom energy key in ASE info")
    parser.add_argument("--forces-key", default=None, help="custom forces key in ASE arrays")
    parser.add_argument("--stress-key", default=None, help="custom stress key in ASE info")
    parser.add_argument("--device", type=int, default=0, metavar="N", help="CUDA GPU index (default 0)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.set_defaults(run=run)


def _collapse_list(items):
    if not items:
        return None
    return items[0] if len(items) == 1 else list(items)


def run(args) -> int:
    from ..distill.auto import auto_distill

    res = auto_distill(
        teacher=args.teacher,
        head=args.head,
        checkpoint=args.checkpoint,
        sha256=args.sha256,
        trusted=bool(args.trusted),
        structure=_collapse_list(args.structure),
        md_data=_collapse_list(args.md_data),
        dft_data=_collapse_list(args.dft_data),
        output=args.output,
        speed=args.speed,
        max_speed_us=args.max_speed,
        budget=args.budget,
        epochs=args.epochs,
        n_md_frames=args.n_md_frames,
        no_teacher_aug=bool(args.no_teacher_aug),
        dft_weight=float(args.dft_weight),
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        device=int(args.device),
        seed=int(args.seed),
        verbose=not bool(args.quiet),
    )
    return 0 if (res.gate_passed and res.md_parity_passed) else 1
