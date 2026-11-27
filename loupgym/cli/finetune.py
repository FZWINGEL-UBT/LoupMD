"""``loupmd finetune``: fine-tune any LoupMD foundation MACE model and export a verified ``.loup`` bundle."""
from __future__ import annotations

_STRATEGIES = ("auto", "full", "top", "readout", "lora")
_BACKENDS = ("cuda", "pytorch")
_DTYPES = ("float32", "float64")


def add_parser(sub):
    parser = sub.add_parser(
        "finetune",
        help="fine-tune a foundation MACE model and export a verified .loup bundle",
        description=(
            "Fine-tunes any LoupMD foundation model (residual, density, multi-head MH-1, "
            "PolarMACE POLAR-1, OMOL-0, or OFF23/OFF24) on reference frames, saves a standard "
            ".model checkpoint, exports a torch-free .loup bundle, and runs the 3x gate check."
        ),
    )
    parser.add_argument("--model", required=True, metavar="MODEL",
                        help="foundation label (e.g. omat0-small, mh1, polar1-s) or .model path")
    parser.add_argument("--head", default=None, help="head name for multi-head models (default Default or single head)")
    parser.add_argument("--checkpoint", default=None, metavar="PATH", help="explicit checkpoint file for a label")
    parser.add_argument("--sha256", default=None, metavar="HEX", help="expected SHA-256 for an external checkpoint")
    parser.add_argument("--trusted", action="store_true", help="trust an external checkpoint without SHA-256 verification")
    parser.add_argument("--download", action="store_true", help="allow downloading an uncached foundation checkpoint")

    parser.add_argument("--train", required=True, action="append", metavar="FILE",
                        help="training frames (.extxyz/.xyz/.traj, repeatable)")
    parser.add_argument("--valid", action="append", metavar="FILE",
                        help="validation frames (.extxyz/.xyz/.traj, repeatable)")
    parser.add_argument("--test", action="append", metavar="FILE",
                        help="test frames for zero-shot vs fine-tuned evaluation (repeatable)")
    parser.add_argument("--replay-data", action="append", metavar="FILE",
                        help="replay frames mixed into training batches to prevent forgetting (repeatable)")
    parser.add_argument("-o", "--output", default="./finetune_out", metavar="PATH",
                        help="output directory, .model path, or .loup path (default ./finetune_out)")

    parser.add_argument("--backend", choices=_BACKENDS, default="cuda",
                        help="fine-tuning backend: cuda (pure-CUDA libloupmd_ft.so, default) or pytorch")
    parser.add_argument("--strategy", choices=_STRATEGIES, default="auto",
                        help="fine-tuning strategy: auto, full, top, readout, or lora (default auto)")
    parser.add_argument("--epochs", type=int, default=30, help="training epochs (default 30)")
    parser.add_argument("--lr", type=float, default=5e-4, help="peak learning rate for head/adapters (default 5e-4)")
    parser.add_argument("--trunk-lr-mult", type=float, default=0.25,
                        help="trunk LR multiplier for full/top strategies (default 0.25)")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank when --strategy lora (default 8)")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha scaling (default 16.0)")
    parser.add_argument("--anchor-weight", type=float, default=1e-2,
                        help="L2 anchor regularization weight toward pretrained weights (default 1e-2)")
    parser.add_argument("--replay-weight", type=float, default=0.25,
                        help="relative loss weight on --replay-data batches (default 0.25)")
    parser.add_argument("--energy-weight", type=float, default=1.0, help="energy loss weight (default 1.0)")
    parser.add_argument("--force-weight", type=float, default=100.0, help="force loss weight (default 100.0)")
    parser.add_argument("--stress-weight", type=float, default=None,
                        help="stress loss weight (default auto: 10.0 on periodic frames)")

    parser.add_argument("--batch-size", type=int, default=None, help="training batch size (default auto)")
    parser.add_argument("--grad-accum", type=int, default=None, help="gradient accumulation steps (default auto)")
    parser.add_argument("--dtype", choices=_DTYPES, default="float32", help="training dtype (default float32)")
    parser.add_argument("--no-cueq", action="store_true", help="disable CuEquivariance acceleration during training")
    parser.add_argument("--keep-e0", action="store_true",
                        help="skip least-squares E0 residual shift on the active head")

    parser.add_argument("--charge", type=float, default=None,
                        help="default total charge (used by PolarMACE / OMOL-0 when absent from frames)")
    parser.add_argument("--spin-multiplicity", type=int, default=None,
                        help="default spin multiplicity (used by PolarMACE / OMOL-0 when absent from frames)")
    parser.add_argument("--energy-key", default=None, help="custom energy key in ASE info")
    parser.add_argument("--forces-key", default=None, help="custom forces key in ASE arrays")
    parser.add_argument("--stress-key", default=None, help="custom stress key in ASE info")

    parser.add_argument("--no-export", action="store_true", help="skip exporting the .loup bundle")
    parser.add_argument("--no-verify", action="store_true", help="skip the 3x float32-vs-float64 .loup gate check")
    parser.add_argument("--device", type=int, default=0, metavar="N", help="CUDA GPU index or -1 for CPU (default 0)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.set_defaults(run=run)


def _collapse_list(items):
    if not items:
        return None
    return items[0] if len(items) == 1 else list(items)


def run(args) -> int:
    from ..finetune.core import finetune

    res = finetune(
        model=args.model,
        train_data=_collapse_list(args.train),
        valid_data=_collapse_list(args.valid),
        test_data=_collapse_list(args.test),
        replay_data=_collapse_list(args.replay_data),
        output=args.output,
        head=args.head,
        checkpoint=args.checkpoint,
        sha256=args.sha256,
        trusted=bool(args.trusted),
        download=bool(args.download),
        backend=args.backend,
        strategy=args.strategy,
        epochs=int(args.epochs),
        lr=float(args.lr),
        trunk_lr_mult=float(args.trunk_lr_mult),
        lora_rank=int(args.lora_rank),
        lora_alpha=float(args.lora_alpha),
        anchor_weight=float(args.anchor_weight),
        replay_weight=float(args.replay_weight),
        energy_weight=float(args.energy_weight),
        force_weight=float(args.force_weight),
        stress_weight=args.stress_weight,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        dtype=args.dtype,
        use_cueq=not bool(args.no_cueq),
        keep_e0=bool(args.keep_e0),
        charge=args.charge,
        spin_multiplicity=args.spin_multiplicity,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        export_loup=not bool(args.no_export),
        verify=not bool(args.no_verify),
        device=int(args.device),
        seed=int(args.seed),
        verbose=not bool(args.quiet),
    )
    if res.gate_passed is False:
        return 1
    return 0
