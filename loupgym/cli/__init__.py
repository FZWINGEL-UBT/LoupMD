"""The ``loupgym`` command: ``loupgym [--version] [--debug] COMMAND``.

Exit codes: 0 success, 1 a LoupMD/LoupGYM error (printed without a traceback unless ``--debug`` is given),
2 a usage error.
"""
from __future__ import annotations

import argparse
import sys

from loupmd.errors import LoupError

from . import finetune as _finetune

COMMANDS = (_finetune,)


def build_parser() -> argparse.ArgumentParser:
    from .. import __version__

    parser = argparse.ArgumentParser(
        prog="loupgym",
        description="Pure-CUDA fine-tuning and auto-distillation for LoupMD models.",
    )
    parser.add_argument("--version", action="version", version=f"loupgym {__version__}")
    parser.add_argument("--debug", action="store_true", help="show the traceback of an error")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    for module in COMMANDS:
        module.add_parser(sub)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_usage(sys.stderr)
        print("loupgym: error: a command is required", file=sys.stderr)
        return 2
    try:
        return int(args.run(args) or 0)
    except LoupError as err:
        if args.debug:
            raise
        print(f"error: {err}", file=sys.stderr)
        return 1
