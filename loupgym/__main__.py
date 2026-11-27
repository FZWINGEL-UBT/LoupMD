"""Allow ``python -m loupgym ...`` to run the LoupGYM CLI."""
from __future__ import annotations

import sys

from .cli import main

raise SystemExit(main(sys.argv[1:]))
