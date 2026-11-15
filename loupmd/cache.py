"""LoupMD's cache root and its sub-directories (PLAN.md section 4.7).

One root, ``$LOUPMD_CACHE`` (default ``~/.cache/loupmd``; ``LOUP_CACHE_DIR`` is accepted as an older name), with
``models/``, ``bundles/``, ``plans/``, ``gate_refs/`` and ``lib/``. The root must lie outside the Python
installation (``sys.prefix``): a cache inside it would be removed with the environment and shipped by tools that
copy environments.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .errors import CacheLocationError

CACHE_VARIABLE = "LOUPMD_CACHE"
OLD_CACHE_VARIABLE = "LOUP_CACHE_DIR"
KINDS = ("models", "bundles", "plans", "gate_refs", "lib")


def cache_root(environ=None) -> Path:
    """The cache root: ``$LOUPMD_CACHE``, else ``$LOUP_CACHE_DIR``, else ``~/.cache/loupmd``. Not created."""
    environ = os.environ if environ is None else environ
    chosen = environ.get(CACHE_VARIABLE) or environ.get(OLD_CACHE_VARIABLE)
    root = Path(os.path.expanduser(chosen)) if chosen else Path.home() / ".cache" / "loupmd"
    root = root.resolve()
    prefix = Path(sys.prefix).resolve()
    if root == prefix or prefix in root.parents:
        raise CacheLocationError(
            f"the cache root {root} lies inside the Python installation {prefix}",
            why="a cache there is deleted with the environment and copied with it",
            next_step=f"Set {CACHE_VARIABLE} to a directory outside the environment.")
    return root


def cache_dir(kind: str, create: bool = True, environ=None) -> Path:
    """``<cache root>/<kind>`` for one of ``KINDS``."""
    if kind not in KINDS:
        raise ValueError(f"unknown cache kind {kind!r}; kinds are {', '.join(KINDS)}")
    path = cache_root(environ) / kind
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
