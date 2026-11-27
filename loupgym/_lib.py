"""Shared native CUDA library discovery (and optional dev auto-build) for LoupGYM."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent


def find_native_lib(
    filename: str,
    *,
    env_var: str | None = None,
    make_target: str | None = None,
    auto_build: bool = False,
    extra_candidates: tuple[Path, ...] = (),
) -> Path:
    """Locate a compiled LoupGYM ``.so`` across env override, wheel ``_lib/``, repo root, and ``build/``."""
    candidates: list[Path] = []
    if env_var:
        env_p = os.environ.get(env_var)
        if env_p:
            candidates.append(Path(env_p))
    candidates.extend([
        _PKG_DIR / "_lib" / filename,
        _REPO_ROOT / filename,
        _REPO_ROOT / "build" / filename,
    ])
    candidates.extend(sorted((_REPO_ROOT / "build").glob(f"sm_*/{filename}"), reverse=True))
    candidates.extend(extra_candidates)

    for c in candidates:
        if c.is_file():
            return c

    if auto_build and make_target and (_REPO_ROOT / "Makefile").is_file():
        res = subprocess.run(["make", make_target], cwd=str(_REPO_ROOT), check=False, capture_output=True, text=True)
        for c in (
            _REPO_ROOT / filename,
            _REPO_ROOT / "build" / filename,
            *sorted((_REPO_ROOT / "build").glob(f"sm_*/{filename}"), reverse=True),
        ):
            if c.is_file():
                return c
        raise FileNotFoundError(
            f"Could not build {filename} via `make {make_target}`.\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )

    hint = f" Build it with `make {make_target}`." if make_target else ""
    raise FileNotFoundError(f"Native library {filename} not found.{hint}")
