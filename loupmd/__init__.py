"""LoupMD: molecular dynamics with MACE potentials on a C++/CUDA engine."""
from __future__ import annotations

from ._core import LoupEngine, LoupMDStats, last_error, library_path
from .errors import BackendError, InputError, LoupError

# Until the layout move (PLAN.md S0-T03) the bundle cache lives in python/loupmd.py of the source checkout.
from ._legacy import EXPORTER_SOURCES, _model_digest, ensure_loup_bundle, exporter_fingerprint  # noqa: E402

__all__ = [
    "BackendError", "InputError", "LoupEngine", "LoupError", "LoupMDStats", "ensure_loup_bundle",
    "exporter_fingerprint", "last_error", "library_path",
]
