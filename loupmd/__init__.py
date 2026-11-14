"""LoupMD: molecular dynamics with MACE potentials on a C++/CUDA engine."""
from __future__ import annotations

from ._core import EngineInfo, LoupEngine, LoupMDStats, last_error, library_path, library_version
from .errors import BackendError, InputError, LoupError

__version__ = "1.0.0a1"  # LOUP_VERSION of include/loupmd.h (tests/test_abi.py keeps the two equal)

# Until the layout move (PLAN.md S0-T03) the bundle cache lives in python/loupmd.py of the source checkout.
from ._legacy import EXPORTER_SOURCES, _model_digest, ensure_loup_bundle, exporter_fingerprint  # noqa: E402

__all__ = [
    "BackendError", "EngineInfo", "InputError", "LoupEngine", "LoupError", "LoupMDStats", "__version__",
    "ensure_loup_bundle", "exporter_fingerprint", "last_error", "library_path", "library_version",
]
