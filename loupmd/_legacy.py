"""The bundle cache of the pre-release layout, until the layout move (PLAN.md S0-T03).

``python/loupmd.py`` still holds ``ensure_loup_bundle`` and the exporter fingerprint, and the scripts beside it
import it as the top-level module ``loupmd``. This package has the same name, so the file is loaded here by path
under a private name and its bundle functions are re-exported. Its ``LoupEngine`` is not: ``loupmd._core`` binds the
library for this package. The move deletes this module.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SOURCE = Path(__file__).resolve().parent.parent / "python" / "loupmd.py"


def _load():
    name = "loupmd._legacy_source"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SOURCE)
    if spec is None or spec.loader is None or not _SOURCE.is_file():
        raise ImportError(f"the bundle cache of the source checkout is missing: {_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


_source = _load()
ensure_loup_bundle = _source.ensure_loup_bundle
exporter_fingerprint = _source.exporter_fingerprint
EXPORTER_SOURCES = _source.EXPORTER_SOURCES
_model_digest = _source._model_digest
