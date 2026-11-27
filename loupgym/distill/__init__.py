"""Distillation tooling and FastLR-SK Auto-Distiller."""
from __future__ import annotations

import importlib


def __getattr__(name: str):
    if name in ("DistillResult", "auto_distill"):
        mod = importlib.import_module(".auto", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
