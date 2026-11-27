"""Universal fine-tuning for LoupMD foundation models (MACE, ScaleShiftMACE, PolarMACE)."""
from __future__ import annotations

import importlib
import sys
import types


class _CallableFinetuneModule(types.ModuleType):
    """Makes ``loupmd.finetune(...)`` work directly even when the subpackage is imported."""

    def __call__(self, *args, **kwargs):
        return importlib.import_module(".core", self.__name__).finetune(*args, **kwargs)


def __getattr__(name: str):
    if name in ("FinetuneResult", "finetune"):
        mod = importlib.import_module(".core", __name__)
        return getattr(mod, name)
    if name in ("CudaMaceTrainer", "finetune_cuda"):
        mod = importlib.import_module(".cuda_backend", __name__)
        return getattr(mod, name)
    if name in ("STRATEGIES", "AnchorRegularizer", "EquivariantLoRALinear", "LoRALinear", "apply_lora_adapters", "merge_lora_adapters", "resolve_strategy"):
        mod = importlib.import_module(".adapters", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


sys.modules[__name__].__class__ = _CallableFinetuneModule
