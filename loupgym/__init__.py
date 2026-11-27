"""LoupGYM: pure-CUDA foundation fine-tuning and auto-distillation for LoupMD."""
from __future__ import annotations

import importlib

__version__ = "0.0.2"

# Lazy front end so `import loupgym` stays light and zero-torch by default.
_LAZY = {
    "auto_distill": ("distill", "auto_distill"),
    "DistillResult": ("distill", "DistillResult"),
    "distill": ("distill", None),
    "finetune": ("finetune", "finetune"),
    "finetune_cuda": ("finetune", "finetune_cuda"),
    "FinetuneResult": ("finetune", "FinetuneResult"),
    "CudaMaceTrainer": ("finetune", "CudaMaceTrainer"),
    "fasttrain": ("fasttrain", None),
}


def __getattr__(name: str):
    if name in _LAZY:
        module_name, attribute = _LAZY[name]
        module = importlib.import_module(f"{__name__}.{module_name}")
        return module if attribute is None else getattr(module, attribute)
    raise AttributeError(f"module 'loupgym' has no attribute {name!r}")


__all__ = [
    "CudaMaceTrainer",
    "DistillResult",
    "FinetuneResult",
    "__version__",
    "auto_distill",
    "distill",
    "fasttrain",
    "finetune",
    "finetune_cuda",
]
