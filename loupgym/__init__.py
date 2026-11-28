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
    "FixedD3BJSpec": ("dispersion_d3", "FixedD3BJSpec"),
    "CnD3BJSpec": ("dispersion_d3", "CnD3BJSpec"),
    "fit_fixed_d3bj": ("dispersion_d3", "fit_fixed_d3bj"),
    "fit_cn_d3bj": ("dispersion_d3", "fit_cn_d3bj"),
    "eval_fixed_d3bj_f64": ("dispersion_d3", "eval_fixed_d3bj_f64"),
    "eval_cn_d3bj_f64": ("dispersion_d3", "eval_cn_d3bj_f64"),
    "eval_d3_spec_f64": ("dispersion_d3", "eval_d3_spec_f64"),
    "transform_labels_for_d3": ("dispersion_d3", "transform_labels_for_d3"),
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
    "FixedD3BJSpec",
    "__version__",
    "auto_distill",
    "distill",
    "eval_fixed_d3bj_f64",
    "fasttrain",
    "finetune",
    "finetune_cuda",
    "fit_fixed_d3bj",
    "transform_labels_for_d3",
]
