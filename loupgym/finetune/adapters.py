"""Fine-tuning parameter strategies, equivariant LoRA adapters, and anchor regularization."""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

STRATEGIES = ("auto", "full", "top", "readout", "lora")


def resolve_strategy(strategy: str, model: nn.Module, n_train: int) -> str:
    """Resolve ``strategy='auto'`` using model parameter count, interaction depth, and dataset size."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown fine-tuning strategy {strategy!r}; choose from {STRATEGIES}")
    if strategy != "auto":
        return strategy
    if n_train < 12:
        return "readout"
    n_params = sum(p.numel() for p in model.parameters())
    is_large_polar = type(model).__name__ == "PolarMACE" and len(getattr(model, "interactions", ())) >= 3
    if n_params > 40_000_000 or is_large_polar:
        return "top"
    return "full"


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a standard ``nn.Linear`` layer, with lossless float64 merge."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / max(1, self.rank)

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        dt = self.base.weight.dtype
        dev = self.base.weight.device
        self.lora_A = nn.Parameter(torch.empty((self.rank, self.base.in_features), dtype=dt, device=dev))
        self.lora_B = nn.Parameter(torch.zeros((self.base.out_features, self.rank), dtype=dt, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight + (self.lora_B @ self.lora_A) * self.scale

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base_out + lora_out

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        delta = (self.lora_B.double() @ self.lora_A.double()) * self.scale
        self.base.weight.data.copy_((self.base.weight.double() + delta).to(self.base.weight.dtype))
        return self.base


class EquivariantLoRALinear(nn.Module):
    """Low-rank adapter wrapping an ``e3nn.o3.Linear`` over each >=8x8 instruction weight block."""

    def __init__(self, base: o3.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        assert base.internal_weights and base.shared_weights, "EquivariantLoRALinear requires internal shared weights"
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / max(1, self.rank)

        self.base.weight.requires_grad_(False)
        if getattr(self.base, "bias", None) is not None and self.base.bias.numel() > 0:
            self.base.bias.requires_grad_(False)

        dt = self.base.weight.dtype
        dev = self.base.weight.device
        self._slice_specs: list[tuple[int, int, int, int | None]] = []
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()

        off = 0
        for inst in self.base.instructions:
            m_in, m_out = int(inst.path_shape[0]), int(inst.path_shape[1])
            sz = m_in * m_out
            if min(m_in, m_out) >= 8:
                r_eff = min(self.rank, min(m_in, m_out))
                a_p = nn.Parameter(torch.empty((r_eff, m_in), dtype=dt, device=dev))
                b_p = nn.Parameter(torch.zeros((m_out, r_eff), dtype=dt, device=dev))
                nn.init.kaiming_uniform_(a_p, a=math.sqrt(5.0))
                idx = len(self.lora_A)
                self.lora_A.append(a_p)
                self.lora_B.append(b_p)
                self._slice_specs.append((off, sz, m_in, idx))
            else:
                self._slice_specs.append((off, sz, m_in, None))
            off += sz

    @property
    def irreps_in(self):
        return self.base.irreps_in

    @property
    def irreps_out(self):
        return self.base.irreps_out

    @property
    def instructions(self):
        return self.base.instructions

    @property
    def weight_numel(self) -> int:
        return self.base.weight_numel

    @property
    def bias_numel(self) -> int:
        return self.base.bias_numel

    @property
    def bias(self):
        return self.base.bias

    def _delta_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        w_dt = dtype or self.base.weight.dtype
        w_dev = self.base.weight.device
        for _, sz, _, idx in self._slice_specs:
            if idx is None:
                parts.append(torch.zeros(sz, dtype=w_dt, device=w_dev))
            else:
                A = self.lora_A[idx].to(w_dt)
                B = self.lora_B[idx].to(w_dt)
                # e3nn.o3.Linear packs each instruction as [m_in, m_out] row-major
                parts.append(((A.t() @ B.t()) * self.scale).reshape(-1))
        return torch.cat(parts, dim=0) if parts else torch.zeros_like(self.base.weight, dtype=w_dt)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight + self._delta_weight()

    def forward(self, features: torch.Tensor, weight: torch.Tensor | None = None, bias: torch.Tensor | None = None):
        if weight is not None:
            return self.base(features, weight=weight, bias=bias)
        eff_w = self.base.weight + self._delta_weight()
        return self.base(features, weight=eff_w, bias=bias)

    @torch.no_grad()
    def merge(self) -> o3.Linear:
        delta64 = self._delta_weight(dtype=torch.float64)
        self.base.weight.data.copy_((self.base.weight.double() + delta64).to(self.base.weight.dtype))
        return self.base


def apply_lora_adapters(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> int:
    """Wrap eligible ``nn.Linear`` and ``o3.Linear`` modules inside ``model`` with LoRA adapters.

    Returns the count of wrapped modules. Skips ``node_embedding`` and projections narrower than 8.
    """
    wrapped = 0

    def _walk(parent: nn.Module, prefix: str = ""):
        nonlocal wrapped
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if "node_embedding" in full or "atomic_energies" in full:
                continue
            if isinstance(child, (LoRALinear, EquivariantLoRALinear)):
                continue
            if isinstance(child, nn.Linear):
                if min(child.in_features, child.out_features) >= 8:
                    setattr(parent, name, LoRALinear(child, rank=rank, alpha=alpha))
                    wrapped += 1
                continue
            if isinstance(child, o3.Linear) and child.internal_weights and child.shared_weights:
                has_big = any(min(int(i.path_shape[0]), int(i.path_shape[1])) >= 8 for i in child.instructions)
                if has_big:
                    setattr(parent, name, EquivariantLoRALinear(child, rank=rank, alpha=alpha))
                    wrapped += 1
                continue
            _walk(child, full)

    _walk(model)
    return wrapped


def merge_lora_adapters(model: nn.Module) -> int:
    """Merge all ``LoRALinear`` / ``EquivariantLoRALinear`` adapters back into vanilla modules."""
    merged = 0

    def _walk(parent: nn.Module):
        nonlocal merged
        for name, child in list(parent.named_children()):
            if isinstance(child, (LoRALinear, EquivariantLoRALinear)):
                setattr(parent, name, child.merge())
                merged += 1
            else:
                _walk(child)

    _walk(model)
    return merged


def _is_head_param(name: str) -> bool:
    head_prefixes = (
        "readouts.",
        "scale_shift.",
        "lr_source_maps.",
        "layer_feature_mixer.",
        "fukui_source_map.",
        "field_dependent_charges_maps.",
        "local_electron_energy.",
    )
    if name.startswith(head_prefixes):
        return True
    if name.startswith("products.") and ".linear." in name:
        return True
    return False


def _is_top_param(name: str, n_interactions: int) -> bool:
    last = max(0, n_interactions - 1)
    if _is_head_param(name):
        return True
    if name.startswith(f"interactions.{last}.") or name.startswith(f"products.{last}."):
        return True
    return False


def configure_trainable_params(
    model: nn.Module,
    strategy: str,
    lr: float,
    *,
    trunk_lr_mult: float = 0.25,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
) -> tuple[list[dict], list[tuple[str, nn.Parameter]], int, int]:
    """Configure parameter ``requires_grad`` and optimizer parameter groups for ``strategy``."""
    n_total = sum(p.numel() for p in model.parameters())
    for p in model.parameters():
        p.requires_grad_(False)

    n_inter = len(getattr(model, "interactions", ()))
    head_params: list[tuple[str, nn.Parameter]] = []
    trunk_params: list[tuple[str, nn.Parameter]] = []

    if strategy == "lora":
        apply_lora_adapters(model, rank=lora_rank, alpha=lora_alpha)
        for name, p in model.named_parameters():
            if "atomic_energies" in name:
                continue
            if ("lora_A" in name) or ("lora_B" in name) or name.startswith("readouts.") or name.startswith("scale_shift."):
                p.requires_grad_(True)
                head_params.append((name, p))
    elif strategy == "readout":
        for name, p in model.named_parameters():
            if "atomic_energies" in name:
                continue
            if (
                name.startswith("readouts.")
                or name.startswith("scale_shift.")
                or name.startswith("fukui_source_map.")
                or name.startswith("local_electron_energy.")
                or (name.startswith(f"products.{max(0, n_inter - 1)}.linear."))
            ):
                p.requires_grad_(True)
                head_params.append((name, p))
    elif strategy == "top":
        for name, p in model.named_parameters():
            if "atomic_energies" in name:
                continue
            if _is_top_param(name, n_inter):
                p.requires_grad_(True)
                if name.startswith("readouts.") or name.startswith("scale_shift."):
                    head_params.append((name, p))
                else:
                    trunk_params.append((name, p))
    elif strategy == "full":
        for name, p in model.named_parameters():
            if "atomic_energies" in name:
                continue
            p.requires_grad_(True)
            if _is_head_param(name):
                head_params.append((name, p))
            else:
                trunk_params.append((name, p))
    else:
        raise ValueError(f"Unsupported strategy: {strategy!r}")

    groups: list[dict] = []
    if head_params:
        groups.append({"params": [p for _, p in head_params], "lr": float(lr), "base_lr": float(lr)})
    if trunk_params:
        t_lr = float(lr) if strategy == "top" else float(lr) * float(trunk_lr_mult)
        groups.append({"params": [p for _, p in trunk_params], "lr": t_lr, "base_lr": t_lr})

    trainable = head_params + trunk_params
    n_trainable = sum(p.numel() for _, p in trainable)
    return groups, trainable, n_trainable, n_total


class AnchorRegularizer:
    """L2 anchor toward initial pretrained parameter values (excluding LoRA adapters)."""

    def __init__(self, named_params: Iterable[tuple[str, nn.Parameter]], weight: float = 0.0):
        self.weight = float(weight)
        self.anchors: list[tuple[nn.Parameter, torch.Tensor]] = []
        self.total_elems = 0
        if self.weight > 0.0:
            for name, p in named_params:
                if "lora_A" in name or "lora_B" in name:
                    continue
                self.anchors.append((p, p.detach().clone()))
                self.total_elems += p.numel()

    def loss(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.weight <= 0.0 or not self.anchors or self.total_elems == 0:
            return torch.zeros((), dtype=dtype, device=device)
        acc = torch.zeros((), dtype=dtype, device=device)
        for p, p0 in self.anchors:
            diff = p - p0.to(device=p.device, dtype=p.dtype)
            acc = acc + (diff * diff).sum()
        return self.weight * (acc / float(self.total_elems))
