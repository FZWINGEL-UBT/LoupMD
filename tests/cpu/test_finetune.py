"""CPU unit tests for ``loupgym.finetune`` adapters, strategies, regularizers, and CLI."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from e3nn import o3

import loupmd
import loupgym
from loupgym.cli import build_parser
from loupgym.finetune.adapters import (
    AnchorRegularizer,
    EquivariantLoRALinear,
    LoRALinear,
    apply_lora_adapters,
    configure_trainable_params,
    merge_lora_adapters,
    resolve_strategy,
)


class _DummyPolarMACE(nn.Module):
    def __init__(self, n_inter: int = 3):
        super().__init__()
        self.interactions = nn.ModuleList([nn.Linear(4, 4) for _ in range(n_inter)])


class _BigDummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.big = nn.Parameter(torch.zeros(40_000_001))


def test_lora_linear_forward_grad_and_exact_merge():
    torch.manual_seed(0)
    base = nn.Linear(16, 12, bias=True, dtype=torch.float64)
    wrapped = LoRALinear(base, rank=4, alpha=8.0)

    # Non-zero lora_B so adapter contributes to output and merge.
    with torch.no_grad():
        wrapped.lora_B.normal_(0.0, 0.2)

    x = torch.randn(5, 16, dtype=torch.float64)
    y_before = wrapped(x)
    loss = y_before.pow(2).sum()
    loss.backward()
    assert wrapped.lora_A.grad is not None and wrapped.lora_A.grad.abs().sum() > 0
    assert wrapped.lora_B.grad is not None and wrapped.lora_B.grad.abs().sum() > 0
    assert base.weight.grad is None
    assert base.bias.grad is None

    merged = wrapped.merge()
    assert merged is base
    with torch.no_grad():
        y_after = merged(x)
    assert (y_before.detach() - y_after).abs().max().item() < 1e-12


def test_equivariant_lora_linear_forward_grad_and_exact_merge():
    torch.manual_seed(1)
    irreps_in = o3.Irreps("12x0e + 10x1o + 4x2e")
    irreps_out = o3.Irreps("12x0e + 8x1o + 4x2e")
    base = o3.Linear(irreps_in, irreps_out, internal_weights=True, shared_weights=True, biases=False).double()
    wrapped = EquivariantLoRALinear(base, rank=4, alpha=8.0)

    # 4x2e instruction (<8) should not allocate a LoRA slice; 12x0e and 10x1o->8x1o should.
    assert len(wrapped.lora_A) == 2
    assert len(wrapped.lora_B) == 2

    with torch.no_grad():
        for b in wrapped.lora_B:
            b.normal_(0.0, 0.15)

    x = irreps_in.randn(6, -1, dtype=torch.float64)
    y_before = wrapped(x)
    y_before.pow(2).sum().backward()
    for a, b in zip(wrapped.lora_A, wrapped.lora_B):
        assert a.grad is not None and a.grad.abs().sum() > 0
        assert b.grad is not None and b.grad.abs().sum() > 0
    assert base.weight.grad is None

    merged = wrapped.merge()
    assert merged is base
    with torch.no_grad():
        y_after = merged(x)
    assert (y_before.detach() - y_after).abs().max().item() < 1e-12


def test_apply_and_merge_lora_adapters_roundtrip():
    torch.manual_seed(2)

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_embedding = nn.Linear(16, 16)
            self.atomic_energies = nn.Parameter(torch.ones(4))
            self.eq = o3.Linear("12x0e + 8x1o", "12x0e + 8x1o")
            self.small = nn.Linear(4, 4)
            self.readouts = nn.ModuleList([nn.Linear(12, 16), nn.Linear(16, 1)])

        def forward(self, x_lin, x_eq):
            return self.readouts[0](x_lin) + self.eq(x_eq)[:, :16]

    m = _Block().double()
    n_wrapped = apply_lora_adapters(m, rank=4, alpha=8.0)
    # Wraps eq, readouts.0 (12x16); skips node_embedding, small (<8), readouts.1 (min=1<8).
    assert n_wrapped == 2
    assert isinstance(m.node_embedding, nn.Linear)
    assert isinstance(m.small, nn.Linear)
    assert isinstance(m.eq, EquivariantLoRALinear)
    assert isinstance(m.readouts[0], LoRALinear)

    with torch.no_grad():
        m.readouts[0].lora_B.normal_(0.0, 0.1)
        m.eq.lora_B[0].normal_(0.0, 0.1)

    x_lin = torch.randn(4, 12, dtype=torch.float64)
    x_eq = o3.Irreps("12x0e + 8x1o").randn(4, -1, dtype=torch.float64)
    with torch.no_grad():
        out_before = m(x_lin, x_eq)

    n_merged = merge_lora_adapters(m)
    assert n_merged == 2
    assert not any(isinstance(mod, (LoRALinear, EquivariantLoRALinear)) for mod in m.modules())
    with torch.no_grad():
        out_after = m(x_lin, x_eq)
    assert (out_before - out_after).abs().max().item() < 1e-12


def test_resolve_strategy_auto_branches():
    small_model = nn.Sequential(nn.Linear(8, 8))
    assert resolve_strategy("auto", small_model, n_train=8) == "readout"
    assert resolve_strategy("auto", small_model, n_train=12) == "full"

    big_model = _BigDummyModel()
    assert resolve_strategy("auto", big_model, n_train=50) == "top"

    # PolarMACE check relies on class name == "PolarMACE"
    PolarMACE = type("PolarMACE", (_DummyPolarMACE,), {})
    polar3 = PolarMACE(n_inter=3)
    polar2 = PolarMACE(n_inter=2)
    assert resolve_strategy("auto", polar3, n_train=50) == "top"
    assert resolve_strategy("auto", polar2, n_train=50) == "full"

    assert resolve_strategy("lora", big_model, n_train=4) == "lora"
    with pytest.raises(ValueError, match="Unknown fine-tuning strategy"):
        resolve_strategy("bogus", small_model, n_train=20)


def test_anchor_regularizer_and_lora_exclusion():
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.atomic_energies = nn.Parameter(torch.zeros(4))
            self.readouts = nn.ModuleList([nn.Linear(12, 12)])

    net = _Net()
    groups, trainable, n_trainable, _ = configure_trainable_params(net, "lora", lr=1e-3, lora_rank=4)
    assert len(groups) == 1
    assert n_trainable > 0
    # atomic_energies is never trainable
    assert not net.atomic_energies.requires_grad

    reg = AnchorRegularizer(trainable, weight=2.0)
    # LoRA A/B are excluded, only readouts.0.base.bias was trainable? Wait: readouts.0.base.bias is frozen under LoRA.
    # Verify non-LoRA trainable tensor anchors properly:
    p_non_lora = nn.Parameter(torch.tensor([1.0, 2.0]))
    reg2 = AnchorRegularizer([("readouts.0.weight", p_non_lora), ("readouts.0.lora_A", nn.Parameter(torch.ones(2, 2)))], weight=3.0)
    assert len(reg2.anchors) == 1
    assert reg2.total_elems == 2
    assert reg2.loss(torch.device("cpu"), torch.float32).item() == 0.0
    with torch.no_grad():
        p_non_lora.add_(1.0)  # diff = [1, 1] -> sum sq = 2 -> mean sq = 1 -> * 3.0 = 3.0
    assert abs(reg2.loss(torch.device("cpu"), torch.float32).item() - 3.0) < 1e-6


def test_cli_finetune_parser_and_public_api():
    parser = build_parser()
    args = parser.parse_args([
        "finetune",
        "--model", "omat0-small",
        "--train", "a.xyz",
        "--train", "b.xyz",
        "--strategy", "lora",
        "--epochs", "15",
        "--no-cueq",
        "--keep-e0",
        "--no-export",
    ])
    assert args.command == "finetune"
    assert args.model == "omat0-small"
    assert args.train == ["a.xyz", "b.xyz"]
    assert args.strategy == "lora"
    assert args.backend == "cuda"
    assert args.epochs == 15
    assert args.no_cueq is True
    assert args.keep_e0 is True
    assert args.no_export is True

    args_pt = parser.parse_args(["finetune", "--model", "omat0-small", "--train", "a.xyz", "--backend", "pytorch"])
    assert args_pt.backend == "pytorch"

    assert callable(loupgym.finetune)
    assert loupgym.FinetuneResult.__name__ == "FinetuneResult"
    from loupgym.finetune import CudaMaceTrainer, finetune_cuda
    assert callable(finetune_cuda)
    assert CudaMaceTrainer.__name__ == "CudaMaceTrainer"
    # Verify clean separation: loupmd exposes zero training entry points
    assert not hasattr(loupmd, "finetune")
    assert not hasattr(loupmd, "FinetuneResult")
    from loupmd.cli import build_parser as loupmd_build_parser
    with pytest.raises(SystemExit):
        loupmd_build_parser().parse_args(["finetune", "--model", "omat0-small", "--train", "a.xyz"])
    with pytest.raises(SystemExit):
        loupmd_build_parser().parse_args(["distill", "--teacher", "omat0-small"])


def test_normalize_frame_extracts_single_point_calculator():
    import numpy as np
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from loupgym.data import normalize_frame

    at = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], cell=[10.0, 10.0, 10.0], pbc=True)
    forces = np.array([[0.1, 0.0, -0.2], [-0.1, 0.0, 0.2]])
    stress6 = np.array([-0.01, -0.02, -0.03, 0.001, 0.002, 0.003])
    at.calc = SinglePointCalculator(at, energy=-12.34, forces=forces, stress=stress6)

    norm, ok = normalize_frame(at)
    assert ok is True
    assert norm.calc is None
    assert abs(norm.info["REF_energy"] - (-12.34)) < 1e-12
    assert np.allclose(norm.arrays["REF_forces"], forces)
    assert norm.info["REF_stress"].shape == (3, 3)
