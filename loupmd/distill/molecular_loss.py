"""Standalone mace-torch 0.3.16 outer-loop adapter; invoked by train.py only."""
import json
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from mace.modules.loss import WeightedEnergyForcesStressLoss


class MolecularStressLoss(torch.nn.Module):
    """Add normalized MSEs of each molecule's net force and torque about its COM."""

    def __init__(self, native, net_scale, torque_scale):
        super().__init__()
        if not isinstance(native, WeightedEnergyForcesStressLoss):
            raise TypeError('expected the pinned stress loss in both training phases')
        self.native = native
        self.net_scale = net_scale
        self.torque_scale = torque_scale

    def forward(self, ref, pred, ddp=None):
        if ddp or torch.distributed.is_initialized():
            raise ValueError('molecular recipe supports single-device training only')
        # PyG concatenates mol_anchor without incrementing it (it is a local
        # atom index). Add the graph offset so same-ID molecules never mix.
        anchor = ref.mol_anchor.reshape(-1).long() + ref.ptr[ref.batch]
        df = pred['forces'] - ref.forces
        net = torch.zeros_like(df).index_add_(0, anchor, df)
        torque = torch.zeros_like(df).index_add_(0, anchor, torch.linalg.cross(ref.mol_arm, df))
        # Only the anchor rows are populated. Count molecules, not atoms/graphs.
        n = (anchor == torch.arange(len(df), device=df.device)).sum()
        molecular = (self.net_scale * net.square().sum() + self.torque_scale * torque.square().sum()) / (3*n)
        return self.native(ref, pred, ddp) + self.native.forces_weight * molecular


def run(plan):
    if version('mace-torch') != '0.3.16' or plan['mace_torch'] != '0.3.16':
        raise RuntimeError('molecular loss requires mace-torch==0.3.16')
    from mace.cli import run_train

    # Pinned integration: MACE constructs distinct losses in these factories.
    # Keep its optimization, EMA, SWA scheduler, checkpoints and export intact.
    scales = plan['molecular']
    get_loss, get_swa = run_train.get_loss_fn, run_train.get_swa

    def wrap(native):
        return MolecularStressLoss(native, scales['net_scale'], scales['torque_scale'])

    def phase_one(*args, **kwargs):
        return wrap(get_loss(*args, **kwargs))

    def phase_two(*args, **kwargs):
        swa, swas = get_swa(*args, **kwargs)
        swa.loss_fn = wrap(swa.loss_fn)
        return swa, swas

    args = run_train.tools.build_default_arg_parser().parse_args(plan['mace_args'])
    run_train.get_loss_fn, run_train.get_swa = phase_one, phase_two
    try:
        run_train.run(args)
    finally:
        run_train.get_loss_fn, run_train.get_swa = get_loss, get_swa


if __name__ == '__main__':
    run(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
