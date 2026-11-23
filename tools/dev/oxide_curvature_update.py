"""CPU acceptance only: ONE disposable stock optimizer step on the actual corrected EMA and sealed pairs.

Supply the sto320 artifact root; stop before any validation/test evaluation or student/checkpoint publication.
The first active (third) base minibatch of epoch 111 is chosen directly; no preceding updates are performed.
"""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from loupmd.distill import curvature as c, curvature_train as child, train


def run(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or child.version('mace-torch') != '0.3.16':
        raise RuntimeError('requires explicit CPU-only use and mace-torch==0.3.16')
    import torch
    from mace.cli import run_train
    from mace.tools import AtomicNumberTable
    from mace.tools.train import take_step

    class Done(Exception):
        """Stop the real MACE CLI before the fixed validation/test tables and checkpoint writers."""

    base, pairdir = root/'base-labels-v2', root/'oxide-curvature-v1'
    with TemporaryDirectory() as tmp:
        out = Path(tmp)/'job'
        args = SimpleNamespace(out=out, train=base/'train.xyz', valid=base/'valid.xyz', teacher_test=base/'teacher-test.xyz',
                               pairs=pairdir/'pairs.npz', seal=pairdir/'pairs.sha256',
                               parent=root/'v2-l4-s64l0-fp32-seed1-r2/checked')
        job, pairs, targets = child.prepare(args)
        train.write_inputs(job, out/'inputs')
        train._seed_resume(job, out/'work/checkpoints')
        options = run_train.tools.build_default_arg_parser().parse_args(job['mace_args'])
        options.device = 'cpu'
        result = {}

        def single(**kw):
            model, opt, ema = kw['model'], kw['optimizer'], kw['ema']
            if (kw['start_epoch'] != 111 or kw['swa'].start != 111 or len(kw['train_loader']) != 75 or
                    not opt.state or any(g['lr'] != .001 for g in opt.param_groups)):
                raise ValueError('did not restore the matched epoch, Adam moments, learning rate and data')
            slots = c.schedule(111)
            it = iter(kw['train_loader'])
            for i in slots:
                batch = next(it)
                if i is not None:
                    break
            _, plus, minus = pairs[i]
            graphs = child.paired_batch(plus, minus, AtomicNumberTable(model.atomic_numbers.tolist()))
            target = torch.as_tensor(targets[i], dtype=torch.float32)
            weight = job['curvature']['on_batch_weight_A2']
            native = kw['swa'].loss_fn
            def combined(ref, pred):
                ordinary = native(ref, pred)
                h = child.auxiliary(model, graphs, target, 'cpu')
                result.update(base_loss=float(ordinary.detach()), curvature_mse=float(h.detach()))
                return ordinary + weight*h
            param = next(model.parameters()); before = param.detach().clone()
            loss, _ = take_step(model, combined, batch, opt, ema, kw['output_args'], kw['max_grad_norm'], 'cpu')
            dx = (param.detach()-before).abs().max().item()
            if not torch.isfinite(loss) or not 0 < dx < float('inf') or ema.num_updates != 1:
                raise ValueError('nonfinite or missing optimizer/EMA update')
            result.update(epoch=111, optimizer_steps=1, ema_updates=ema.num_updates, pair_id=i,
                          pairs_sha256=job['curvature']['pairs_sha256'], boundary_sha256=job['resume']['sha256'],
                          on_batch_weight_A2=weight, total_loss=float(loss.detach()), max_first_parameter_update=dx)
            raise Done

        with patch.object(run_train.tools, 'train', single):
            try:
                run_train.run(options)
            except Done:
                return result
        raise RuntimeError('pinned CLI did not take the disposable step')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sto320', type=Path)
    print(json.dumps(run(parser.parse_args().sto320.resolve()), allow_nan=False), flush=True)
