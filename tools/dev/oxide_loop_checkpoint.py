"""CPU-only check of an actual oxide resume plan and trusted epoch-110 parent, without training.

On leto, upload stock-trainer/train.py, resume-plan.json and pre_stage_two.pt into
one directory. Run `CUDA_VISIBLE_DEVICES='' python oxide_loop_checkpoint.py DIR`.
The output on stdout identifies the phase/weights/optimizer loaded by pinned MACE.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    d = p.parse_args().directory
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('this is a CPU check: set CUDA_VISIBLE_DEVICES to the empty string')
    import torch
    from importlib.metadata import version
    from mace.tools.checkpoint import CheckpointIO
    if hashlib.sha256((d/'train.py').read_bytes()).hexdigest() != '7461141c9323e25460935f76bbc53a2039c131f7778c9e7a3d95ae5d35ccf05b':
        raise ValueError('not the frozen paired trainer')
    spec = importlib.util.spec_from_file_location('stock_trainer', d/'train.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    plan = json.loads((d/'resume-plan.json').read_text())
    if version('mace-torch') != plan['mace_torch'] or plan['mace_torch'] != '0.3.16':
        raise ValueError('requires the pinned stage-0 training backend')
    source = d / 'pre_stage_two.pt'
    if trainer.file_digest(source) != plan['resume']['sha256']:
        raise ValueError('wrong epoch-110 checkpoint for this resume plan')
    plan['resume']['checkpoint'] = str(source)
    def same(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b
    with TemporaryDirectory(dir=d) as tmp:
        path = Path(tmp) / 'checkpoints'
        trainer._seed_resume(plan, path)
        seed = plan['recipe']['seed']
        boundary = plan['recipe']['stage_two_start']
        src, dst = trainer.restart_paths(path, plan)
        if trainer.file_digest(src) != plan['resume']['sha256']:
            raise ValueError('source EMA changed')
        io = CheckpointIO(str(path), tag=f'student_run-{seed}', swa_start=boundary)
        pre, pre_epoch = io.load_latest(swa=False, device='cpu')
        resume, resume_epoch = io.load_latest(swa=True, device='cpu')
        if (pre_epoch, resume_epoch) != (boundary - 1, boundary):
            raise ValueError('MACE selected the wrong restart epoch')
        groups = pre['optimizer']['param_groups']
        for group in groups:
            group['lr'] = group['swa_lr']
        if not same(pre, resume):
            raise ValueError('altered EMA weights, optimizer moments, scheduler, or unexpected learning rate')
        print(json.dumps(dict(mace_torch=version('mace-torch'), dtype=plan['recipe']['dtype'], pair_id=plan['pair_id'],
            pre_sha256=trainer.file_digest(src), pre_epoch=pre_epoch, restart_epoch=resume_epoch,
            restart_filename=dst.name, learning_rates=[g['lr'] for g in groups], optimizer_groups=len(groups),
            unchanged_except_phase_two_lr=True, optimizer_steps_executed=0)))


if __name__ == '__main__':
    main()
