"""Standalone, non-baseline child of the corrected FP32 s64l0 EMA boundary. No phase one."""
import argparse
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil

if __package__:
    from . import curvature as c, train
else:
    import curvature as c
    import train


def paired_batch(plus, minus, table):
    from mace import data
    from mace.tools.torch_geometric.batch import Batch

    # No shared neighbor list: independently search each actual geometry at the student's cutoff.
    graphs = [data.AtomicData.from_config(data.Configuration(a.numbers, a.positions, {}, {},
                    a.cell.array, tuple(a.pbc)), z_table=table, cutoff=5.) for a in (plus, minus)]
    batch = Batch.from_data_list(graphs)
    if batch.ptr.tolist() != [0, len(plus), 2 * len(plus)]:
        raise ValueError('ambiguous pair graph boundaries')
    return batch.to_dict()


def auxiliary(model, pair, target, device):
    # Make a fresh dict: stock MACE modifies the position/displacement graph. Only
    # forces need differentiating; no stress labels, coordinate Hessian or third derivatives.
    batch = {k: tensor.detach().to(device) for k, tensor in pair.items()}
    force = model(batch, training=True, compute_force=True, compute_stress=False)['forces']
    plus, minus = force.reshape(2, *target.shape).unbind()
    return (c.difference(plus, minus) - target.to(device)).square().mean()


def run_stock(job, pairs, targets):
    import torch
    from mace.cli import run_train
    from mace.modules.loss import WeightedEnergyForcesStressLoss
    from mace.tools import AtomicNumberTable

    outer = importlib.import_module('mace.tools.train')
    native_epoch = outer.train_one_epoch
    cache = []
    epochs = []

    def paired_epoch(**kw):
        epoch = kw['epoch']
        native = kw['loss_fn']
        if (epoch != 111 + len(epochs) or len(kw['data_loader']) != 75 or
                not isinstance(native, WeightedEnergyForcesStressLoss) or
                [float(native.energy_weight), float(native.forces_weight), float(native.stress_weight)] != [1000, 100, 10]
                or kw['distributed'] or kw['ema'] is None or torch.get_default_dtype() != torch.float32):
            raise RuntimeError('not the expected single-device stage-two recipe')
        if not cache:
            table = AtomicNumberTable(kw['model'].atomic_numbers.tolist())
            for (_, plus, minus), target in zip(pairs, targets, strict=True):
                cache.append((paired_batch(plus, minus, table), torch.as_tensor(target, dtype=torch.float32)))
        slots = iter(c.schedule(epoch))
        used = []

        def training_loss(ref, pred):
            base = native(ref, pred)  # unchanged normalizations on all four original graphs
            i = next(slots)
            if i is None:
                return base
            used.append(i)
            return base + job['curvature']['on_batch_weight_A2'] * auxiliary(kw['model'], *cache[i], kw['device'])

        # 0.3.16 calls loss_fn exactly once inside take_step before clipping, AdamW
        # and EMA. The epoch-local wrapper never enters validation, SWA or final tables.
        native_epoch(**{**kw, 'loss_fn': training_loss})
        if list(slots) or sorted(used) != list(range(c.PAIRS)):
            raise RuntimeError('incomplete pair epoch')
        epochs.append(epoch)
        print(f'Curvature epoch {epoch}: pair IDs in update order {used}', flush=True)

    args = run_train.tools.build_default_arg_parser().parse_args(job['mace_args'])
    outer.train_one_epoch = paired_epoch
    try:
        run_train.run(args)
    finally:
        outer.train_one_epoch = native_epoch
    if epochs != list(range(111, 150)):
        raise RuntimeError('incomplete phase two')


def prepare(args):
    if train.file_digest(args.parent / 'manifest.json') != c.PARENT:
        raise ValueError('requires the corrected centered-FP32 seed-1 parent manifest')
    parent = json.loads((args.parent / 'manifest.json').read_text())
    checkpoint = (args.parent / 'pre_stage_two.pt').resolve()
    if train.file_digest(checkpoint) != parent['pre_stage_two_sha256']:
        raise ValueError('wrong pre-phase-two checkpoint')
    job = train.plan_job(train.parse_args(['--train', str(args.train), '--valid', str(args.valid), '--teacher-test',
                         str(args.teacher_test), '--out', str(args.out), '--dtype', 'float32', '--stage-two-start', '111']))
    if (job['recipe'] != parent['recipe'] or [s['sha256'] for s in job['sources']] !=
            [s['sha256'] for s in parent['sources']] or job['pair_id'] != parent['pair_id']):
        raise ValueError('not the matched frozen base recipe')
    digest, name = args.seal.read_text().split()
    if name != args.pairs.name:
        raise ValueError('seal filename mismatch')
    pairs, targets, summary = c.checked(args.pairs, digest, args.train)
    job.update(curvature=summary, experiment='oxide-curvature-phase-two-child',
               parent_manifest_sha256=c.PARENT, parent_pair_id=parent['pair_id'],
               pair_id=f"curvature-v1:{digest}:{parent['pair_id']}", energy_offsets_eV=parent['energy_offsets_eV'],
               avg_num_neighbors=parent['avg_num_neighbors'],
               resume=dict(checkpoint=str(checkpoint), sha256=parent['pre_stage_two_sha256']))
    job['mace_args'] = train._arguments(job)
    return job, pairs, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'valid', 'teacher-test', 'pairs', 'seal', 'parent', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    job, pairs, targets = prepare(args)
    if args.plan:
        print(json.dumps(job, indent=2))
        return
    if version('mace-torch') != train.MACE_VERSION:
        raise RuntimeError('requires stock mace-torch==0.3.16')
    out = Path(job['out'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'plan.json').write_text(json.dumps(job, indent=2) + '\n')
    train.write_inputs(job, out / 'inputs')
    checkpoints = out / 'work/checkpoints'
    train._seed_resume(job, checkpoints)
    run_stock(job, pairs, targets)
    pre, _ = train.restart_paths(checkpoints, job)
    if (train.file_digest(pre) != job['resume']['sha256'] or
            not (checkpoints / 'student_run-1_epoch-149_swa.pt').is_file()):
        raise RuntimeError('missing matched boundary or final checkpoint')
    avg = train._export_physical(job, out)
    if avg != job['avg_num_neighbors']:
        raise ValueError('neighbor normalization changed')
    shutil.copyfile(pre, out / 'pre_stage_two.pt')
    job.update(student_epoch=149, pre_stage_two_epoch=110, pre_stage_two_sha256=train.file_digest(pre),
               student_sha256=train.file_digest(out / 'student.model'))
    (out / 'manifest.json').write_text(json.dumps(job, indent=2) + '\n')


if __name__ == '__main__':
    main()
