"""Static engine checks on real frames: parity with mace-torch on a padded batch, and epoch timing.

    parity: --mode parity --dtype float64   (energy/forces/stress/loss/parameter gradients)
    timing: --mode bench  [--no-graph]      (GraphTrainer epochs, Adam/AMSGrad + EMA as mace)
Run under the local GPU reservation.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch


def parity(args):
    from fasttrain_parity import load_items
    from mace.tools import torch_geometric
    from mace.modules.loss import WeightedEnergyForcesStressLoss
    from loupmd.fasttrain.data import frames_from_dataset
    from loupmd.fasttrain.engine import PaddedFrames, FastStudent
    dtype, dev = getattr(torch, args.dtype), torch.device('cuda')
    model = torch.load(args.model, map_location='cpu').to(dtype)
    items = load_items(args.xyz, model, dtype, 7)
    model = model.to(dev)
    we, wf, ws = (float(x) for x in args.weights.split(','))
    loss_fn = WeightedEnergyForcesStressLoss(we, wf, ws).to(dev)
    pick = [5, 1, 3, 0]                     # a batch of 4 in a padded shape of 5 graph slots
    sub = [items[i] for i in pick]
    batch = next(iter(torch_geometric.dataloader.DataLoader(sub, batch_size=len(sub), shuffle=False))).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    out = model(batch.to_dict(), training=True, compute_force=True, compute_stress=True)
    loss = loss_fn(pred=out, ref=batch)
    g_ref = torch.autograd.grad(loss, params)

    frames = frames_from_dataset(items, model)
    pf = PaddedFrames(frames, dtype, dev, batch_size=5, r_max=float(model.r_max))
    h = pf.host_batch(pick)
    idx = {}
    for k, v in h.items():
        if k == 'counts':
            idx[k] = torch.from_numpy(v).to(device=dev, dtype=dtype)
        elif k in pf.INT32:
            idx[k] = torch.from_numpy(v.astype(np.int32)).to(dev)
        else:
            idx[k] = torch.from_numpy(v.astype(np.int64)).to(dev)
    bt = pf.gather(idx)
    st = FastStudent(model)
    e, f, s = st.evaluate(bt, training=True)
    le, lf, ls = st.loss_terms(bt, e, f, s)
    loss2 = we * le + wf * lf + ws * ls
    g_new = torch.autograd.grad(loss2, params)
    G = len(pick)
    n_real = int(h['counts'][0])
    rel = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-300))
    print(f'dtype {dtype} N_pad {pf.N_pad} (real {n_real}, dummy nodes {pf.N_pad - n_real}) '
          f'E_pad {pf.E_pad} (real {int(batch.edge_index.shape[1])})')
    print('energy max abs diff', float((out['energy'] - e[:G]).abs().max()),
          'dummy graph energy', float(e[G:].abs().max()))
    ref_rows = {tuple(np.round(r, 12)): i for i, r in enumerate(batch.forces.detach().cpu().double().numpy())}
    fr = bt.f_ref[:n_real].detach().cpu().double().numpy()
    perm = torch.tensor([ref_rows[tuple(np.round(r, 12))] for r in fr], device=dev)
    print('forces rel diff', rel(f[:n_real], out['forces'][perm]), 'dummy forces max', float(f[n_real:].abs().max()))
    print('stress rel diff', rel(s[:G], out['stress']))
    print('loss mace', float(loss), 'engine', float(loss2))
    worst = 0.0
    for n, a, c in zip([n for n, p in model.named_parameters() if p.requires_grad], g_ref, g_new):
        worst = max(worst, rel(c, a))
    print('worst grad rel diff', worst)


def bench(args):
    from fasttrain_parity import load_items
    from mace.modules.loss import WeightedEnergyForcesStressLoss
    from torch_ema import ExponentialMovingAverage
    from loupmd.fasttrain.engine import GraphTrainer
    dtype, dev = torch.float32, torch.device('cuda')
    model = torch.load(args.model, map_location='cpu').to(dtype)
    items = load_items(args.xyz, model, dtype, args.frames)
    model = model.to(dev)
    loss_fn = WeightedEnergyForcesStressLoss(1.0, 100.0, 1.0).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=0.01, amsgrad=True)
    ema = ExponentialMovingAverage(params, decay=0.99)
    t0 = time.time()
    tr = GraphTrainer(model, items, batch_size=args.batch, dtype=dtype, use_graph=not args.no_graph)
    print(f'setup {time.time() - t0:.1f}s N_pad {tr.data.N_pad} E_pad {tr.data.E_pad} dummy nodes {tr.data.D}')
    rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        order = rng.permutation(len(items))
        batches = [order[i:i + args.batch].tolist() for i in range(0, len(items), args.batch)]
        torch.cuda.synchronize()
        t = time.time()
        losses, sec = tr.run_epoch(batches, loss_fn, opt, ema, 10.0)
        torch.cuda.synchronize()
        dt = time.time() - t
        print(f'epoch {ep}: {dt:.3f}s ({1e3 * dt / len(batches):.2f} ms/step) loss first {losses[0]:.5g} '
              f'last {losses[-1]:.5g} mean {np.mean(losses):.5g}  peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')
    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        order = rng.permutation(len(items))[:40]
        batches = [order[i:i + args.batch].tolist() for i in range(0, len(order), args.batch)]
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            tr.run_epoch(batches, loss_fn, opt, ema, 10.0)
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=45, max_name_column_width=70))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='parity', choices=('parity', 'bench'))
    ap.add_argument('--model', required=True)
    ap.add_argument('--xyz', required=True)
    ap.add_argument('--dtype', default='float64')
    ap.add_argument('--weights', default='1,100,1')
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--no-graph', action='store_true')
    ap.add_argument('--profile', action='store_true')
    args = ap.parse_args()
    parity(args) if args.mode == 'parity' else bench(args)


if __name__ == '__main__':
    main()
