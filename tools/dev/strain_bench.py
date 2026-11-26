"""Epoch throughput of the training engines on a frozen training set (local GPU).

    python tools/dev/strain_bench.py --model M --xyz X [--engine cuda|triton] [--batch 4] [--epochs 6]

Runs whole epochs (shuffled batches, mace-like AMSGrad with weight decay, clip 10, EMA 0.99) and prints the
seconds of each epoch's optimizer steps; the first epoch includes the graph capture.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--xyz', required=True)
    ap.add_argument('--frames', type=int, default=100000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--engine', default='cuda', choices=('cuda', 'triton'))
    ap.add_argument('--dtype', default='float32')
    ap.add_argument('--nsys', action='store_true', help='cudaProfilerStart/Stop around the last epoch')
    ap.add_argument('--muon', action='store_true', help='Muon on the matrix blocks as the fast recipes (8 + 2 NS)')
    ap.add_argument('--soap', type=int, default=0, help='with --muon: SOAP refresh interval (max dim 64)')
    args = ap.parse_args()
    from fasttrain_parity import load_items
    from strain_parity import make_opt
    from mace.modules.loss import WeightedEnergyForcesStressLoss
    dt = getattr(torch, args.dtype)
    model = torch.load(args.model, map_location='cpu').to(dt)
    items = load_items(args.xyz, model, dt, args.frames)
    model = model.to('cuda')
    if args.engine == 'cuda':
        from loupmd.fasttrain.cuda import CudaTrainer
        tr = CudaTrainer(model, items, batch_size=args.batch, dtype=dt)
    else:
        from loupmd.fasttrain.engine import GraphTrainer
        tr = GraphTrainer(model, items, batch_size=args.batch, dtype=dt)
    d = tr.data
    print(f'{args.engine} {args.dtype}: {len(items)} frames, batch {args.batch}: N {d.N_pad} E {d.E_pad} '
          f'P {getattr(d, "P_pad", 0)}, setup {tr.setup_seconds:.1f} s')
    opt, ema = make_opt(model, lr=0.01)
    if args.muon and args.engine == 'cuda':
        from loupmd.fasttrain import muon
        tr.set_muon(muon.library_blocks(model, tr.params), 0.95, True, 0.2, 8, 2)
        if args.soap:
            tr.set_soap(every=args.soap, max_dim=64)
    loss_fn = WeightedEnergyForcesStressLoss(energy_weight=1000.0, forces_weight=100.0, stress_weight=10.0)
    rng = np.random.default_rng(0)
    n = len(items)
    times = []
    for ep in range(args.epochs):
        perm = rng.permutation(n)
        batches = [perm[i:i + args.batch].tolist() for i in range(0, n, args.batch)]
        if args.nsys and ep == args.epochs - 1:
            torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.synchronize()
        t = time.time()
        losses, _ = tr.run_epoch(batches, loss_fn, opt, ema, 10.0)
        torch.cuda.synchronize()
        times.append(time.time() - t)
        if args.nsys and ep == args.epochs - 1:
            torch.cuda.cudart().cudaProfilerStop()
        print(f'epoch {ep}: {times[-1]:.3f} s ({1e3 * times[-1] / len(batches):.2f} ms/step), '
              f'mean loss {np.mean(losses):.4g}' + (' host %.3f s, gpu %.3f s' % tr.last_times if hasattr(tr, 'last_times') else ''))
    steady = times[1:] if len(times) > 1 else times
    print(f'steady: {np.median(steady):.3f} s/epoch, {1e3 * np.median(steady) / len(batches):.2f} ms/step; '
          f'peak alloc {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')


if __name__ == '__main__':
    main()
