"""Throughput of the training engine's optimizer step on the frozen frames (local GPU, FP32).

Times full steps (packing, energy, forces/stress, loss, parameter backward, clip, Adam/AMSGrad)
over shuffled batches of the training split, and optionally writes a torch profiler table.
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
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--steps', type=int, default=75)
    ap.add_argument('--conv', default='reference')
    ap.add_argument('--basis', default='e3nn', choices=('e3nn', 'fused'))
    ap.add_argument('--product', default='grouped', choices=('grouped', 'poly'))
    ap.add_argument('--tangent', action='store_true')
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--engine', default='fast', choices=('fast', 'mace'))
    args = ap.parse_args()
    from fasttrain_parity import load_items
    from loupgym.fasttrain.model import Student, loss_terms
    from loupgym.fasttrain.data import frames_from_dataset, DeviceFrames
    dtype, dev = torch.float32, torch.device('cuda')
    model = torch.load(args.model, map_location='cpu').to(dtype)
    t0 = time.time()
    items = load_items(args.xyz, model, dtype, args.frames)
    t1 = time.time()
    model = model.to(dev)
    conv = None
    if args.conv == 'triton':
        from loupgym.fasttrain.ops import conv_triton
        conv = conv_triton
    basis = None
    if args.basis == 'fused':
        from loupgym.fasttrain.ops import edge_basis as basis
    student = Student(model, conv=conv, basis=basis, product=args.product, tangent=args.tangent)
    frames = frames_from_dataset(items, model)
    dfr = DeviceFrames(frames, dtype, dev)
    t2 = time.time()
    print(f'items {t1 - t0:.1f}s frames+device {t2 - t1:.1f}s')
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3, amsgrad=True)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(frames))
    nb = len(frames) // args.batch

    def step(i, mace_batches=None):
        ids = order[(i % nb) * args.batch:(i % nb + 1) * args.batch]
        opt.zero_grad(set_to_none=True)
        b = dfr.batch(ids, csr=(args.conv != 'reference'))
        e, f, s = student.evaluate(b, training=True)
        le, lf, ls = loss_terms(b, e, f, s)
        loss = le + 100 * lf + ls
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        return loss

    for i in range(5):
        step(i)
    torch.cuda.synchronize()
    t = time.time()
    for i in range(args.steps):
        loss = step(i)
    torch.cuda.synchronize()
    dt = time.time() - t
    print(f'{args.steps} steps: {dt:.3f}s, {1e3 * dt / args.steps:.2f} ms/step, loss {float(loss):.4g}, '
          f'peak alloc {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB')
    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(10):
                step(i)
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=40))


if __name__ == '__main__':
    main()
