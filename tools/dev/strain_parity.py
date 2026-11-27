"""Stage-by-stage parity of the training library (libloupstrain.so) with tools/dev/strain_reference.py.

FP64: every stage and every raw parameter gradient should agree to ~1e-12 relative. FP32 (--dtype float32):
the library in FP32 against the FP64 reference (expect ~1e-6). With --steps n it also runs n optimizer steps
through the CUDA graph and through mace's own torch step (autograd, clip, Adam / AMSGrad, EMA) and compares
the parameters.

Usage (local GPU reservation): python tools/dev/strain_parity.py --model M --xyz X [--dtype float32] [--steps 3]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

STAGES = ['Y', 'x0', 'A0', 'B0', 'f0', 'x1', 'sc', 'A1', 'B1', 'f1', 'f2', 'u', 'E', 'G0', 'G1', 'x1b', 'f0b', 'f1b',
          'f2b', 's_e', 'gy', 'g', 'F', 'S', 'sb', 'gyb', 'G0s', 'x0s', 'x1s', 'A0s', 'A1s']


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--xyz', required=True)
    ap.add_argument('--stage', type=int, default=2)
    ap.add_argument('--dtype', default='float64')
    ap.add_argument('--steps', type=int, default=0)
    args = ap.parse_args()
    from fasttrain_parity import load_items
    from strain_reference import Poly, step
    from loupgym.fasttrain import _product_poly as pp
    from loupgym.fasttrain.cuda import CudaTrainer
    from loupgym.fasttrain.engine import PaddedFrames
    from loupgym.fasttrain.data import frames_from_dataset
    from loupgym.fasttrain.model import silu_norm_constant
    dev = torch.device('cuda')
    dt = getattr(torch, args.dtype)
    model64 = torch.load(args.model, map_location='cpu').to(torch.float64)
    items = load_items(args.xyz, model64, torch.float64, 7)
    model64 = model64.to(dev)
    lw = (1.0, 100.0, 1.0) if args.stage == 1 else (1000.0, 100.0, 10.0)
    batches = [[5, 1, 3, 0], [2, 6, 4]]
    # reference (FP64)
    frames = frames_from_dataset(items, model64)
    pf = PaddedFrames(frames, torch.float64, dev, batch_size=5, r_max=float(model64.r_max))
    bt = pf.gather(pf.device_batch(pf.host_batch(batches[0])))
    act = silu_norm_constant()
    poly = Poly(pp.TERMS, pp.K, dev, torch.float64)
    loss, _, st = step(model64, bt, poly, act, lw)
    raw = [p for p in model64.parameters() if p.requires_grad]
    from loupgym.fasttrain.engine import FastStudent
    fs = FastStudent(model64)
    e, f, s = fs.evaluate(bt, training=True)
    le, lf, ls = fs.loss_terms(bt, e, f, s)
    g_ref = torch.autograd.grad(lw[0] * le + lw[1] * lf + lw[2] * ls, raw)
    # library
    model = copy.deepcopy(model64).to(dt)
    items_l = items if dt == torch.float64 else load_items(args.xyz, model, dt, 7)
    tr = CudaTrainer(model, items_l, batch_size=5, dtype=dt)
    assert tr.data.N_pad == pf.N_pad and tr.data.E_pad == pf.E_pad
    tr.debug_step(batches, lw, which=0)
    torch.cuda.synchronize()
    print(f'{args.dtype}: padded N {tr.data.N_pad} E {tr.data.E_pad} P {tr.data.P_pad}')
    print(f'loss reference {float(loss):.15g}')
    worst = 0.0
    for k in STAGES:
        a = tr.stage(k)
        b = st[k]
        r = rel(a, b.reshape(-1)[:a.numel()]) if a.numel() == b.numel() else float('nan')
        if a.numel() != b.numel():
            print(f'  stage {k:5s} size {a.numel()} vs {b.numel()}')
            continue
        worst = max(worst, r)
        print(f'  stage {k:5s} {r:.2e}')
    rt = tr.stage('RT0').view(2, -1)
    print(f'  stage R0    {rel(rt[0], st["R0"]):.2e}   T0 {rel(rt[1], st["T0"]):.2e}')
    rb = tr.stage('RTb0').view(2, -1)
    print(f'  stage R0b   {rel(rb[0], st["R0b"]):.2e}   T0b {rel(rb[1], st["T0b"]):.2e}')
    rb = tr.stage('RTb1').view(2, -1)
    print(f'  stage R1b   {rel(rb[0], st["R1b"]):.2e}   T1b {rel(rb[1], st["T1b"]):.2e}')
    gw = 0.0
    for (n, _), a, b in zip([(n, p) for n, p in model64.named_parameters() if p.requires_grad], tr.raw_grads(), g_ref):
        r = rel(a, b)
        gw = max(gw, r)
        print(f'  grad {n:62s} {r:.2e}')
    print(f'worst stage {worst:.2e}  worst grad {gw:.2e}')
    if args.steps:
        steps_parity(args, model64, items, items_l, dt)


def make_opt(model, lr=float(os.environ.get("PARITY_LR", "0.01"))):
    """mace-like Adam: AMSGrad, decay on interactions and products, none elsewhere; EMA 0.99."""
    from torch_ema import ExponentialMovingAverage
    dec, nodec = [], []
    for n, p in model.named_parameters():
        (dec if ('linear.weight' in n and 'interactions' in n) or 'products' in n else nodec).append(p)
    opt = torch.optim.Adam([dict(params=nodec, weight_decay=0.0), dict(params=dec, weight_decay=5e-7)], lr=lr,
                           amsgrad=True)
    ema = ExponentialMovingAverage(list(model.parameters()), decay=0.99)
    return opt, ema


def steps_parity(args, model64, items, items_l, dt):
    from mace.modules.loss import WeightedEnergyForcesStressLoss
    from loupgym.fasttrain.cuda import CudaTrainer
    from loupgym.fasttrain.engine import GraphTrainer
    lw = (1.0, 100.0, 1.0) if args.stage == 1 else (1000.0, 100.0, 10.0)
    loss_fn = WeightedEnergyForcesStressLoss(energy_weight=lw[0], forces_weight=lw[1], stress_weight=lw[2])
    epochs = [[[5, 1, 3, 0], [2, 6, 4]], [[0, 1, 2, 3, 4], [6, 5]]][:max(1, args.steps // 2)]
    ma = copy.deepcopy(model64)
    mb = copy.deepcopy(model64).to(dt)
    oa, ea = make_opt(ma)
    ob, eb = make_opt(mb)
    ta = GraphTrainer(ma, items, batch_size=5, dtype=torch.float64, use_graph=False)
    tb = CudaTrainer(mb, items_l, batch_size=5, dtype=dt)
    for ep, bs in enumerate(epochs):
        la, _ = ta.run_epoch(bs, loss_fn, oa, ea, 10.0)
        lb, _ = tb.run_epoch(bs, loss_fn, ob, eb, 10.0)
        print(f'epoch {ep}: losses torch {["%.10g" % x for x in la]}  library {["%.10g" % x for x in lb]}')
    worst = {}
    for (n, pa), pb in zip(ma.named_parameters(), mb.parameters()):
        sa, sb = oa.state[pa], ob.state[pb]
        for k, a, b in (('param', pa, pb), ('exp_avg', sa['exp_avg'], sb['exp_avg']),
                        ('exp_avg_sq', sa['exp_avg_sq'], sb['exp_avg_sq']),
                        ('max_exp_avg_sq', sa['max_exp_avg_sq'], sb['max_exp_avg_sq'])):
            worst[k] = max(worst.get(k, 0.0), rel(b, a))
        if float(sa['step']) != float(sb['step']):
            print(f'  step count {n}: {float(sa["step"])} vs {float(sb["step"])}')
    for a, b in zip(ea.shadow_params, eb.shadow_params):
        worst['ema'] = max(worst.get('ema', 0.0), rel(b, a))
    print('after steps, worst rel diff: ' + '  '.join(f'{k} {v:.2e}' for k, v in worst.items()))
    print(f'ema num_updates torch {ea.num_updates} library {eb.num_updates}')


if __name__ == '__main__':
    main()
