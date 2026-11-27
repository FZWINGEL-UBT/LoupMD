"""Parity of the training engine against stock mace-torch on real frames.

Compares energy, forces, stress and the parameter gradient of the stock stress loss (create_graph
path) between ``mace`` and ``loupgym.fasttrain`` for one packed batch. Run on CPU in float64 or on a
GPU in float32 (``--device cuda --dtype float32``) under the local GPU reservation.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch


def load_items(xyz, model, dtype, count):
    from mace import data
    from mace.data import KeySpecification
    from mace.tools import AtomicNumberTable
    torch.set_default_dtype(dtype)
    ks = KeySpecification(info_keys={'energy': 'REF_energy', 'stress': 'REF_stress', 'virials': 'REF_virials', 'dipole': 'dipole', 'head': 'head'},
                          arrays_keys={'forces': 'REF_forces', 'charges': 'REF_charges'})
    _, configs = data.load_from_xyz(xyz, ks, head_name='Default')
    z = AtomicNumberTable([int(x) for x in model.atomic_numbers])
    items = [data.AtomicData.from_config(c, z_table=z, cutoff=float(model.r_max), heads=['Default'])
             for c in configs[:count]]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--xyz', required=True)
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--dtype', default='float64')
    ap.add_argument('--conv', default='reference', choices=('reference', 'triton'))
    ap.add_argument('--basis', default='e3nn', choices=('e3nn', 'fused'))
    ap.add_argument('--product', default='grouped', choices=('grouped', 'poly'))
    ap.add_argument('--tangent', action='store_true')
    ap.add_argument('--weights', default='1,100,1')
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    dev = torch.device(args.device)
    from mace.tools import torch_geometric
    from mace.modules.loss import WeightedEnergyForcesStressLoss
    from loupgym.fasttrain.model import Student, loss_terms
    from loupgym.fasttrain.data import frames_from_dataset, DeviceFrames

    model = torch.load(args.model, map_location='cpu')
    model = model.to(dtype)
    items = load_items(args.xyz, model, dtype, args.frames)
    model = model.to(dev)
    we, wf, ws = (float(x) for x in args.weights.split(','))
    loss_fn = WeightedEnergyForcesStressLoss(we, wf, ws).to(dev)

    batch = next(iter(torch_geometric.dataloader.DataLoader(items, batch_size=len(items), shuffle=False))).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    out = model(batch.to_dict(), training=True, compute_force=True, compute_stress=True)
    loss = loss_fn(pred=out, ref=batch)
    g_ref = torch.autograd.grad(loss, params)

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
    b = dfr.batch(list(range(len(items))), csr=True)
    e, f, s = student.evaluate(b, training=True)
    le, lf, ls = loss_terms(b, e, f, s)
    loss2 = we * le + wf * lf + ws * ls
    g_new = torch.autograd.grad(loss2, params)

    # permute reference forces into the engine node order
    perm = []
    for g, fr in enumerate(frames):
        pass
    f_ref_engine = b.f_ref
    def rel(a, b_):
        return float((a - b_).norm() / b_.norm().clamp_min(1e-300))
    print(f'dtype {dtype} device {dev} frames {len(items)} edges {b.sender.numel()}')
    print('energy  mace', out['energy'].tolist())
    print('energy  fast', e.tolist())
    print('energy max abs diff', float((out['energy'] - e).abs().max()))
    # forces: compare via loss-consistent sorted order
    fm = out['forces']
    # map mace node order -> engine order through the reference labels (unique rows)
    order = torch.cat([torch.sort(it.node_attrs.argmax(1), stable=True).indices + off
                       for it, off in zip(items, np.cumsum([0] + [len(it.positions) for it in items])[:-1])])
    # engine order is species-major across graphs; rebuild by matching reference forces rows
    ref_rows = {tuple(np.round(r, 12)): i for i, r in enumerate(batch.forces.detach().cpu().double().numpy())}
    idx = torch.tensor([ref_rows[tuple(np.round(r, 12))] for r in b.f_ref.detach().cpu().double().numpy()], device=dev)
    print('forces rel diff', rel(f, fm[idx]), 'max abs', float((f - fm[idx]).abs().max()))
    print('stress rel diff', rel(s, out['stress']), 'max abs', float((s - out['stress']).abs().max()))
    print('loss mace', float(loss), 'fast', float(loss2))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    worst = 0.
    for n, a, c in zip(names, g_ref, g_new):
        r = rel(c, a)
        worst = max(worst, r)
        print(f'  grad {n:70s} rel {r:.3e}  |g| {float(a.norm()):.3e}')
    print('worst grad rel diff', worst)


if __name__ == '__main__':
    main()
