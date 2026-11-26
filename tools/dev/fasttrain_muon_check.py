"""The training library's fused Muon step (lst_set_muon, ABI 5) against the torch reference of
loupmd.fasttrain.muon (the library's gradient pass per step, the Muon / Adam update and the EMA in torch).

FP64 should agree to ~1e-14 relative after a dozen steps, FP32 to ~1e-6. --time also times the fused step with
and without Muon.

Usage (local GPU reservation): python tools/dev/fasttrain_muon_check.py --model M --xyz X [--steps 12] [--time]
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

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--xyz', required=True)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--epochs', type=int, default=1, help='epochs of --steps steps, varying lr')
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--time', action='store_true')
    ap.add_argument('--ns-steps', type=int, default=5)
    ap.add_argument('--ns-polish', type=int, default=0)
    ap.add_argument('--normuon', type=float, default=None, help='NorMuon beta (library vs muon.py matrix normuon)')
    ap.add_argument('--soap', type=int, default=0, help='SOAP refresh interval (library vs muon.py matrix soap, '
                    'Jacobi refresh, max dim --soap-max-dim)')
    ap.add_argument('--soap-max-dim', type=int, default=64)
    ap.add_argument('--soap-ref', default='', help='eigh: side A is torch SOAP with eigh bases instead of the '
                    'library (torch vs torch: the drift between two correct solvers)')
    ap.add_argument('--soap-eps', type=float, default=1e-8, help='SOAP eps (larger: damps near-null eigen-directions)')
    ap.add_argument('--lib-ab', default='', help='VAR=VAL[,VAR=VAL]: side B is the library again with these '
                    'environment switches (read at engine creation / set_soap): an exact switch matches to ~1e-14')
    ap.add_argument('--cautious', action='store_true', help='cautious Adam (library vs muon.py cautious adam)')
    ap.add_argument('--weight-decay', type=float, default=5e-7, help='L2 of the Adam groups (mace: 5e-7)')
    ap.add_argument('--muon-wd', type=float, default=0.0, help='decoupled weight decay of the Muon blocks '
                    '(library lst_set_muon_wd vs muon.py muon_wd)')
    a = ap.parse_args()
    from torch_ema import ExponentialMovingAverage
    from mace.modules.loss import WeightedEnergyForcesStressLoss
    from fasttrain_parity import load_items
    from loupmd.fasttrain import muon
    from loupmd.fasttrain.cuda import CudaTrainer

    def setup(dtype, items):
        m = torch.load(a.model, map_location='cpu', weights_only=False).to(dtype).cuda()
        opt = torch.optim.Adam(m.parameters(), lr=a.lr, amsgrad=True, weight_decay=a.weight_decay)
        return m, opt, CudaTrainer(m, items, batch_size=4, dtype=dtype), ExponentialMovingAverage(m.parameters(),
                                                                                                    decay=0.99)

    def rel(xs, ys):
        return max(float((x - y).abs().max() / x.abs().max().clamp_min(1e-30)) for x, y in zip(xs, ys))

    ok = True
    for dtype, tol in ((torch.float64, 1e-12), (torch.float32, 2e-5)):
        items = load_items(a.xyz, torch.load(a.model, map_location='cpu', weights_only=False).to(dtype), dtype, 100000)
        loss_fn = WeightedEnergyForcesStressLoss(1.0, 100.0, 1.0).to(dtype).cuda()
        n = a.steps
        batches = [[(4 * i + j) % len(items) for j in range(4)] for i in range(n)]
        G = 0
        w_steps = [[1.0, 100.0, 1.0]] * n
        mA, oA, trA, eA = setup(dtype, items)
        if not a.soap_ref:
            trA.set_muon(muon.library_blocks(mA, trA.params), 0.95, True, 0.2, a.ns_steps, a.ns_polish,
                         normuon_beta=a.normuon)
            if a.soap:
                trA.set_soap(every=a.soap, max_dim=a.soap_max_dim, eps=a.soap_eps)
            if a.cautious:
                trA.set_cautious(True)
            if a.muon_wd:
                trA.set_muon_wd(a.muon_wd)
        G = len(oA.param_groups)
        sched = [[[a.lr * (1.0 + 0.5 * ((e * n + i) % 7) / 7.0)] * G for i in range(n)] for e in range(a.epochs)]
        la = []
        if a.soap_ref:                                     # torch vs torch (bases from soap_eig=a.soap_ref)
            moA = muon.MuonAdam(mA, list(mA.named_parameters()), betas=(0.9, 0.999), eps=1e-8, amsgrad=True,
                                ns_steps=a.ns_steps, ns_polish=a.ns_polish, matrix='soap', soap_every=a.soap,
                                soap_eig=a.soap_ref, soap_max_dim=a.soap_max_dim, soap_eps=a.soap_eps,
                                soap_dtype=str(dtype).split('.')[-1])
            for g in moA.adam.param_groups:
                g['weight_decay'] = a.weight_decay
            trA._prepare_optimizer(oA, eA, 1.0)            # pylint: disable=protected-access
            for e in range(a.epochs):
                la += muon.run_epoch(trA, moA, batches, sched[e], w_steps, eA, 1.0)
        else:
            for e in range(a.epochs):
                la += trA.run_epoch(batches, loss_fn, oA, eA, 1.0, lr_steps=sched[e], weight_steps=w_steps)[0]
        if a.lib_ab:
            kv = dict(x.split('=', 1) for x in a.lib_ab.split(','))
            old = {k: os.environ.get(k) for k in kv}
            os.environ.update(kv)
            mB, oB, trB, eB = setup(dtype, items)
            trB.set_muon(muon.library_blocks(mB, trB.params), 0.95, True, 0.2, a.ns_steps, a.ns_polish,
                         normuon_beta=a.normuon)
            if a.soap:
                trB.set_soap(every=a.soap, max_dim=a.soap_max_dim, eps=a.soap_eps)
            if a.cautious:
                trB.set_cautious(True)
            lb = []
            for e in range(a.epochs):
                lb += trB.run_epoch(batches, loss_fn, oB, eB, 1.0, lr_steps=sched[e], weight_steps=w_steps)[0]
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            wp = rel(list(mA.parameters()), list(mB.parameters()))
            we = rel(eA.shadow_params, eB.shadow_params)
            wl = max(abs(x - y) / max(abs(y), 1e-30) for x, y in zip(la, lb))
            good = max(wp, we) < tol
            ok &= good
            print(f'{dtype}: library vs library with {a.lib_ab}, {a.epochs} x {n} steps: worst relative difference '
                  f'parameters {wp:.2e}, EMA {we:.2e}, losses {wl:.2e} ({"ok" if good else "FAIL"}, tolerance {tol:g})')
            continue
        mB, oB, trB, eB = setup(dtype, items)
        moB = muon.MuonAdam(mB, list(mB.named_parameters()), betas=(0.9, 0.999), eps=1e-8, amsgrad=True,
                            ns_steps=a.ns_steps, ns_polish=a.ns_polish, cautious='adam' if a.cautious else '',
                            muon_wd=a.muon_wd,
                            **({} if a.normuon is None else dict(matrix='normuon', normuon_beta=a.normuon)),
                            **({} if not a.soap else dict(matrix='soap', soap_every=a.soap, soap_eig='jacobi',
                                                          soap_max_dim=a.soap_max_dim, soap_eps=a.soap_eps,
                                                          soap_dtype=str(dtype).split('.')[-1])))
        for g in moB.adam.param_groups:
            g['weight_decay'] = a.weight_decay
        trB._prepare_optimizer(oB, eB, 1.0)                # pylint: disable=protected-access
        lb = []
        for e in range(a.epochs):
            lb += muon.run_epoch(trB, moB, batches, sched[e], w_steps, eB, 1.0)
        wp = rel(list(mA.parameters()), list(mB.parameters()))
        we = rel(eA.shadow_params, eB.shadow_params)
        wl = max(abs(x - y) / abs(y) for x, y in zip(la, lb))
        good = max(wp, we) < tol
        ok &= good
        print(f'{dtype}: {len(trA.muon_blocks) if not a.soap_ref else "torch-ref"} blocks, {a.epochs} x {n} steps: worst relative difference parameters {wp:.2e}, '
              f'EMA {we:.2e}, losses {wl:.2e} ({"ok" if good else "FAIL"}, tolerance {tol:g})')
    if a.time:
        items = load_items(a.xyz, torch.load(a.model, map_location='cpu', weights_only=False).float(),
                           torch.float32, 100000)
        loss_fn = WeightedEnergyForcesStressLoss(1.0, 100.0, 1.0).cuda()
        bt = [[(4 * i + j) % len(items) for j in range(4)] for i in range(75)]
        for use in (False, True):
            m, o, tr, e = setup(torch.float32, items)
            if use:
                tr.set_muon(muon.library_blocks(m, tr.params), 0.95, True, 0.2, a.ns_steps, a.ns_polish)
            kw = dict(lr_steps=[[0.01]] * 75, weight_steps=[[1.0, 100.0, 1.0]] * 75)
            for _ in range(2):
                tr.run_epoch(bt, loss_fn, o, e, 1.0, **kw)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(4):
                tr.run_epoch(bt, loss_fn, o, e, 1.0, **kw)
            torch.cuda.synchronize()
            print(f'fused step {"Muon" if use else "Adam"}: {(time.time() - t0) / 300 * 1e3:.2f} ms/step')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
