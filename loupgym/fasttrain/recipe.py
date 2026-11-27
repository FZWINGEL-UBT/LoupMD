"""Training recipes: the epoch loop of a mace-torch run, configurable.

mace's ``run_train`` builds the data, the statistics (E0s, average neighbours, scaling), the model, the optimizer,
the EMA and the checkpoint handler, then calls ``mace.tools.train``; after it returns, it reloads the latest
stage-one and stage-two checkpoints for the error tables and the exported models. With a recipe (``LOUP_RECIPE``,
or ``--loup_recipe`` on ``python -m loupmd.fasttrain.run``), :func:`train` replaces that loop: its own batches,
learning-rate and loss-weight schedules per optimizer step, EMA, validation cadence and stopping rules, all run on
the C++/CUDA training library. The recipe ``mace`` (the default) keeps mace's own loop.

A recipe is a JSON file (or the name of one in ``loupmd/fasttrain/recipes``)::

    {"name": "cosine-60",
     "epochs": 60,                     # total epochs (each a pass over the training set in random batches)
     "batch_size": 4,                  # null: mace's --batch_size
     "seed": 1,
     "lr": {"schedule": "cosine", "peak": 0.01, "final": 1e-4, "warmup_epochs": 2},
     "weights": [{"from_epoch": 0,  "energy": 1,    "forces": 100, "stress": 1},
                 {"from_epoch": 45, "energy": 1000, "forces": 100, "stress": 10, "ramp_epochs": 3}],
     "ema_decay": 0.99,                # null: mace's --ema_decay (needs --ema)
     "max_grad_norm": 10.0,            # null: mace's --clip_grad
     "eval_every": 1,                  # epochs between validations (and the last epoch always)
     "max_seconds": null,              # stop after this much wall time (training + validation)
     "target_rmse_f": null}            # stop once the validation force RMSE (meV/A) reaches this

Learning-rate schedules (per optimizer step; ``warmup_epochs`` ramps linearly from ``peak * warmup_start``):
``constant`` (``peak``), ``cosine`` (``peak`` -> ``final``), ``linear`` (``peak`` -> ``final``), ``exp`` (geometric
``peak`` -> ``final``), ``piecewise`` (``points``: [[epoch, lr], ...], linear in between, flat after the last),
``wsd`` (``peak`` until the last ``decay_epochs``, then to ``final`` by ``decay_shape`` ``1-sqrt`` or ``linear``).
The rate applies to every parameter group (mace's groups keep their own weight decay). Loss weights take the
last phase whose ``from_epoch`` has started, reached linearly over its ``ramp_epochs`` from the previous phase's.

``optimizer`` (all optional): ``amsgrad``, ``betas``, ``eps`` (mace's Adam groups), ``weight_decay`` (replaces the
L2 weight decay of mace's decay groups, 5e-7 by default; 0 trains better with both optimizers), ``update_clip``
(StableAdamW per-tensor update clipping), ``math`` (``fp32``, ``tf32`` or ``bf16``: the library's cuBLAS GEMMs and
radial MLPs on tensor cores with operands rounded to TF32 / bf16, FP32 accumulation and storage; ``math_until``:
the epoch from which the library runs FP32 again, e.g. TF32 up to the energy-weighted end phase) and ``kind``:
``adam`` (the default) or ``muon`` (:mod:`loupmd.fasttrain.muon`: orthogonalised Nesterov momentum on the matrix
blocks, Adam on the rest, in the library's fused step) with ``momentum`` (0.95), ``nesterov`` (true), ``scale``
(0.2: update RMS of AdamW, so Adam learning rates carry over), ``muon_lr_mult`` (1.0), ``ns_steps`` (5) and
``ns_polish`` (0; DPA4 uses 8 + 2), ``muon_wd`` (0: decoupled weight decay of the Muon blocks, p *= 1 - lr
muon_lr_mult muon_wd before each step), ``impl`` (``library``, or ``torch`` for the step-by-step reference, which
also takes ``magma`` (DPA4's Magma-lite damping)). Muon tensors take no L2 weight decay.

``sampling`` (optional): from epoch ``from_epoch`` (1) a fraction ``uniform`` (0.5) of each epoch's draws is a
uniform permutation and the rest is drawn with replacement in proportion to ``score ** power`` (1.0), a frame's score
being its loss at the step it was last drawn, relative to that epoch's mean loss; ``is_beta`` (0) scales each step's
loss weights by the importance factor ((1 / n) / p) ** is_beta (normalised to mean 1 over the epoch); ``smooth`` (0)
keeps an exponential average of a frame's relative losses instead (score = smooth * score + (1 - smooth) * loss).

Every validation is written to ``<work_dir>/results/recipe_log.jsonl`` with the wall time since the loop
started (training and validation; the set-up before the loop is logged once as ``setup_seconds``), so runs of
different recipes compare by time to accuracy. The final EMA model is saved as both the stage-one and (with
``--stage_two``) the stage-two checkpoint, so mace's tables and exports follow as usual.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

RECIPE_DIR = Path(__file__).resolve().parent / 'recipes'


@dataclass
class Recipe:
    name: str = 'custom'
    epochs: int = 150
    batch_size: Optional[int] = None
    seed: int = 1
    lr: dict = field(default_factory=lambda: {'schedule': 'constant', 'peak': 0.01})
    weights: List[dict] = field(default_factory=lambda: [{'from_epoch': 0, 'energy': 1.0, 'forces': 100.0,
                                                          'stress': 1.0}])
    ema_decay: Optional[float] = None
    max_grad_norm: Optional[float] = None
    eval_every: int = 1
    max_seconds: Optional[float] = None
    target_rmse_f: Optional[float] = None
    # Adam settings: {'amsgrad': bool, 'betas': [b1, b2], 'eps': float, 'update_clip': float (StableAdamW, 1.0)}
    optimizer: dict = field(default_factory=dict)
    # weight averaging scored at each evaluation: {'lawa': [epochs, ...], 'snap_ema': [beta, ...]}
    average: Optional[dict] = None
    # teacher per-atom energies (prototype, optimizer.impl "torch"): {'weight': w, 'xyz': file with absolute
    # REF_node_energy, 'e0': {Z: eV} the run's fitted reference energies (plan.json energy_offsets_eV)}
    node_energy: Optional[dict] = None
    # frames drawn by their last loss: {'power': a, 'uniform': u, 'from_epoch': e, 'is_beta': b} (see _draw)
    sampling: Optional[dict] = None

    @classmethod
    def load(cls, spec: str) -> 'Recipe':
        p = Path(spec)
        if not p.suffix and not p.exists():
            p = RECIPE_DIR / f'{spec}.json'
        d = json.loads(p.read_text())
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'recipe {p}: unknown keys {sorted(unknown)}')
        r = cls(**d)
        r.check()
        return r

    def check(self):
        if self.epochs < 1 or self.eval_every < 1:
            raise ValueError('epochs and eval_every must be >= 1')
        if self.lr.get('schedule') not in ('constant', 'cosine', 'linear', 'exp', 'piecewise', 'wsd'):
            raise ValueError(f'lr schedule {self.lr.get("schedule")!r}')
        if not self.weights or self.weights[0].get('from_epoch', 0) != 0:
            raise ValueError('the first loss-weight phase must start at epoch 0')
        unknown = set(self.optimizer) - {'amsgrad', 'betas', 'eps', 'update_clip', 'kind', 'momentum', 'nesterov',
                                         'scale', 'muon_lr_mult', 'impl',
                                         'ns_steps', 'ns_polish', 'magma', 'muon_wd', 'matrix', 'adam_rms', 'soap_betas',
                                       'soap_every', 'soap_dtype', 'soap_eig', 'soap_max_dim', 'soap_sweeps', 'soap_block', 'normuon_beta', 'cautious', 'zclip', 'weight_decay', 'math', 'math_mlp',
                                         'math_until'}
        if unknown:
            raise ValueError(f'optimizer: unknown keys {sorted(unknown)}')
        if self.optimizer.get('kind', 'adam') not in ('adam', 'muon'):
            raise ValueError(f'optimizer kind {self.optimizer.get("kind")!r}: adam or muon')
        if self.sampling is not None:
            unknown = set(self.sampling) - {'power', 'uniform', 'from_epoch', 'is_beta', 'smooth'}
            if unknown:
                raise ValueError(f'sampling: unknown keys {sorted(unknown)}')
            if not 0.0 <= float(self.sampling.get('uniform', 0.5)) <= 1.0:
                raise ValueError('sampling.uniform must be in [0, 1]')

    # ------------------------------------------------------------------ schedules
    def lr_at(self, t: float) -> float:
        """Learning rate at fractional epoch t (0 <= t < epochs)."""
        s = self.lr
        kind = s['schedule']
        peak = float(s.get('peak', 0.01))
        final = float(s.get('final', peak))
        warm = float(s.get('warmup_epochs', 0.0))
        if warm > 0 and t < warm:
            start = peak * float(s.get('warmup_start', 0.0))
            return start + (peak - start) * t / warm
        if kind == 'piecewise':
            pts = sorted((float(e), float(v)) for e, v in s['points'])
            if t <= pts[0][0]:
                return pts[0][1]
            for (e0, v0), (e1, v1) in zip(pts, pts[1:]):
                if t <= e1:
                    return v0 + (v1 - v0) * (t - e0) / max(e1 - e0, 1e-12)
            return pts[-1][1]
        if kind == 'wsd':
            # warm-up, stable at peak, then a cooldown over the last decay_epochs to final
            # (shape '1-sqrt' (Hagele et al. 2024) or 'linear')
            dec = float(s['decay_epochs'])
            u = min(max((t - (self.epochs - dec)) / max(dec, 1e-12), 0.0), 1.0)
            w = math.sqrt(u) if s.get('decay_shape', '1-sqrt') == '1-sqrt' else u
            return peak + (final - peak) * w
        span = max(self.epochs - warm, 1e-12)
        u = min(max((t - warm) / span, 0.0), 1.0)
        if kind == 'constant':
            return peak
        if kind == 'cosine':
            return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * u))
        if kind == 'linear':
            return peak + (final - peak) * u
        return peak * (final / peak) ** u          # exp

    def weights_at(self, t: float):
        """(energy, forces, stress) loss weights at fractional epoch t."""
        phases = sorted(self.weights, key=lambda w: w.get('from_epoch', 0))
        key = ('energy', 'forces', 'stress')
        cur = [float(phases[0][k]) for k in key]
        for prev, ph in zip(phases, phases[1:]):
            e0 = float(ph['from_epoch'])
            if t < e0:
                break
            target = [float(ph[k]) for k in key]
            before = [float(prev[k]) for k in key]
            ramp = float(ph.get('ramp_epochs', 0.0))
            a = 1.0 if ramp <= 0 else min((t - e0) / ramp, 1.0)
            cur = [b + (c - b) * a for b, c in zip(before, target)]
        return cur


def _draw(rng, n: int, bs: int, spe: int, score, smp: dict):
    """One epoch's batches when frames are drawn by their last loss: a fraction ``uniform`` of the epoch's
    ``spe * bs`` draws is a uniform permutation (without replacement), the rest is drawn with replacement with
    probability proportional to ``score ** power`` (``score``: each frame's loss at its last step, relative to its
    epoch's mean). Returns the batches and each step's importance factor ((1 / n) / p) ** is_beta, normalised to
    mean 1 over the epoch (all 1 with ``is_beta`` 0)."""
    m = spe * bs
    u = float(smp.get('uniform', 0.5))
    w = np.power(np.maximum(score, 1e-12), float(smp.get('power', 1.0)))
    p = u / n + (1.0 - u) * w / w.sum()              # the mixture's probability of each frame per draw
    k = int(round(u * m))
    idx = np.concatenate([rng.permutation(n)[:k], rng.choice(n, size=m - k, p=w / w.sum())]).astype(np.int64)
    rng.shuffle(idx)
    batches = [idx[i * bs:(i + 1) * bs].tolist() for i in range(spe)]
    beta = float(smp.get('is_beta', 0.0))
    if beta <= 0:
        return batches, [1.0] * spe
    f = np.array([np.mean(((1.0 / n) / p[b]) ** beta) for b in batches])
    return batches, (f / f.mean()).tolist()


def active_recipe() -> Optional[Recipe]:
    spec = os.environ.get('LOUP_RECIPE', 'mace')
    if spec in ('', 'mace'):
        return None
    return Recipe.load(spec)


# ---------------------------------------------------------------------- the loop
def train(model, loss_fn, train_loader, valid_loaders, optimizer, lr_scheduler, start_epoch, max_num_epochs,
          patience, checkpoint_handler, logger, eval_interval, output_args, device, log_errors, swa=None, ema=None,
          max_grad_norm=10.0, log_wandb=False, distributed=False, save_all_checkpoints=False, plotter=None,
          distributed_model=None, train_sampler=None, rank=0):
    """Drop-in for ``mace.tools.train.train`` running the active recipe (see the module docstring)."""
    from mace.tools.checkpoint import CheckpointState
    from mace.tools.train import valid_err_log
    from .cuda import CudaTrainer
    from .run import evaluate, _supported
    rc = active_recipe()
    why = _supported(model, loss_fn, optimizer, distributed)
    if why or start_epoch != 0:
        raise RuntimeError(f'recipe {rc.name}: not supported here ({why or "restart from epoch %d" % start_epoch})')
    if ema is not None and rc.ema_decay is not None:
        ema.decay = float(rc.ema_decay)
    clip = rc.max_grad_norm if rc.max_grad_norm is not None else max_grad_norm
    bs = rc.batch_size or train_loader.batch_size
    dataset = list(train_loader.dataset)
    t_setup = time.time()
    oc = rc.optimizer
    if oc.get('math'):                 # fp32 | tf32 | bf16: the library's GEMMs and radial MLPs (read at creation)
        if oc['math'] not in ('fp32', 'tf32', 'bf16'):
            raise ValueError(f'recipe {rc.name}: optimizer.math {oc["math"]!r}: fp32, tf32 or bf16')
        os.environ['LST_MATH'] = oc['math']
    if oc.get('math_mlp'):
        if oc['math_mlp'] not in ('fp32', 'tf32', 'bf16'):
            raise ValueError(f'recipe {rc.name}: optimizer.math_mlp {oc["math_mlp"]!r}: fp32, tf32 or bf16')
        os.environ['LST_MLP_TC'] = {'fp32': '0'}.get(oc['math_mlp'], oc['math_mlp'])
    tr = CudaTrainer(model, dataset, batch_size=bs, dtype=next(model.parameters()).dtype)
    for g in optimizer.param_groups:
        if 'amsgrad' in oc:
            g['amsgrad'] = bool(oc['amsgrad'])
        if 'betas' in oc:
            g['betas'] = tuple(float(b) for b in oc['betas'])
        if 'eps' in oc:
            g['eps'] = float(oc['eps'])
        if 'weight_decay' in oc and g.get('weight_decay', 0.0):   # mace's L2 groups (5e-7); 0 turns it off
            g['weight_decay'] = float(oc['weight_decay'])
    if 'amsgrad' in oc and not oc['amsgrad']:
        for st in optimizer.state.values():
            st.pop('max_exp_avg_sq', None)
    tr.update_clip = float(oc.get('update_clip', 0.0))
    mo = None
    if oc.get('kind', 'adam') == 'muon' and oc.get('impl', 'library') == 'library':
        from . import muon
        soap = oc.get('matrix') == 'soap'
        extra = sorted(k for k in ('magma', 'adam_rms', 'zclip') if oc.get(k))
        if oc.get('cautious') not in (None, '', 'adam'):
            extra.append('cautious')
        extra += [k for k in ('soap_betas', 'soap_every', 'soap_sweeps', 'soap_block') if oc.get(k) and not soap]
        if oc.get('matrix', 'ns') not in ('ns', 'normuon', 'soap'):
            extra.append('matrix')
        if extra:
            raise RuntimeError(f'recipe {rc.name}: {extra} only with optimizer.impl "torch" so far')
        tr.set_muon(muon.library_blocks(model, tr.params), momentum=oc.get('momentum', 0.95),
                    nesterov=oc.get('nesterov', True), scale=oc.get('scale', 0.2) * oc.get('muon_lr_mult', 1.0),
                    ns_steps=oc.get('ns_steps', 5), ns_polish=oc.get('ns_polish', 0),
                    normuon_beta=oc.get('normuon_beta', 0.95) if oc.get('matrix') == 'normuon' else None)
        if oc.get('cautious') == 'adam':
            tr.set_cautious(True)
        if oc.get('muon_wd'):
            tr.set_muon_wd(oc['muon_wd'], lr_mult=oc.get('muon_lr_mult', 1.0))
        if soap:
            tr.set_soap(betas=tuple(oc.get('soap_betas', (0.95, 0.95))), every=int(oc.get('soap_every', 10)),
                        max_dim=int(oc.get('soap_max_dim', 64)), lr_mult=oc.get('muon_lr_mult', 1.0),
                        sweeps=oc.get('soap_sweeps'), block=oc.get('soap_block'))
        logging.info(f'recipe {rc.name}: Muon in the library on {len(tr.muon_blocks)} blocks'
                     + {'normuon': ' (NorMuon)', 'soap': ' (SOAP)'}.get(oc.get('matrix'), ''))
    elif oc.get('kind', 'adam') == 'muon':               # muon.py: library gradients, update in torch
        from . import muon
        g0 = optimizer.param_groups[0]
        named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
        mo = muon.MuonAdam(model, named, momentum=oc.get('momentum', 0.95), nesterov=oc.get('nesterov', True),
                           scale=oc.get('scale', 0.2), muon_lr_mult=oc.get('muon_lr_mult', 1.0),
                           betas=g0['betas'], eps=g0['eps'], amsgrad=g0.get('amsgrad', False),
                           ns_steps=oc.get('ns_steps', 5), ns_polish=oc.get('ns_polish', 0),
                           magma=oc.get('magma', False), muon_wd=oc.get('muon_wd', 0.0),
                           matrix=oc.get('matrix', 'ns'), adam_rms=oc.get('adam_rms', False),
                           soap_betas=oc.get('soap_betas', (0.95, 0.95)), soap_every=oc.get('soap_every', 10),
                           soap_dtype=oc.get('soap_dtype', 'float64'), soap_eig=oc.get('soap_eig', 'eigh'),
                           soap_max_dim=oc.get('soap_max_dim', 0),
                           normuon_beta=oc.get('normuon_beta', 0.95), cautious=oc.get('cautious', ''))
        tr._prepare_optimizer(optimizer, ema, clip)      # pylint: disable=protected-access
        logging.info(f'recipe {rc.name}: {mo.describe()}')
        mo.zclip = float(oc.get('zclip', 0.0))
        if rc.node_energy:
            e0 = {int(z): float(v) for z, v in rc.node_energy.get('e0', {}).items()}
            mo.node_energy = muon.NodeEnergy(model, dataset, rc.node_energy['xyz'], float(rc.node_energy['weight']),
                                             e0=e0, recenter=bool(rc.node_energy.get('recenter', False)))
            logging.info(f'recipe {rc.name}: per-atom energy targets, weight {rc.node_energy["weight"]}')
    if rc.node_energy and mo is None:
        raise RuntimeError(f'recipe {rc.name}: node_energy needs optimizer.kind "muon" with impl "torch"')
    n = len(dataset)
    spe = n // bs                                  # drop the last partial batch, as mace's train loader does
    rng = np.random.default_rng(rc.seed)
    score = np.full(n, np.nan)                     # rc.sampling: each frame's last loss (relative)
    out_dir = Path(getattr(logger, 'directory', '.'))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'recipe_log.jsonl'
    logging.info(f'Loup-S recipe {rc.name}: {rc.epochs} epochs of {spe} steps (batch {bs}), '
                 f'set-up {time.time() - t_setup:.1f} s; log {log_path}')
    with open(log_path, 'a') as fh:
        fh.write(json.dumps({'recipe': asdict(rc), 'setup_seconds': time.time() - t_setup,
                             'frames': n, 'steps_per_epoch': spe}) + '\n')
    t0 = time.time()
    train_s = valid_s = 0.0
    step = 0
    last = None
    av = rc.average or {}
    snaps, sema = [], {float(b): None for b in av.get('snap_ema', [])}
    keep = max([int(k) for k in av.get('lawa', [])] or [0])
    tparams = [q for q in model.parameters() if q.requires_grad]
    math_until = oc.get('math_until')
    if math_until is not None and mo is not None:
        raise RuntimeError(f'recipe {rc.name}: optimizer.math_until needs the library step (impl "library")')
    if math_until is not None and not hasattr(tr.lib, 'lst_set_math'):
        raise RuntimeError(f'recipe {rc.name}: optimizer.math_until needs a libloupstrain with lst_set_math')
    for epoch in range(rc.epochs):
        if math_until is not None and epoch >= float(math_until):
            tr.set_math('fp32', 'fp32')               # FP32 from here on (the step graph is captured again)
            logging.info(f'recipe {rc.name}: FP32 arithmetic from epoch {epoch}')
            math_until = None
        smp = rc.sampling
        if smp is not None and epoch >= float(smp.get('from_epoch', 1)) and not np.isnan(score).any():
            batches, fac = _draw(rng, n, bs, spe, score, smp)
        else:
            perm = rng.permutation(n)
            batches = [perm[i * bs:(i + 1) * bs].tolist() for i in range(spe)]
            fac = None
        ts = [epoch + i / spe for i in range(spe)]
        lr_steps = [[rc.lr_at(t)] * len(optimizer.param_groups) for t in ts]
        w_steps = [rc.weights_at(t) for t in ts]
        if fac is not None:
            w_steps = [[w * f for w in ws] for ws, f in zip(w_steps, fac)]
        t1 = time.time()
        if mo is not None:
            losses = muon.run_epoch(tr, mo, batches, lr_steps, w_steps, ema, clip)
        else:
            losses, _ = tr.run_epoch(batches, loss_fn, optimizer, ema, clip, lr_steps=lr_steps,
                                     weight_steps=w_steps)
        train_s += time.time() - t1
        if rc.sampling is not None:                   # each frame's last loss, relative to the epoch's mean
            la = np.asarray(losses, dtype=np.float64)
            if fac is not None:
                la = la / np.asarray(fac)
            la = la / max(float(la.mean()), 1e-300)
            sm = float(rc.sampling.get('smooth', 0.0))
            for b, v in zip(batches, la):
                old = score[b]
                score[b] = np.where(np.isnan(old), v, sm * old + (1.0 - sm) * v)
        gn = None
        if mo is not None and mo.norms:                 # pre-clip gradient norms of the epoch: p10, median, max
            q = np.percentile(mo.norms, [10, 50, 100])
            gn = [float(x) for x in q]
            mo.norms.clear()
        step += spe
        for g in optimizer.param_groups:            # what mace's checkpoint and logs then show
            g['lr'] = lr_steps[-1][0]
        for k, w in zip(('energy_weight', 'forces_weight', 'stress_weight'), w_steps[-1]):
            getattr(loss_fn, k).fill_(w)              # validation loss with the current weights
        if rank == 0:
            logger.log({'loss': float(np.mean(losses)), 'time': (time.time() - t1) / spe, 'mode': 'opt',
                        'epoch': epoch})
        if av:
            cur = [q.detach().clone() for q in tparams]
            if keep:
                snaps = (snaps + [cur])[-keep:]
            for b in sema:
                sema[b] = cur if sema[b] is None else [b * s + (1 - b) * c for s, c in zip(sema[b], cur)]
        final = epoch == rc.epochs - 1
        if epoch % rc.eval_every == 0 or final:
            t2 = time.time()
            ctx = ema.average_parameters() if ema is not None else nullcontext()
            res = {}
            with ctx:
                for name, vl in valid_loaders.items():
                    vloss, metrics = evaluate(model=model, loss_fn=loss_fn, data_loader=vl, output_args=output_args,
                                              device=device)
                    valid_err_log(vloss, metrics, logger, log_errors, epoch, name)
                    res[name] = {'loss': float(vloss), 'rmse_e_per_atom': float(metrics['rmse_e_per_atom']) * 1e3,
                                 'rmse_f': float(metrics['rmse_f']) * 1e3,
                                 'rmse_stress': float(metrics['rmse_stress']) * 1e3
                                 if metrics.get('rmse_stress') is not None else None}
            if av:
                live = [q.detach().clone() for q in tparams]
                cands = [(f'lawa{k}', [sum(s[i] for s in snaps[-int(k):]) / len(snaps[-int(k):])
                                       for i in range(len(tparams))]) for k in av.get('lawa', [])]
                cands += [(f'sema{b}', sema[b]) for b in sema]
                for tag, ws in cands:
                    with torch.no_grad():
                        for q, w in zip(tparams, ws):
                            q.copy_(w)
                    for name, vl in valid_loaders.items():
                        _, mt = evaluate(model=model, loss_fn=loss_fn, data_loader=vl, output_args=output_args,
                                         device=device)
                        res[f'{tag}:{name}'] = {'rmse_e_per_atom': float(mt['rmse_e_per_atom']) * 1e3,
                                                'rmse_f': float(mt['rmse_f']) * 1e3}
                with torch.no_grad():
                    for q, w in zip(tparams, live):
                        q.copy_(w)
            valid_s += time.time() - t2
            last = {k: v for k, v in res.items() if ':' not in k}
            wall = time.time() - t0
            with open(log_path, 'a') as fh:
                fh.write(json.dumps({'epoch': epoch, 'step': step, 'wall': wall, 'train_s': train_s,
                                     'valid_s': valid_s, 'lr': lr_steps[-1][0], 'weights': w_steps[-1],
                                     'valid': res, 'gnorm': gn}) + '\n')
            rmse_f = min(v['rmse_f'] for k, v in res.items() if ':' not in k)
            if rc.target_rmse_f is not None and rmse_f <= rc.target_rmse_f:
                logging.info(f'recipe {rc.name}: validation force RMSE {rmse_f:.2f} meV/A reached the target '
                             f'{rc.target_rmse_f} at epoch {epoch}, {wall:.1f} s')
                break
            if rc.max_seconds is not None and wall >= rc.max_seconds:
                logging.info(f'recipe {rc.name}: time budget {rc.max_seconds} s spent at epoch {epoch}')
                break
    logging.info(f'recipe {rc.name}: {time.time() - t0:.1f} s ({train_s:.1f} s steps, {valid_s:.1f} s validation)')
    # the final EMA model as the stage-one checkpoint and, with stage two, as the stage-two one
    ctx = ema.average_parameters() if ema is not None else nullcontext()
    with ctx:
        if swa is not None:
            checkpoint_handler.save(state=CheckpointState(model, optimizer, lr_scheduler),
                                    epochs=max(swa.start - 1, 0), keep_last=True)
            checkpoint_handler.save(state=CheckpointState(model, optimizer, lr_scheduler),
                                    epochs=max(swa.start, epoch), keep_last=True)
        else:
            checkpoint_handler.save(state=CheckpointState(model, optimizer, lr_scheduler), epochs=epoch,
                                    keep_last=True)
    return last
