"""mace-torch 0.3.16 training with the Loup-S engine's optimizer steps.

    python -m loupmd.fasttrain.run <mace run_train arguments>

Runs mace's own ``run_train`` (data, model, optimizer, ReduceLROnPlateau, SWA stage two, EMA, checkpoints,
evaluation and logs stay mace's) and replaces only ``mace.tools.train.train_one_epoch``: each epoch's
batches are drawn from mace's own loader (the same sampler and generator draws, so the same batches) and
run as CUDA-graph-captured steps of :class:`loupmd.fasttrain.engine.GraphTrainer` on the same parameter,
optimizer-state and EMA tensors. Anything the engine does not cover (LBFGS, distributed, schedule-free,
other losses or architectures) falls back to mace's loop, which is logged.

The steps run in the C++/CUDA training library (``libloupstrain.so``, :class:`loupmd.fasttrain.cuda.CudaTrainer`)
when it loads and supports the model, else in the Triton engine; ``LOUP_FASTTRAIN_ENGINE=triton`` forces the latter.
``LOUP_RECIPE=<name or path>`` (or ``--loup_recipe``) replaces mace's epoch loop by a configurable recipe
(:mod:`loupmd.fasttrain.recipe`); ``mace`` (the default) keeps mace's loop. A recipe run skips mace's training
plots unless the arguments say ``--plot True``. mace's optional cuEquivariance / OpenEquivariance imports are skipped unless
``--enable_cueq`` / ``--only_cueq`` / ``--enable_oeq`` (or ``LOUP_FT_EQ_LIBS=1``). ``LOUP_FT_CORE_DEV=cpu`` computes
the frames' ZBL core terms on the host (saves the float64 kernels' first use on small sets). A CUDA run starts a
thread that loads the kernel libraries of the frame conversion and cuBLAS's while mace reads its data
(:mod:`loupmd.fasttrain.warm`; ``LOUP_FT_WARM=0`` off, ``LOUP_FT_WARM_GEMM=0`` without the GEMMs); mace's
``AtomicData.from_config`` and the frame conversion run on ``LOUP_FT_DATA_THREADS`` torch threads (default 1, 0 leaves
torch's count; the results are the same bits).
``LOUP_FASTTRAIN=0`` runs stock mace; ``LOUP_FASTTRAIN_EVAL=0`` keeps mace's evaluation during training, ``LOUP_FASTTRAIN_TABLES=0`` for the final error tables.
"""
from __future__ import annotations

import logging
import os
import sys
import time

import torch

_TRAINERS = {}
_EVALUATORS = {}
_STOCK = {}


def _supported(model, loss_fn, optimizer, distributed, check_loss=True) -> str:
    from mace.modules.loss import WeightedEnergyForcesStressLoss
    if distributed:
        return 'distributed training'
    if check_loss and type(loss_fn) is not WeightedEnergyForcesStressLoss:
        return f'loss {type(loss_fn).__name__}'
    if type(optimizer) is not torch.optim.Adam:
        return f'optimizer {type(optimizer).__name__}'
    try:
        from .model import check_supported
        check_supported(model)
    except Exception as exc:  # pylint: disable=broad-except
        return f'model: {exc}'
    return ''


class _TableLoader:
    """The frames a data loader draws, in larger batches, for mace's final error tables (their RMSE / MAE columns
    are over all frames, whatever the batching): mace's training loader (batch 4) cost a batch's set-up per 4
    frames there."""

    def __init__(self, data_loader, batch_size):
        self.dataset = data_loader.dataset
        self.collate_fn = data_loader.collate_fn
        self.num_workers = 0
        self.batch_size = batch_size
        ids = sorted(i for b in epoch_batches(data_loader) for i in b)    # the frames mace's loop would see
        self.batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __len__(self):
        return len(self.batches)

    def __iter__(self):                      # mace's stock evaluate (the fallback) iterates the loader
        for b in self.batches:
            yield self.collate_fn([self.dataset[i] for i in b])


def evaluate_table(model, loss_fn, data_loader, output_args, device):
    """``tables_utils.evaluate``: :func:`evaluate`, in batches of ``LOUP_FT_TABLE_BATCH`` (16) frames where no
    evaluator of the loader's own batching exists yet (the validation loader keeps its warm one)."""
    bs = int(os.environ.get('LOUP_FT_TABLE_BATCH', '16'))
    if (bs > data_loader.batch_size and data_loader.num_workers == 0
            and (id(data_loader.dataset), data_loader.batch_size) not in _EVALUATORS):
        data_loader = _TableLoader(data_loader, bs)
    return evaluate(model, loss_fn, data_loader, output_args, device)


def epoch_batches(data_loader):
    """The index batches mace's ``for batch in data_loader`` would draw, with the same generator draws."""
    if isinstance(data_loader, _TableLoader):
        return data_loader.batches
    assert data_loader.num_workers == 0
    it = iter(data_loader)
    out = []
    while True:
        try:
            out.append([int(i) for i in it._next_index()])  # pylint: disable=protected-access
        except StopIteration:
            return out


def _trainer(model, data_loader):
    from .engine import GraphTrainer
    key = (id(model), id(data_loader.dataset))
    tr = _TRAINERS.get(key)
    if tr is None:
        dtype = next(model.parameters()).dtype
        dataset = list(data_loader.dataset)
        engine = os.environ.get('LOUP_FASTTRAIN_ENGINE', 'cuda')
        if engine not in ('cuda', 'triton'):
            raise ValueError(f'LOUP_FASTTRAIN_ENGINE={engine!r}: cuda or triton')
        if engine == 'cuda':
            from .cuda import CudaTrainer
            try:
                tr = CudaTrainer(model, dataset, batch_size=data_loader.batch_size, dtype=dtype)
                logging.info('Loup-S engine: C++/CUDA training library')
            except (OSError, RuntimeError) as exc:
                logging.warning(f'Loup-S engine: training library not used ({exc}): Triton steps')
        if tr is None:
            tr = GraphTrainer(model, dataset, batch_size=data_loader.batch_size, dtype=dtype)
        logging.info(f'Loup-S engine: {len(tr.data.frames)} training frames, padded batch of '
                     f'{tr.data.N_pad} nodes / {tr.data.E_pad} edges'
                     f'{" (edge pairs)" if tr.data.paired else ""}, set up in {tr.setup_seconds:.1f} s')
        _TRAINERS[key] = tr
    return tr


def train_one_epoch(model, loss_fn, data_loader, optimizer, epoch, output_args, max_grad_norm, ema, logger,
                    device, distributed, distributed_model=None, rank=0):
    why = _supported(model, loss_fn, optimizer, distributed)
    if not why and data_loader.num_workers != 0:
        why = 'loader worker processes'
    if why:
        if not _STOCK.get('warned'):
            logging.warning(f'Loup-S engine not used ({why}): mace training loop')
            _STOCK['warned'] = True
        return _STOCK['train_one_epoch'](model=model, loss_fn=loss_fn, data_loader=data_loader, optimizer=optimizer,
                                         epoch=epoch, output_args=output_args, max_grad_norm=max_grad_norm,
                                         ema=ema, logger=logger, device=device, distributed=distributed,
                                         distributed_model=distributed_model, rank=rank)
    tr = _trainer(model, data_loader)
    batches = epoch_batches(data_loader)
    t0 = time.time()
    losses, _ = tr.run_epoch(batches, loss_fn, optimizer, ema, max_grad_norm)
    step = (time.time() - t0) / max(1, len(batches))
    if rank == 0:
        for loss in losses:
            logger.log({'loss': loss, 'time': step, 'mode': 'opt', 'epoch': epoch,
                        'lr': optimizer.param_groups[0]['lr']})


def _evaluator(model, data_loader):
    """The evaluator of a loader's set: the training library's (``CudaEvaluator``) when the library trains
    (``LOUP_FT_CUDA_EVAL=0``: the Triton ``Evaluator``), kept per set and batch size and rebound to a new model
    with the same constants."""
    from .engine import Evaluator
    from .data import _model_constants
    key = (id(data_loader.dataset), data_loader.batch_size)
    hit = _EVALUATORS.get(key)
    if hit is not None:
        ev, consts = hit
        if ev.model is model:
            return ev
        if (ev.dtype == next(model.parameters()).dtype and ev.device == next(model.parameters()).device
                and float(ev.model.r_max) == float(model.r_max) and consts == _model_constants(model)):
            try:
                ev.rebind(model)                         # same frames and batches, the new model's weights
                return ev
            except Exception:  # pylint: disable=broad-except
                pass
    ev = None
    if (os.environ.get('LOUP_FASTTRAIN_ENGINE', 'cuda') == 'cuda'
            and os.environ.get('LOUP_FT_CUDA_EVAL', '1') != '0'):
        from .cuda import CudaEvaluator
        try:
            ev = CudaEvaluator(model, list(data_loader.dataset), data_loader.batch_size)
        except (OSError, RuntimeError, ValueError) as exc:
            if not _STOCK.get('warned_cuda_eval'):
                logging.warning(f'Loup-S engine: training library not used for evaluation ({exc}): Triton')
                _STOCK['warned_cuda_eval'] = True
    if ev is None:
        ev = Evaluator(model, list(data_loader.dataset), data_loader.batch_size)
    _EVALUATORS[key] = (ev, _model_constants(model))
    return ev


class _SameModel:
    """Results of one data loader kept with a copy of the parameters they came from: mace's post-training
    tables and plots evaluate the stage-one and stage-two checkpoints, which a recipe saves from one model."""

    def __init__(self):
        self.store = {}

    def get(self, key, model):
        hit = self.store.get(key)
        if hit is None:
            return None
        params, value = hit
        cur = list(model.parameters())
        if len(cur) != len(params) or not all(torch.equal(a, b) for a, b in zip(cur, params)):
            return None
        return value

    def put(self, key, model, value):
        self.store[key] = ([p.detach().clone() for p in model.parameters()], value)


_MEMO = _SameModel()


def evaluate(model, loss_fn, data_loader, output_args, device):
    """mace's ``evaluate`` with the engine's energies, forces and stress (mace's metrics on them)."""
    from mace.tools.train import MACELoss
    why = _supported(model, loss_fn, torch.optim.Adam([torch.zeros(1)]), False)
    if why or data_loader.num_workers != 0 or os.environ.get('LOUP_FASTTRAIN_EVAL', '1') == '0':
        return _STOCK['evaluate'](model=model, loss_fn=loss_fn, data_loader=data_loader, output_args=output_args,
                                  device=device)
    start = time.time()
    memo_key = ('evaluate', id(model), id(data_loader.dataset), data_loader.batch_size, id(loss_fn),
                tuple(float(getattr(loss_fn, k)) for k in ('energy_weight', 'forces_weight', 'stress_weight')))
    hit = _MEMO.get(memo_key, model)
    if hit is not None:
        avg_loss, aux = hit
        aux = dict(aux, time=time.time() - start)
        return avg_loss, aux
    ev = _evaluator(model, data_loader)
    metrics = MACELoss(loss_fn=loss_fn).to(device)
    for ids, batch, output in ev.run(epoch_batches(data_loader), data_loader, light=True):
        # already on the device (Evaluator keeps them); update, not forward: mace's loop discards the per-batch
        # values forward computes, and the accumulated state is the same
        metrics.update(batch, output)
    avg_loss, aux = metrics.compute()
    aux['time'] = time.time() - start
    metrics.reset()
    _MEMO.put(memo_key, model, (avg_loss, dict(aux)))
    return avg_loss, aux


def model_inference(all_data_loaders, model, output_args, device, distributed=False):
    """``mace.cli.visualise_train.model_inference`` (the training plots' scatter data) on the engine."""
    from mace.cli import visualise_train as vt
    stock = _STOCK['model_inference']
    loaders = list(all_data_loaders.values())
    why = '' if not distributed else 'distributed'
    if not why:
        why = _supported(model, None, torch.optim.Adam([torch.zeros(1)]), False, check_loss=False)
    if why or any(dl.num_workers != 0 for dl in loaders) or os.environ.get('LOUP_FASTTRAIN_EVAL', '1') == '0':
        return stock(all_data_loaders, model, output_args, device, distributed)
    e0 = model.atomic_energies_fn.atomic_energies.reshape(-1)
    results = {}
    if True:                         # the engine's force pass differentiates internally
        for name, data_loader in all_data_loaders.items():
            key = ('inference', id(model), id(data_loader.dataset), data_loader.batch_size)
            hit = _MEMO.get(key, model)
            if hit is None:
                metric = vt.InferenceMetric().to(device)
                for ids, batch, output in _evaluator(model, data_loader).run(epoch_batches(data_loader),
                                                                             data_loader):
                    # stock ScaleShiftMACE: interaction energy = total less the graph's isolated-atom energies
                    e0_graph = torch.zeros_like(output['energy']).index_add_(
                        0, batch.batch, (batch.node_attrs.to(e0.dtype) @ e0).to(output['energy'].dtype))
                    metric(batch, dict(output, interaction_energy=output['energy'] - e0_graph))
                hit = metric.compute()
                metric.reset()
                _MEMO.put(key, model, hit)
            results[name] = hit
    return results


def avg_num_neighbors(data_loader) -> float:
    """``mace.modules.compute_avg_num_neighbors`` from the dataset's edge lists, without collating batches: the
    mean over atoms with at least one neighbour of their receiver counts (the same integer sum, divided in the
    default dtype as torch.mean does)."""
    import numpy as np
    total = count = 0
    bs = data_loader.batch_size or 1
    if (type(getattr(data_loader, 'sampler', None)).__name__ == 'DistributedSampler'
            or (data_loader.drop_last and len(data_loader.dataset) % bs)):
        return _STOCK['avg_num_neighbors'](data_loader)     # a shard, or a random dropped batch: mace's own
    try:
        items = data_loader.dataset
        for item in items:
            c = np.bincount(item.edge_index[1].cpu().numpy())
            c = c[c > 0]
            total += int(c.sum())
            count += int(c.size)
    except Exception:  # pylint: disable=broad-except
        return _STOCK['avg_num_neighbors'](data_loader)
    if count == 0 or total >= 2 ** 24:          # float32 sums above 2^24 round: keep mace's own order
        return _STOCK['avg_num_neighbors'](data_loader)
    dt = torch.get_default_dtype()
    return float((torch.tensor(float(total), dtype=dt) / count).item())


def mean_rms_energy_forces(data_loader, atomic_energies):
    """``mace.modules.compute_mean_rms_energy_forces`` (``--scaling rms_forces_scaling``) without collating whole
    batches: the batches the loader would draw (the same generator draws), each frame's ``node_attrs``,
    ``energy``, ``forces`` and ``head`` concatenated as the collate does, and mace's own operations on them, so
    the mean and RMS are bit for bit mace's."""
    stock = _STOCK['rms_forces_scaling']
    gen = getattr(data_loader, 'generator', None)
    if (data_loader.num_workers != 0 or gen is None
            or type(getattr(data_loader, 'sampler', None)).__name__ == 'DistributedSampler'):
        return stock(data_loader, atomic_energies)
    state = gen.get_state()
    try:
        from mace.modules.blocks import AtomicEnergiesBlock
        from mace.modules.utils import _check_non_zero
        from mace.tools import to_numpy
        from mace.tools.scatter import scatter_mean, scatter_sum
        fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)
        ds = data_loader.dataset
        e_list, f_list, h_list, hb_list = [], [], [], []
        for ids in epoch_batches(data_loader):
            items = [ds[i] for i in ids]
            sizes = [int(it.node_attrs.shape[0]) for it in items]
            node_attrs = torch.cat([it.node_attrs for it in items], dim=0)
            bvec = torch.repeat_interleave(torch.arange(len(items)), torch.tensor(sizes))
            head = torch.stack([it.head.reshape(()) for it in items])
            energy = torch.stack([it.energy.reshape(()) for it in items])
            ng = len(items)
            graph_e0s = scatter_sum(src=fn(node_attrs), index=bvec, dim=0, dim_size=ng)[torch.arange(ng), head]
            graph_sizes = torch.tensor(sizes, dtype=torch.long)
            e_list.append((energy - graph_e0s) / graph_sizes)
            f_list.append(torch.cat([it.forces for it in items], dim=0))
            h_list.append(head)
            hb_list.append(head[bvec])
        atom_energies = torch.cat(e_list, dim=0)
        forces = torch.cat(f_list, dim=0)
        head = torch.cat(h_list, dim=0)
        head_batch = torch.cat(hb_list, dim=0)
        mean = to_numpy(scatter_mean(src=atom_energies, index=head, dim=0).squeeze(-1))
        rms = to_numpy(torch.sqrt(scatter_mean(src=torch.square(forces), index=head_batch, dim=0).mean(-1)))
        return mean, _check_non_zero(rms)
    except Exception:  # pylint: disable=broad-except
        gen.set_state(state)                    # mace's own pass, from the same generator state
        return stock(data_loader, atomic_energies)


def _cg_cached(stock):
    """``mace.tools.cg.U_matrix_real`` (the symmetric contractions' Clebsch-Gordan matrices, ~0.4 s of each model
    build) kept in ``<cache root>/train/cg``: keyed by every argument (irreps as their strings), the default dtype,
    mace's CG switches and the mace / e3nn / torch versions; stored as the label strings and the tensors (torch.save,
    loaded with weights_only), so a hit returns mace's own result. Off with ``LOUP_FT_CG_CACHE=0``."""
    import hashlib
    import inspect
    import importlib.metadata as md
    sig = inspect.signature(stock)
    cg = sys.modules[stock.__module__]

    def path_for(args):
        from e3nn import o3
        norm = {}
        for k, v in args.items():
            if k in ('irreps_in', 'irreps_out'):
                v = str(o3.Irreps(v))
            elif k == 'filter_ir_mid' and v is not None:
                v = [str(o3.Irrep(ir)) for ir in v]
            norm[k] = repr(v)
        key = repr((sorted(norm.items()), str(torch.get_default_dtype()), getattr(cg, 'CUET_AVAILABLE', None),
                    getattr(cg, 'USE_CUEQ_CG', None), md.version('mace-torch'), md.version('e3nn'),
                    torch.__version__))
        from loupmd.cache import cache_dir
        d = cache_dir('train') / 'cg'
        d.mkdir(parents=True, exist_ok=True)
        return d / (hashlib.sha256(key.encode()).hexdigest()[:32] + '.pt')

    def U_matrix_real(*a, **k):  # pylint: disable=invalid-name
        try:
            bound = sig.bind(*a, **k)
            bound.apply_defaults()
            path = path_for(dict(bound.arguments))
        except Exception:  # pylint: disable=broad-except
            return stock(*a, **k)
        from e3nn import o3
        if path.is_file():
            try:
                rec = torch.load(path, map_location='cpu', weights_only=True)
                out = []
                for kind, label, t in zip(rec['kinds'], rec['labels'], rec['tensors']):
                    out += [o3.Irrep(label) if kind == 'irrep' else label, t]
                return out
            except Exception:  # pylint: disable=broad-except
                pass                                    # unreadable: make it again
        out = stock(*a, **k)
        try:
            kinds, labels, tensors = [], [], []
            if len(out) % 2:
                return out
            for label, t in zip(out[0::2], out[1::2]):
                if isinstance(label, o3.Irrep):
                    kinds.append('irrep')
                    labels.append(str(label))
                elif type(label) is str:
                    kinds.append('str')
                    labels.append(label)
                else:
                    return out
                if not isinstance(t, torch.Tensor):
                    return out
                tensors.append(t)
            tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp')
            torch.save({'kinds': kinds, 'labels': labels, 'tensors': tensors}, tmp)
            os.replace(tmp, path)
        except Exception:  # pylint: disable=broad-except
            pass
        return out

    U_matrix_real.__wrapped__ = stock
    return U_matrix_real


def _from_config_threads(stock):
    """``mace.data.AtomicData.from_config`` under :func:`loupmd.fasttrain.data.host_threads` (one intra-op
    thread: 0.85 instead of 1.16 s for the 390 frames of a soap30 run on the 5060 Ti PC, every field bit for bit
    equal; it also keeps mace's data loading from stalling beside :mod:`loupmd.fasttrain.warm`)."""
    from .data import host_threads
    fn = stock.__func__

    def from_config(cls, *a, **k):
        with host_threads():
            return fn(cls, *a, **k)

    from_config.__wrapped__ = fn
    return classmethod(from_config)


def install():
    import importlib
    mm = importlib.import_module('mace.modules')
    if os.environ.get('LOUP_FT_DATA_THREADS', '1') != '0':
        ad = importlib.import_module('mace.data.atomic_data').AtomicData
        if 'from_config' not in _STOCK:
            _STOCK['from_config'] = ad.__dict__['from_config']
            ad.from_config = _from_config_threads(_STOCK['from_config'])
    if os.environ.get('LOUP_FT_CG_CACHE', '1') != '0':
        cg = importlib.import_module('mace.tools.cg')
        stock = _STOCK.setdefault('U_matrix_real', cg.U_matrix_real)
        cached = _cg_cached(stock)
        # the name is bound at import in the modules that use it
        for name in ('mace.tools.cg', 'mace.tools', 'mace.modules.symmetric_contraction'):
            mod = sys.modules.get(name)
            if mod is not None and getattr(mod, 'U_matrix_real', None) is stock:
                mod.U_matrix_real = cached
    if 'avg_num_neighbors' not in _STOCK:
        _STOCK['avg_num_neighbors'] = mm.compute_avg_num_neighbors
    if os.environ.get('LOUP_FT_FAST_AVGNB', '1') != '0':
        mm.compute_avg_num_neighbors = avg_num_neighbors
    _STOCK.setdefault('rms_forces_scaling', mm.scaling_classes['rms_forces_scaling'])
    if os.environ.get('LOUP_FT_FAST_STATS', '1') != '0':
        mm.scaling_classes['rms_forces_scaling'] = mean_rms_energy_forces
    mt = importlib.import_module('mace.tools.train')     # mace.tools.train is also a function name there
    if 'train_one_epoch' not in _STOCK:
        _STOCK['train_one_epoch'] = mt.train_one_epoch
        _STOCK['evaluate'] = mt.evaluate
    mt.train_one_epoch = train_one_epoch
    mt.evaluate = evaluate
    if os.environ.get('LOUP_FASTTRAIN_TABLES', '1') != '0':
        # the final error tables (train / valid / test) as well
        importlib.import_module('mace.tools.tables_utils').evaluate = evaluate_table
        # and the training plots' scatter data (stock inference over train / valid / test per stage)
        vt = importlib.import_module('mace.cli.visualise_train')
        _STOCK.setdefault('model_inference', vt.model_inference)
        vt.model_inference = model_inference
    from .recipe import active_recipe, train as recipe_train
    rc = active_recipe()
    if rc is not None:
        # run_train calls ``tools.train`` (the package attribute) for its epoch loop
        tools = importlib.import_module('mace.tools')
        _STOCK.setdefault('train', tools.train)
        tools.train = recipe_train
        logging.info(f'Loup-S recipe {rc.name} replaces mace\'s training loop')
        if os.environ.get('LOUP_FT_JIT_MEMO', '1') != '0':
            # a recipe saves its stage-one and stage-two checkpoints from one model: mace's second TorchScript
            # compile (the _stagetwo_compiled file, ~0.4 s) gets the first one's result when nothing differs
            ej = importlib.import_module('e3nn.util.jit')
            _STOCK.setdefault('jit_compile', ej.compile)
            ej.compile = _jit_compile_memo(_STOCK['jit_compile'])


def _jit_compile_memo(stock):
    """e3nn.util.jit.compile, returning the previous result for a module of the same type, repr and state."""
    last = {}

    def same(sd, old):
        return sd.keys() == old.keys() and all(
            a.dtype == b.dtype and a.device == b.device and a.shape == b.shape and torch.equal(a, b)
            for a, b in ((sd[k], old[k]) for k in sd))

    def compile(mod, *args, **kwargs):          # pylint: disable=redefined-builtin
        if args or kwargs or not isinstance(mod, torch.nn.Module):
            return stock(mod, *args, **kwargs)
        sd, rep = mod.state_dict(), repr(mod)
        hit = last.get(type(mod))
        if hit is not None and hit[1] == rep and same(sd, hit[0]):
            return hit[2]
        out = stock(mod)
        last[type(mod)] = ({k: v.detach().clone() for k, v in sd.items()}, rep, out)
        return out

    return compile


def _lazy_dynamo():
    """mace's ``tools.compile`` imports ``torch._dynamo`` at import time (about 0.5 s here) for its torch.compile path,
    which training never takes: a stand-in module imports the real one at its first attribute access, so a run that
    uses it pays the same then. Its ``disable`` (what torch's own ``_disable_dynamo`` wrappers, e.g. every
    ``torch.optim`` optimizer's constructor, ask for) runs the function as it is while the real module is not
    loaded (nothing can be compiling then) and hands over to the real ``disable`` at the first call after it is.
    ``LOUP_DEBUG_LAZY_DYNAMO=1`` logs the stack of the real module's first use."""
    import functools
    import importlib
    import types
    if 'torch._dynamo' in sys.modules:
        return

    class _LazyDynamo(types.ModuleType):
        def disable(self, fn=None, recursive=True, **kw):
            if fn is None:
                return functools.partial(self.disable, recursive=recursive, **kw)

            @functools.wraps(fn)
            def call(*args, **kwargs):
                real = sys.modules.get('torch._dynamo')
                if real is None or real is self:
                    return fn(*args, **kwargs)
                d = real.disable(fn, recursive, **kw)
                try:
                    fn.__dynamo_disable = d        # torch._compile's cache of the disabled function
                except (AttributeError, TypeError):
                    pass
                return d(*args, **kwargs)
            return call

        def __getattr__(self, name):
            if name.startswith('__') and name not in ('__path__', '__all__') and sys.modules.get('torch._dynamo') is self:
                # inspect.getmodule asks every module in sys.modules for __file__ (torch.library does so when an op
                # is registered): no reason to load the package; submodule imports need __path__, * needs __all__
                raise AttributeError(name)
            if sys.modules.get('torch._dynamo') is self:
                del sys.modules['torch._dynamo']
                if os.environ.get('LOUP_DEBUG_LAZY_DYNAMO', '0') == '1':
                    import traceback
                    logging.warning('torch._dynamo loaded on first use (%s):\n%s', name,
                                    ''.join(traceback.format_stack(limit=12)))
                torch._dynamo = importlib.import_module('torch._dynamo')
            return getattr(sys.modules['torch._dynamo'], name)

    sys.modules['torch._dynamo'] = _LazyDynamo('torch._dynamo')


def main(argv=None):
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--loup_recipe' in argv:                      # ours, not mace's: name or path of a recipe (recipe.py)
        i = argv.index('--loup_recipe')
        os.environ['LOUP_RECIPE'] = argv[i + 1]
        del argv[i:i + 2]
    if os.environ.get('LOUP_RECIPE', 'mace') != 'mace' and not any(
            a == '--plot' or a.startswith('--plot=') for a in argv):
        # a recipe run skips mace's training plots (two 300-dpi figures, ~7 s); --plot True keeps them
        argv += ['--plot', 'False']
    def flag(name):
        for i, a in enumerate(argv):
            v = argv[i + 1] if a == name and i + 1 < len(argv) else a.split('=', 1)[1] if a.startswith(name + '=') else None
            if v is not None:
                return v.strip().lower() in ('true', '1', 'yes')
        return False
    if os.environ.get('LOUP_FT_EQ_LIBS', '0') != '1' and not any(
            flag(f) for f in ('--enable_cueq', '--only_cueq', '--enable_oeq')):
        # mace imports these optional backends (2-3 s, OpenEquivariance checks its JIT build) only to offer them
        for m in ('cuequivariance', 'cuequivariance_torch', 'cuequivariance_ops_torch', 'openequivariance'):
            sys.modules.setdefault(m, None)
    if os.environ.get('LOUP_FT_LAZY_DYNAMO', '1') != '0':
        _lazy_dynamo()
    ready = None
    path = os.environ.get('PATH', '')
    keep = [p for p in path.split(os.pathsep) if not (len(p) > 5 and p.startswith('/mnt/') and p[6:7] in ('/', ''))]
    if len(keep) < len(path.split(os.pathsep)):
        # under WSL the Windows drives on PATH make torchmetrics' import-time shutil.which('latex') take ~0.4 s;
        # drop them while mace is imported (install() imports it first)
        os.environ['PATH'] = os.pathsep.join(keep)
    try:
        if os.environ.get('LOUP_FASTTRAIN', '1') != '0':
            from . import warm
            ready = warm.start(argv)   # CUDA's kernel libraries for the frame conversion, beside mace's host work
            install()
        from mace.cli import run_train
    finally:
        os.environ['PATH'] = path
        if ready is not None:
            ready.set()
    sys.argv = [sys.argv[0]] + argv
    run_train.main()


if __name__ == '__main__':
    # run from the package module: the recipe's ``from .run import evaluate`` must see the stock functions and
    # evaluator caches that install() set, not a second copy of this file (``python -m`` runs it as __main__)
    from loupgym.fasttrain import run as _package_run
    _package_run.main()
    if os.environ.get('LOUP_FT_FAST_EXIT', '1') != '0':
        # the run's files are written and closed; skip the interpreter's teardown (module and CUDA state, ~0.6 s):
        # flush and close the logging handlers and the standard streams, then leave
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
