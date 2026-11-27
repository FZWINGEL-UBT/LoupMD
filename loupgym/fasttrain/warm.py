"""Load CUDA's kernel libraries for the frame conversion while mace is still busy on the host.

Under lazy module loading every PyTorch kernel library is loaded at its first launch (``cuLibraryLoadData``):
the recipe's frame conversion (:func:`loupmd.fasttrain.data.frames_from_dataset`, :class:`PaddedFrames`) touches
about 25 of them at once, ~1.4 s on a 5060 Ti under WSL 2. A thread started before mace's ``run_train`` creates the
CUDA context and runs the same conversion on two synthetic three-atom frames while mace reads its data and builds
the model; PyTorch releases the GIL inside its operations, so the loads overlap that work.

The thread draws no random numbers, sets no default dtype and works on the legacy default stream; :func:`join`
waits for it and every path that captures a CUDA graph (the trainers, the evaluators) calls it first, since work on
the legacy stream would invalidate a capture in progress. ``LOUP_FT_WARM=0`` turns it off.
"""
from __future__ import annotations

import logging
import threading
import time

_THREAD = None
_STATE = {}


def _stub_inputs():
    import torch
    from types import SimpleNamespace as NS
    from mace.modules.radial import ZBLBasis
    f32 = torch.float32
    dev = torch.device('cuda')
    model = NS(
        pair_repulsion_fn=ZBLBasis(p=5), atomic_numbers=torch.tensor([8, 22, 38], device=dev),
        scale_shift=NS(scale=torch.tensor(1.0, device=dev), shift=torch.tensor(0.0, device=dev)),
        atomic_energies_fn=NS(atomic_energies=torch.zeros(3, device=dev)),
        parameters=lambda: iter([torch.zeros(1, device=dev)]))
    items = []
    for k in range(2):
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.7, 0.1, 0.0], [0.2, 1.9, 0.3 + k]], dtype=f32)
        snd, rcv = zip(*[(i, j) for i in range(3) for j in range(3) if i != j])
        items.append(NS(
            node_attrs=torch.eye(3, dtype=f32)[[2, 0, 1]], edge_index=torch.tensor([snd, rcv]),
            positions=pos, shifts=torch.zeros(6, 3, dtype=f32), cell=10.0 * torch.eye(3, dtype=f32),
            weight=torch.tensor(1.0), energy=torch.tensor(-1.0), forces=torch.zeros(3, 3, dtype=f32),
            stress=torch.zeros(1, 3, 3, dtype=f32), energy_weight=torch.tensor(1.0),
            forces_weight=torch.tensor(1.0), stress_weight=torch.tensor(1.0)))
    return model, items


def _preload_indexing():
    """Tensor indexing (``x[i]``, ``x[i] = v``, boolean masks) keeps the GIL through its kernel's first launch, so
    the libraries it needs would load with mace's thread waiting: load them first through calls that release the
    GIL (one library per source file holds every dtype)."""
    import torch
    dev = torch.device('cuda')
    i = torch.arange(2, device=dev)
    for dt in (torch.float32, torch.float64, torch.int64):
        x = torch.zeros(4, 3, dtype=dt, device=dev)
        torch.index_select(x, 0, i)                          # Indexing.cu
        x.index_put_((i,), torch.ones(2, 3, dtype=dt, device=dev))      # IndexKernel.cu (index, index_put)
        x.index_put_((i, i), torch.ones(2, dtype=dt, device=dev))
        torch.gather(x, 0, torch.zeros(1, 3, dtype=torch.int64, device=dev))
    m = torch.tensor([True, False, True], device=dev)
    torch.nonzero(m)
    torch.zeros(3, device=dev).masked_fill_(m, 1.0)
    torch.zeros(3, dtype=torch.bool, device=dev).index_put_((i,), torch.tensor(True, device=dev))
    torch.cuda.synchronize()


def _preload_gemm():
    """cuBLAS loads its kernel libraries at the first GEMM of each family: float32 GEMMs of a few shapes and
    transposes through torch's handle (the training library links the same ``libcublas.so.13``, which the loader
    shares by its soname), so the trainer's first CUDA graph capture finds them loaded. ``LOUP_FT_WARM_GEMM=0``
    leaves them to that capture."""
    import torch
    dev = torch.device('cuda')
    for m, n, k in ((32, 32, 32), (128, 64, 96), (320, 128, 64), (2048, 64, 64), (64, 64, 4096)):
        a = torch.ones(m, k, device=dev)
        b = torch.ones(k, n, device=dev)
        a @ b
        a.t().contiguous().t() @ b
        a @ b.t().contiguous().t()
        b.t() @ a.t()
    torch.cuda.synchronize()


def _preload_edge_basis(argv):
    """The evaluators' Triton edge-basis kernel (large: ~150 ms to load from Triton's cache at the first
    validation): launched once on three edges in the run's default dtype with mace's cutoff-polynomial order and
    Bessel count (the same constexprs, 16-byte aligned pointers and an edge count neither 1 nor a multiple of 16,
    as the evaluators' batches have), so the first validation finds it loaded. Off by default
    (``LOUP_FT_WARM_TRITON=1`` on): it lengthens the thread past the trainer's set-up, which then waits for it about
    as long as the first validation saves (ledger ft-e2e-warm, round 94)."""
    import torch
    from .ops import edge_basis_raw
    dtype = torch.float32 if _arg(argv, '--default_dtype', 'float64') == 'float32' else torch.float64
    p = int(_arg(argv, '--num_cutoff_basis', '5'))
    nb = int(_arg(argv, '--num_radial_basis', '8'))
    r_max = float(_arg(argv, '--r_max', '5.0'))
    dev = torch.device('cuda')
    vec = torch.tensor([[1.0, 0.2, 0.1], [0.3, 1.5, 0.2], [0.1, 0.4, 2.0]], dtype=dtype, device=dev)
    edge_basis_raw(vec, torch.ones(nb, dtype=dtype, device=dev), 1.0, r_max, p)
    torch.cuda.synchronize()


def _arg(argv, name, default=None):
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + '='):
            return a.split('=', 1)[1]
    return default


def _run(ready: threading.Event, argv=()):
    t0 = time.perf_counter()
    try:
        import torch
        torch.zeros(1, device='cuda')                       # the context, while mace's modules import
        _STATE['context_s'] = time.perf_counter() - t0
        ready.wait(60)
        _preload_indexing()
        _STATE['preload_s'] = time.perf_counter() - t0
        from .data import _convert
        from .engine import PaddedFrames
        model, items = _stub_inputs()
        frames = _convert(items, model)
        for dt in (torch.float32, torch.float64):
            PaddedFrames(frames, dt, torch.device('cuda'), 1, 5.0)
        torch.cuda.synchronize()
        import os
        if os.environ.get('LOUP_FT_WARM_GEMM', '1') != '0':
            _preload_gemm()
            _STATE['gemm_s'] = time.perf_counter() - t0
        if os.environ.get('LOUP_FT_WARM_TRITON', '0') == '1':
            _preload_edge_basis(argv)
            _STATE['triton_s'] = time.perf_counter() - t0
        _STATE['ok'] = True
    except Exception as exc:  # pylint: disable=broad-except
        _STATE['error'] = f'{type(exc).__name__}: {exc}'
    _STATE['seconds'] = time.perf_counter() - t0


def start(argv) -> threading.Event | None:
    """Start the warm-up thread for a CUDA run of ``mace run_train`` arguments ``argv``; returns the event that
    lets it go past the context creation (set it once mace's modules are imported), or None."""
    global _THREAD
    import os
    if (_THREAD is not None or os.environ.get('LOUP_FT_WARM', '1') == '0'
            or _arg(argv, '--device', 'cpu') != 'cuda' or '--distributed' in argv):
        return None
    ready = threading.Event()
    _STATE['ready'] = ready
    _THREAD = threading.Thread(target=_run, args=(ready, list(argv)), name='loup-cuda-warm', daemon=True)
    _THREAD.start()
    return ready


def join():
    """Wait for the warm-up thread (a no-op without one); call before anything captures a CUDA graph."""
    global _THREAD
    th = _THREAD
    if th is None:
        return
    t = time.perf_counter()
    _STATE['ready'].set()                                   # past the context at once if mace never got there
    th.join()
    _THREAD = None
    waited = time.perf_counter() - t
    if 'error' in _STATE:
        logging.debug(f'CUDA warm-up failed after {_STATE.get("seconds", 0):.2f} s: {_STATE["error"]}')
    else:
        logging.debug(f'CUDA warm-up: {_STATE.get("seconds", 0):.2f} s (context {_STATE.get("context_s", 0):.2f} s), '
                      f'waited {waited:.2f} s for it')
