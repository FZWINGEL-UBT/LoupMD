"""Stage 0 SrTiO3 census. Run only inside the approved Ada reservation (Linux).

    python tools/dev/student_census.py sweep student.loup --sha256 DIGEST --output census-s64l0
    python tools/dev/student_census.py parse capture.sqlite --steps 16 --output kernel-sums.json

`sweep` needs an exported .loup, the SM89 LoupMD runtime, ASE, nvidia-ml-py (pynvml), and nsys.
No checkpoint deserialisation or export. Each size gets two fresh processes. Only `time` yields
step times or enters the memory slope; `trace` warms up without capture before profiling nodes.
`parse` is CPU-only. Its input must be one isolated cudaProfilerApi range, not a whole MD run.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from pathlib import Path

SIZES = {320: 4, 5000: 10, 40000: 20}
SAMPLE_INTERVAL_S = 0.05
WARMUP_STEPS = 1000  # 1 ps at 1 fs: ten Langevin damping times, independent of cell size


def parse_trace(path, steps):
    """Group by full demangled name and launch shape; union all streams of this process/device."""
    if steps < 1:
        raise ValueError('need a positive number of captured MD steps')
    sql = '''SELECT k.start, k.end, k.deviceId, k.globalPid, s.value, k.gridX, k.gridY, k.gridZ,
                    k.blockX, k.blockY, k.blockZ, k.staticSharedMemory, k.dynamicSharedMemory
             FROM CUPTI_ACTIVITY_KIND_KERNEL AS k LEFT JOIN StringIds AS s ON k.demangledName=s.id
             ORDER BY k.start'''
    groups = {}
    first = right = domain = None
    launches = kernel_sum = union = 0
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        for start, end, device, pid, name, *shape in db.execute(sql):
            if not name or end < start:
                raise ValueError('invalid kernel interval or unresolved kernel name')
            if first is None:
                first = right = start
                domain = (device, pid)
            if (device, pid) != domain:
                raise ValueError('expected kernels from a single process and device')
            duration = end - start
            count_ns = groups.setdefault((name, *shape), [0, 0])
            count_ns[0] += 1
            count_ns[1] += duration
            launches += 1
            kernel_sum += duration
            union += max(0, end - max(right, start))
            right = max(right, end)
    if first is None or right == first:
        raise ValueError('no kernel activity; check capture range and --cuda-graph-trace=node')
    span = right - first
    kernels = [dict(name=k[0], grid=k[1:4], block=k[4:7], static_shared_bytes=k[7], dynamic_shared_bytes=k[8],
                    launches=count, launches_per_step=count / steps, sum_ns=ns, sum_ms_per_step=ns / (1e6 * steps))
               for k, (count, ns) in sorted(groups.items(), key=lambda item: -item[1][1])]
    return dict(profiled_steps=steps, launches=launches, launches_per_step=launches / steps, kernel_sum_ns=kernel_sum,
                kernel_union_ns=union, overlap_ns=kernel_sum - union, kernel_span_ns=span,
                kernel_sum_ms_per_step=kernel_sum / (1e6 * steps), kernel_union_ms_per_step=union / (1e6 * steps),
                inter_kernel_gap_ms_per_step=(span - union) / (1e6 * steps),
                inter_kernel_gap_ns=span - union, gap_fraction_of_kernel_span=(span - union) / span, kernels=kernels,
                interpretation='Span is first kernel start to last kernel end, not wall step time. Gaps mean no '
                'kernel executing (not necessarily launch overhead); simultaneous streams count once in union. '
                'Names and launch shapes are evidence, not guessed GEMM roles.')


def require_ada(name, capability):
    if name.removeprefix('NVIDIA ') not in ('RTX 4000 Ada Generation', 'RTX 4000 Ada') or capability != 89:
        raise ValueError(f'requires RTX 4000 Ada (SM89); got {name}, SM{capability}')


def process_bytes(rows, pid):
    for row in rows:
        if row.pid == pid:
            used = row.usedGpuMemory
            if used is None or used < 0 or used == 2**64 - 1:
                raise ValueError('NVML process memory unavailable')
            return used
    return 0  # legitimately absent before this fresh process creates its context


def _check_cuda(name, code):
    if code != 0:
        raise RuntimeError(f'{name} returned CUDA error {code}')


def _pci_bus_of_cuda_zero():
    # NVML ordinals ignore CUDA_VISIBLE_DEVICES. Resolve the actual CUDA ordinal by PCI address,
    # without creating a CUDA context or renumbering the device reservation.
    driver = ctypes.CDLL('libcuda.so.1')
    dev = ctypes.c_int()
    bus = ctypes.create_string_buffer(32)
    _check_cuda('cuInit', driver.cuInit(0))
    _check_cuda('cuDeviceGet', driver.cuDeviceGet(ctypes.byref(dev), 0))
    _check_cuda('cuDeviceGetPCIBusId', driver.cuDeviceGetPCIBusId(bus, len(bus), dev))
    return bus.value.decode()


def _text(s):
    return s.decode() if isinstance(s, bytes) else s


@contextmanager
def telemetry():
    import pynvml as nv

    bus = _pci_bus_of_cuda_zero()
    nv.nvmlInit()
    try:
        h = nv.nvmlDeviceGetHandleByPciBusId(bus)
        name = _text(nv.nvmlDeviceGetName(h))
        major, minor = nv.nvmlDeviceGetCudaComputeCapability(h)
        require_ada(name, 10 * major + minor)
        gpu = dict(gpu_name=name, pci_bus_id=bus, gpu_uuid=_text(nv.nvmlDeviceGetUUID(h)),
                   driver=_text(nv.nvmlSystemGetDriverVersion()), nvml_poll_interval_s=SAMPLE_INTERVAL_S)
        pid = os.getpid()

        def sample():
            used = process_bytes(nv.nvmlDeviceGetComputeRunningProcesses(h), pid)
            return dict(t_ns=time.perf_counter_ns(), process_bytes=used,
                        sm_MHz=nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                        power_W=nv.nvmlDeviceGetPowerUsage(h) / 1000)

        samples = [sample()]
        stop = threading.Event()

        def poll():
            while not stop.wait(SAMPLE_INTERVAL_S):
                samples.append(sample())

        with ThreadPoolExecutor(max_workers=1) as executor:
            worker = executor.submit(poll)
            try:
                yield gpu, samples, sample
            finally:
                stop.set()
                worker.result()  # do not silently accept missing telemetry
    finally:
        nv.nvmlShutdown()


def hot_cell(n):
    import numpy as np
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from loupmd.bench.suites import srtio3

    at = srtio3(SIZES[n])
    MaxwellBoltzmannDistribution(at, temperature_K=300.0, rng=np.random.default_rng(42))
    return at


def timing_summary(start_ns, blocks):
    end_ns = blocks[-1]['end_ns']
    if end_ns - start_ns < 3_000_000_000:
        raise ValueError('need at least 3 s of continuous unprofiled blocks')
    previous = start_ns
    for i, block in enumerate(blocks):
        if 2 * (previous - start_ns) >= end_ns - start_ns:
            steps = sum(b['steps'] for b in blocks[i:])
            return dict(start_ns=start_ns, end_ns=end_ns, read_start_ns=previous, read_steps=steps,
                        ms_per_step=(end_ns - previous) / (1e6 * steps))
        previous = block['end_ns']
    raise ValueError('no full blocks in second half')


def _measurement_temperature(blocks, read_start_ns, atoms):
    # Endpoint temperatures for complete measured blocks, not the initial velocities or final snapshot.
    temp = statistics.median(b['final_temperature_K'] for b in blocks if b['end_ns'] > read_start_ns)
    # Gross kinetic-temperature screen, not a confidence interval for correlated blocks.
    if abs(temp - 300.) > 3 * 300. * math.sqrt(2. / (3 * atoms - 3)):
        raise ValueError(f'median measured temperature {temp:.2f} K outside the 300 K thermal window')
    return temp


def _sustain(run):
    # Size each block to about 150 ms after thermalisation and first-use autotuning.
    # The engine returns host-side final thermo, so run() has completed the steps when it returns.
    warm = run(WARMUP_STEPS, 1)
    n = max(1, min(4096, round(150 / max(warm.step_ms, 0.001))))
    blocks = []
    start = time.perf_counter_ns()
    while True:
        seed = len(blocks) + 2
        stats = run(n, seed)
        end = time.perf_counter_ns()
        blocks.append(dict(steps=n, seed=seed, end_ns=end, final_temperature_K=stats.final_temperature_k,
                           rebuilds=stats.rebuilds, dangerous_builds=stats.dangerous_builds))
        if end - start >= 3_000_000_000 and len(blocks) >= 2 and 2 * (blocks[-2]['end_ns'] - start) >= end - start:
            break
    return blocks, timing_summary(start, blocks)


def _capture(run, steps, seed):
    from loupmd import library_path

    # dlsym on the engine searches its dependencies: use the libcudart actually linked to
    # libloupmd, not an independently discovered toolkit (or a slow ldconfig subprocess).
    lib = ctypes.CDLL(str(library_path()))
    _check_cuda('cudaDeviceSynchronize', lib.cudaDeviceSynchronize())
    _check_cuda('cudaProfilerStart', lib.cudaProfilerStart())
    try:
        stats = run(steps, seed)
        _check_cuda('cudaDeviceSynchronize', lib.cudaDeviceSynchronize())
    finally:
        _check_cuda('cudaProfilerStop', lib.cudaProfilerStop())
    return stats


def _verified_bundle(path, expected):
    from loupmd.models import sha256_of

    with path.open('rb') as f:
        if f.read(8) != b'LOUPC001':
            raise ValueError('expected an exported .loup bundle, not a training checkpoint')
    digest = sha256_of(path)
    if digest != expected.lower():
        raise ValueError(f'bundle SHA256 mismatch: {path} is {digest}')
    return digest


def measure(args):
    from loupmd import LoupEngine, library_version

    digest = _verified_bundle(args.bundle, args.sha256)
    at = hot_cell(args.atoms)
    with telemetry() as (gpu, samples, sample):
        eng = LoupEngine(args.bundle, device_id=0)
        info = eng.info()
        require_ada(info.gpu_name, info.compute_capability)
        model_baseline = sample()
        eng.set_system(at.numbers, at.get_masses(), skin=0.5)
        eng.set_state(at.positions, at.get_velocities(), at.cell.array)

        def run(n, seed):
            st = eng.run_langevin(n, dt_fs=1.0, temp_k=300.0, friction_per_fs=0.01, seed=seed)
            if not (0 < st.final_temperature_k < math.inf and math.isfinite(st.final_potential_energy_ev)):
                raise ValueError(f'nonphysical trajectory at seed {seed}; refusing census')
            return st

        blocks, timing = _sustain(run)
        result = dict(mode=args.mode, bundle=str(args.bundle.resolve()), bundle_sha256=digest, build=library_version(),
                      kernel_signature=info.kernel_signature, cutoff_A=info.r_max, atoms=len(at),
                      system='periodic cubic SrTiO3', cell_A=at.cell.array.tolist(), pid=os.getpid(), **gpu,
                      initial_velocities='Maxwell-Boltzmann', velocity_seed=42, initial_temperature_K=at.get_temperature(),
                      target_temperature_K=300.0, integrator='Langevin', dt_fs=1.0, friction_per_fs=0.01, skin_A=0.5,
                      warmup_steps=WARMUP_STEPS, warmup_seed=1, conditioning_steps=sum(b['steps'] for b in blocks),
                      measured_block_median_K=_measurement_temperature(blocks, timing['read_start_ns'], len(at)),
                      conditioning_blocks=len(blocks), conditioning_block_steps=blocks[0]['steps'],
                      final_temperature_K=blocks[-1]['final_temperature_K'])
        if args.mode == 'trace':
            seed = blocks[-1]['seed'] + 1
            st = _capture(run, args.trace_steps, seed)
            result.update(profiled_steps=args.trace_steps, trace_seed=seed, final_temperature_K=st.final_temperature_k,
                          traced_rebuilds=st.rebuilds, traced_dangerous_builds=st.dangerous_builds)
        else:
            latter = [s for s in samples if timing['read_start_ns'] <= s['t_ns'] <= timing['end_ns']]
            if not latter:
                raise ValueError('no clock/power samples during measured second half')
            result.update(timing=timing, blocks=blocks, latter_half_clock_MHz=[s['sm_MHz'] for s in latter],
                          latter_half_power_W=[s['power_W'] for s in latter])
        last = sample()  # while the engine and its allocations still exist
    if last['process_bytes'] <= 0:
        raise ValueError('engine PID not visible to NVML; no usable memory measurement')
    samples.extend((model_baseline, last))
    samples.sort(key=lambda s: s['t_ns'])
    result.update(telemetry=samples, memory=dict(baseline_bytes=samples[0]['process_bytes'],
                  model_before_set_system_bytes=model_baseline['process_bytes'],
                  sampled_peak_bytes=max(s['process_bytes'] for s in samples)))
    return result


def _save(path, result):
    data = json.dumps(result, indent=2, allow_nan=False)
    with path.open('x', encoding='utf-8') as f:
        f.write(data + '\n')


def sweep(args):
    args.output.mkdir(parents=True, exist_ok=False)
    timings = {}
    records = {}
    for n in SIZES:
        prefix = args.output / str(n)
        timing = prefix.with_suffix('.time.json')
        trace = prefix.with_suffix('.trace.json')
        database = prefix.with_suffix('.sqlite')
        kernels = prefix.with_suffix('.kernels.json')
        python = [sys.executable, str(Path(__file__).resolve())]
        common = [str(args.bundle), '--sha256', args.sha256, '--atoms', str(n)]
        subprocess.run([*python, 'time', *common, '--output', str(timing)], check=True)
        subprocess.run(['nsys', 'profile', '--sample=none', '--cpuctxsw=none', '--trace=cuda',
                        '--cuda-graph-trace=node', '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                        '--output', str(prefix), *python, 'trace', *common, '--trace-steps', str(args.trace_steps),
                        '--output', str(trace)], check=True)
        subprocess.run(['nsys', 'export', '--type', 'sqlite', '--output', str(database),
                        str(prefix.with_suffix('.nsys-rep'))], check=True)
        tr = json.loads(trace.read_text())
        timings[n] = json.loads(timing.read_text())
        _save(kernels, parse_trace(database, tr['profiled_steps']))
        if any(tr[k] != timings[n][k] or tr[k] != timings[320][k] for k in ('gpu_uuid', 'bundle_sha256', 'build')):
            raise ValueError('GPU, bundle or build changed during census; refusing summary')
        records[n] = dict(timing=timing.name, trace=trace.name, kernels=kernels.name)

    def footprint(n):
        mem = timings[n]['memory']
        return mem['sampled_peak_bytes'] - mem['baseline_bytes']

    return dict(bundle_sha256=timings[320]['bundle_sha256'], gpu_uuid=timings[320]['gpu_uuid'], records=records,
                memory_slope_bytes_per_atom_5000_to_40000=(footprint(40000) - footprint(5000)) / 35000)


def _positive(text):
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError('steps must be positive')
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    for mode in ('sweep', 'time', 'trace', 'parse'):
        p = sub.add_parser(mode)
        p.add_argument('--output', required=True, type=Path, help='new directory (sweep) or new JSON file')
        if mode == 'parse':
            p.add_argument('database', type=Path)
            p.add_argument('--steps', required=True, type=_positive, help='steps inside this capture range only')
        else:
            p.add_argument('bundle', type=Path)
            p.add_argument('--sha256', required=True, help='digest of the trusted exported .loup')
            if mode != 'sweep':
                p.add_argument('--atoms', type=int, required=True, choices=SIZES)
            if mode != 'time':
                p.add_argument('--trace-steps', type=_positive, default=16)
    args = ap.parse_args(argv)
    if args.mode == 'sweep':
        _save(args.output / 'summary.json', sweep(args))
    elif args.mode == 'parse':
        _save(args.output, parse_trace(args.database, args.steps))
    else:
        _save(args.output, measure(args))


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
