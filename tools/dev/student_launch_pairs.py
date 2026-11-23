"""§9.6 Ada-only horizontal fusion experiment. Run under gpu_guard.py in an isolated archive.

Build with -DLOUP_STUDENT_LAUNCH_BENCH. Group the existing unsplit H2 projections of
layer 0, leaving the atom order, operands, tiles, arithmetic and FP32 fallbacks alone.
The changed dispatch also changes concurrency and block placement; this is a real fusion
comparison, not a per-launch latency calibration or a bound on all possible fusions.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from tools.dev import student_census as census


ARMS = ('separate', 'grouped', 'grouped', 'separate')
STEPS = 1024
BLOCKS = 24
TRACE_STEPS = 32


def compare_pairs(arms, times, separate_nodes, grouped_nodes):
    if tuple(arms) != ARMS or not 0 < grouped_nodes < separate_nodes:
        raise ValueError('need ABBA and a measured reduction in MD kernel nodes')
    if len(times) != 4 or any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError('need four positive finite step times')
    gains = [1 - times[1] / times[0], 1 - times[2] / times[3]]
    return dict(nodes_removed=separate_nodes - grouped_nodes, fractional_time_saved_AB_BA=gains,
                material_in_both_orders=all(g >= .05 for g in gains),
                majority_saved_in_both_orders=all(g >= .5 for g in gains))


def measure(args):
    import numpy as np
    from loupmd import LoupEngine, library_path

    arm, bundle = args.arm, args.bundle
    if arm == 'grouped':
        os.environ['LOUP_BENCH_GROUP_PROJECTIONS'] = '1'
    else:
        os.environ.pop('LOUP_BENCH_GROUP_PROJECTIONS', None)
    digest = census._verified_bundle(bundle, args.sha256)
    at = census.hot_cell(320)
    with census.telemetry() as (gpu, samples, _):
        eng = LoupEngine(bundle, device_id=0)
        info = eng.info()
        census.require_ada(info.gpu_name, info.compute_capability)
        eng.set_system(at.numbers, at.get_masses(), skin=.5)
        check = {}
        for seed in (1, 2, 3):
            pos = at.positions + np.random.default_rng(seed).normal(0, .04, at.positions.shape)
            e, f, s = eng.evaluate(pos, at.cell.array)
            check.update({f'{seed}_E': e, f'{seed}_F': f, f'{seed}_S': s})
        eng.set_state(at.positions, at.get_velocities(), at.cell.array)

        def run(n, seed):
            stats = eng.run_langevin(n, dt_fs=1., temp_k=300., friction_per_fs=.01, seed=seed)
            if not (0 < stats.final_temperature_k < math.inf and math.isfinite(stats.final_potential_energy_ev)):
                raise ValueError('non-finite MD')
            return stats

        run(8, 17)
        for key, data in zip(('position', 'velocity', 'force', 'energy'), eng.get_state()):
            check[key] = data
        if not all(np.isfinite(v).all() for v in check.values()):
            raise ValueError('non-finite validation output')
        np.savez(str(args.output) + '.npz', **check)
        run(100, 23)
        report = dict(arm=arm, bundle_sha256=digest, library_sha256=hashlib.sha256(Path(library_path()).read_bytes()).hexdigest(), **gpu)
        if args.mode == 'trace':
            stats = census._capture(run, TRACE_STEPS, 29)
            report.update(steps=TRACE_STEPS, rebuilds=stats.rebuilds)
        else:
            blocks = []
            start = time.perf_counter_ns()
            for i in range(BLOCKS):
                stats = run(STEPS, 100 + i)
                blocks.append(dict(steps=STEPS, end_ns=time.perf_counter_ns(), rebuilds=stats.rebuilds,
                                   final_temperature_K=stats.final_temperature_k, dangerous_builds=stats.dangerous_builds))
            timing = census.timing_summary(start, blocks)
            read = [s for s in samples if timing['read_start_ns'] <= s['t_ns'] <= timing['end_ns']]
            if not read:
                raise ValueError('missing latter-half telemetry')
            report.update(blocks=blocks, timing=timing, latter_half_clock_MHz=[s['sm_MHz'] for s in read],
                          latter_half_power_W=[s['power_W'] for s in read])
    census._save(args.output, report)


def compare_numerics(a, b):
    import numpy as np
    maxerr = {}
    with np.load(a) as ref, np.load(b) as changed:
        for key in ref:
            name = key.split('_')[-1]
            error = float(np.max(np.abs(changed[key] - ref[key])))
            maxerr[name] = max(error, maxerr.get(name, 0.))
    # Same geometries: total eV (320 atoms), eV/A, eV/A^3. Short common-seed MD:
    # Cartesian A, ASE velocity, eV/A and total eV. This is not the adoption/teacher gate.
    limits = dict(E=1e-4, F=1e-5, S=1e-7, position=1e-6, velocity=1e-6, force=1e-5, energy=1e-4)
    if set(maxerr) != limits.keys() or any(not e <= limits[k] for k, e in maxerr.items()):
        raise ValueError(f'same-build numerical controls failed: {maxerr}')
    return maxerr


def sweep(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cli = [sys.executable, '-m', 'tools.dev.student_launch_pairs']
    flags = [str(args.bundle), '--sha256', args.sha256]
    times = []
    traces = {}
    for i, arm in enumerate(ARMS):
        output = out / f'time-{i}-{arm}.json'
        subprocess.run([*cli, 'time', *flags, '--arm', arm, '--output', str(output)], check=True)
        row = json.loads(output.read_text())
        times.append(row)
    validations = [compare_numerics(str(out / 'time-0-separate.json.npz'), str(out / f'time-{i}-{arm}.json.npz'))
                   for i, arm in enumerate(ARMS) if i]
    for arm in ('separate', 'grouped'):
        prefix = out / arm
        record = out / (arm + '.json')
        subprocess.run(['nsys', 'profile', '--sample=none', '--cpuctxsw=none', '--trace=cuda', '--cuda-graph-trace=node',
                        '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--output', str(prefix),
                        *cli, 'trace', *flags, '--arm', arm, '--output', str(record)], check=True)
        subprocess.run(['nsys', 'export', '--type', 'sqlite', '--output', str(prefix.with_suffix('.sqlite')),
                        str(prefix.with_suffix('.nsys-rep'))], check=True)
        traces[arm] = census.parse_trace(prefix.with_suffix('.sqlite'), TRACE_STEPS)
        census._save(prefix.with_suffix('.kernels.json'), traces[arm])
        if any(json.loads(record.read_text())[k] != times[0][k] for k in ('gpu_uuid', 'bundle_sha256', 'library_sha256')):
            raise ValueError('trace identity mismatch')
    if any(row[k] != times[0][k] for row in times for k in ('gpu_uuid', 'bundle_sha256', 'library_sha256')):
        raise ValueError('paired run identity mismatch')
    geometry = {}
    for arm, trace in traces.items():
        h2 = [k for k in trace['kernels'] if 'hgemm2f_kernel<' in k['name']]
        geometry[arm] = dict(h2_blocks=sum(math.prod(k['grid']) * k['launches'] for k in h2),
                             grouped_nodes=sum(k['launches'] for k in h2 if 'StudentProjectionGroup' in k['name']))
    if (geometry['grouped']['h2_blocks'] != geometry['separate']['h2_blocks'] or
            geometry['separate']['grouped_nodes'] or geometry['grouped']['grouped_nodes'] != 2 * TRACE_STEPS):
        raise ValueError(f'unexpected tile work or missing grouping: {geometry}')
    ms = [r['timing']['ms_per_step'] for r in times]
    nodes = {k: v['launches_per_step'] for k, v in traces.items()}
    result = compare_pairs(ARMS, ms, nodes['separate'], nodes['grouped'])
    census._save(out / 'summary.json', dict(ms_per_step=ms, launches_per_step=nodes, geometry=geometry,
                 median_clock_MHz=[statistics.median(r['latter_half_clock_MHz']) for r in times],
                 median_power_W=[statistics.median(r['latter_half_power_W']) for r in times],
                 numerical_max_abs=validations, **result))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('sweep', 'time', 'trace'))
    ap.add_argument('bundle', type=Path)
    ap.add_argument('--sha256', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--arm', choices=set(ARMS), default='separate')
    args = ap.parse_args()
    (sweep if args.mode == 'sweep' else measure)(args)


if __name__ == '__main__':
    main()
