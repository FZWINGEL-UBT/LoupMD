"""Stage 0 calibration, one model and seed per reservation; CPU-only reductions.

Run `python -m loupmd.distill.observables COMMAND --help` for inputs. Example
for a SHORT path check (not a physical equilibration or uncertainty estimate):

  md --input oxide.extxyz --model mh1 --head omat_pbe --role teacher --seed 1 \
     --ensemble npt --temperature-K 300 --pressure-GPa 0 --timestep-fs 1 \
     --directory teacher-1 --steps 8
  extend --previous teacher-1/segment-8.json --steps 8

Repeat with seed 2; repeat with the student on the identical protocol and seed
pair. Use the segment JSONs (not mutable directories) as inputs to `statistics`.
`--discard-frames` removes equilibration; size blocks longer than the relevant
correlation time and inspect sensitivity to block length. Choose the diffusion
species and fit lags in the diffusive, not ballistic, regime. Do not infer
convergence, statistical independence of blocks or a calibrated tolerance from
a smoke run. For NVE use equilibrated input geometry, a production dt and a
smaller candidate dt, matching initial velocities, physical sample cadence
and total duration. For example sample every 1 x 1 fs and 2 x 0.5 fs for the
same number of samples. The smaller step is a candidate, not evidence that the
teacher conserves. For static measurements: relax -> phonons (shard by DOF) ->
spectrum -> static-compare. Only GPU commands (md, extend, relax, phonons) need
a reservation; budget their --steps or --start/--stop separately for each card.
"""
from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import numpy as np

from . import observable_md as md
from . import observable_static as static
from .observable_io import publish, record
from .observable_metrics import energy_metrics, partial_rdf, spread, statistics, unwrapped_path


def _pair(teacher, student, ensemble):
    t = [md.Snapshot(p) for p in teacher]
    s = [md.Snapshot(p) for p in student]
    first = t[0]
    if first.protocol['ensemble'] not in ensemble:
        raise ValueError(f'need ensemble in {ensemble}')
    for role, runs in [('teacher', t), ('student', s)]:
        for r in runs:
            if (r.protocol != first.protocol or r.end != first.end or r.origin['role'] != role
                    or r.origin['model'] != runs[0].origin['model']):
                raise ValueError('ensemble, role, model, protocol or endpoint mismatch')
        if len(set(r.origin['seed'] for r in runs)) != len(runs):
            raise ValueError('two different seeds are required')
    t.sort(key=lambda r: r.origin['seed'])
    s.sort(key=lambda r: r.origin['seed'])
    if s and [(r.origin['seed'], r.origin['initial_momenta_sha256']) for r in t] != [
            (r.origin['seed'], r.origin['initial_momenta_sha256']) for r in s]:
        raise ValueError('teacher/student initial states and seeds must be paired')
    return t, s


def _describe(run):
    return dict(endpoint_sha256=run.id, end_step=run.end, **run.origin)


def statistical_report(opts):
    t, s = _pair(opts.teacher, opts.student or [], ('npt', 'nvt'))
    if opts.bins < 1 or opts.discard_frames < 0 or not np.isfinite(opts.rmax_A) or opts.rmax_A <= 0:
        raise ValueError('invalid bins, rmax or equilibration length')
    edges = np.linspace(0., opts.rmax_A, opts.bins + 1)
    protocol = t[0].protocol
    interval = protocol['stride'] * protocol['timestep_fs']
    def evaluate(r):
        frames = islice(r.data(thermo=False), opts.discard_frames, None)
        return dict(run=_describe(r), **statistics(frames, interval, opts.block_frames, opts.lags, edges,
                                                   opts.diffuse, protocol['ensemble'] == 'npt'))
    t_values, s_values = list(map(evaluate, t)), list(map(evaluate, s))
    return dict(kind='statistics', sample_fs=interval, discard_frames=opts.discard_frames, block_frames=opts.block_frames,
                lag_fs=(np.asarray(opts.lags)*interval).tolist(), rdf_edges_A=edges.tolist(), diffusion_species=opts.diffuse,
                protocol=protocol, teacher=t_values, student=s_values, comparisons=spread(t_values, s_values),
                spread_definition='Absolute difference of two teacher seed means; compare absolute mean paired '
                'student-minus-teacher differences component by component. Descriptive, not a confidence interval.',
                block_error_definition='Standard deviation of equal nonoverlapping block estimates (ddof=1) / sqrt(blocks).',
                displacement_definition='RMS over all within-block time origins and atoms of each species; nearest-image '
                'increments exclude affine dilation. Sample faster than half a box crossing.',
                diffusion_definition='3D MSD slope including intercept over the requested lag grid, divided by six. '
                'Only meaningful on a demonstrated diffusive timescale.')


def _energy(run, discard, blocks):
    rows = list(islice(run.data(thermo=True), discard, None))
    t = np.asarray([float(r['time_ps'])*1000 for r in rows])
    e = np.asarray([float(r['total_eV_atom']) for r in rows])
    if len(e)//blocks < 2:
        raise ValueError('need at least two NVE samples per error block')
    metrics = energy_metrics(t, e)
    n = len(e)//blocks
    estimates = [energy_metrics(t[i*n:(i+1)*n], e[i*n:(i+1)*n]) for i in range(blocks)]
    expected = np.arange(discard, discard+len(e))*run.protocol['stride']*run.protocol['timestep_fs']
    if not np.allclose(t, expected, rtol=1e-12, atol=1e-9):
        raise ValueError('thermo times do not match integration and sampling steps')
    return dict(run=_describe(run), samples=len(e), time_fs=[t[0], t[-1]], block_frames=n, unused_sem_frames=len(e)%blocks,
                block_sem={k: float(np.std([x[k] for x in estimates], ddof=1) / np.sqrt(blocks)) for k in metrics}, **metrics)


def nve_report(opts):
    if opts.discard_frames < 0 or opts.blocks < 2:
        raise ValueError('need nonnegative discard and at least two error blocks')
    production, conserving = [], []
    for pair, dest in [(opts.production, production), (opts.conserving, conserving)]:
        if len(pair) not in (1, 2):
            raise ValueError('pass the teacher endpoint, then optionally the student endpoint, for each step')
        t, s = _pair(pair[:1], pair[1:], ('nve',))
        dest.extend(t+s)
    if len(production) != len(conserving):
        raise ValueError('both NVE step sets must have the same roles')
    for a,b in zip(production, conserving):
        pa, pb = a.protocol, b.protocol
        dta, dtb = pa['timestep_fs'], pb['timestep_fs']
        if (not dtb < dta or a.origin['seed'] != b.origin['seed'] or a.origin['model'] != b.origin['model']
                or a.origin['initial_momenta_sha256'] != b.origin['initial_momenta_sha256']
                or any(pa[k] != pb[k] for k in ('system', 'engine_build', 'initial_K', 'integrator'))
                or not np.isclose(a.end*dta, b.end*dtb, rtol=1e-12, atol=1e-12)
                or not np.isclose(pa['stride']*dta, pb['stride']*dtb, rtol=1e-12, atol=1e-12)):
            raise ValueError('need paired NVE initial conditions, equal physical duration/cadence and a smaller candidate step')
    result = {'kind':'nve', 'blocks':opts.blocks, 'discard_frames':opts.discard_frames,
              'criterion':'Compare full-window mean-centered total-energy RMS and absolute least-squares drift '
              'separately at each step. Smaller step is not certified conserving; inspect teacher data.'}
    for label, runs in [('production', production), ('candidate_conserving', conserving)]:
        rows = [_energy(r, opts.discard_frames, opts.blocks) for r in runs]
        result[label] = {'teacher': rows[0]}
        if len(rows) == 2:
            result[label].update(student=rows[1], no_worse_than_teacher={
                k: abs(rows[1][k]) <= abs(rows[0][k]) for k in ('drift_meV_atom_ps', 'rms_meV_atom')})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    md.add_commands(sub)
    static.add_commands(sub)
    p = sub.add_parser('statistics', help='CPU: matched two-seed spread, within-run block error and optional student comparison')
    p.add_argument('--teacher', nargs=2, type=Path, required=True, help='independent native segment endpoints')
    p.add_argument('--student', nargs=2, type=Path, help='same seeds and settings as teacher')
    p.add_argument('--discard-frames', type=int, required=True, help='discard first n samples for equilibration')
    p.add_argument('--block-frames', type=int, required=True, help='equal nonoverlapping block length in saved samples')
    p.add_argument('--lags', nargs='+', type=int, required=True, help='RMS displacements at these lags in saved samples; '
                   'also the MSD regression window for --diffuse')
    p.add_argument('--diffuse', nargs='+', default=[], help='species to fit, only when transport is applicable')
    p.add_argument('--rmax-A', type=float, required=True)
    p.add_argument('--bins', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('nve', help='CPU: compare NVE RMS and drift at matched production and smaller step')
    p.add_argument('--production', nargs='+', type=Path, required=True, help='teacher then optional paired student endpoint')
    p.add_argument('--conserving', nargs='+', type=Path, required=True, help='smaller candidate step: same roles and seeds')
    p.add_argument('--discard-frames', type=int, default=0)
    p.add_argument('--blocks', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    opts = parser.parse_args(argv)
    if opts.command in ('md', 'extend'):
        md.run(opts)
    elif opts.command in ('statistics', 'nve'):
        if opts.output.exists():
            raise FileExistsError(opts.output)
        publish(opts.output, statistical_report(opts) if opts.command == 'statistics' else nve_report(opts))
    else:
        static.run(opts)


if __name__ == '__main__':
    main()
