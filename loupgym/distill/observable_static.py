"""Static cell relaxation and central-difference supercell Γ phonons.

The input cell sets the q coverage: repeating a primitive cell folds the
commensurate q mesh into Γ. Report signed, sorted frequencies of that *whole*
supercell, including imaginary and translational modes; no ASR projection or
absolute-value substitution. This is not a primitive-cell band-path comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import units
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import FIRE
from ase.vibrations.data import VibrationsData

from .observable_io import atoms_from, digest, geometry, load_model, model_args, publish, record


def relax(atoms, pressure_GPa, fmax, steps):
    if not np.isfinite(pressure_GPa) or not np.isfinite(fmax) or fmax <= 0 or steps < 1:
        raise ValueError('need finite pressure, positive fmax and a positive step budget')
    target = FrechetCellFilter(atoms, scalar_pressure=pressure_GPa * units.GPa)
    opt = FIRE(target, logfile=None)
    converged = opt.run(fmax=fmax, steps=steps)
    return {'converged': bool(converged), 'optimizer_steps': opt.nsteps,
            'max_filtered_force_eV_A': float(np.linalg.norm(target.get_forces(), axis=1).max()),
            'geometry': geometry(atoms)}


def columns(atoms, delta, start, stop):
    """Rows d²E/dx_i dx_j from +/- force evaluations, no single-sided fallback."""
    if not np.isfinite(delta) or delta <= 0 or not 0 <= start < stop <= 3*len(atoms):
        raise ValueError('invalid displacement or half-open Cartesian DOF range')
    x = atoms.positions.copy()
    rows = []
    try:
        for ix in range(start, stop):
            atom, axis = divmod(ix, 3)
            atoms.positions[atom, axis] = x[atom, axis] + delta
            plus = atoms.get_forces()
            atoms.positions[atom, axis] = x[atom, axis] - delta
            minus = atoms.get_forces()
            rows.append(((minus - plus).ravel() / (2 * delta)).tolist())
            atoms.positions[atom, axis] = x[atom, axis]
    finally:
        atoms.positions[:] = x
    return {'start': start, 'stop': stop, 'rows_eV_A2': rows}


def frequencies(atoms, shards):
    n = 3 * len(atoms)
    h = np.empty((n, n))
    seen = np.zeros(n, bool)
    for shard in shards:
        i, j = shard['start'], shard['stop']
        rows = np.asarray(shard['rows_eV_A2'])
        if not 0 <= i < j <= n or rows.shape != (j-i, n) or not np.isfinite(rows).all():
            raise ValueError('invalid Hessian shard')
        if seen[i:j].any():
            raise ValueError('overlap in phonon Hessian shards')
        h[i:j], seen[i:j] = rows, True
    if not seen.all():
        raise ValueError('missing Cartesian phonon displacements')
    energies = VibrationsData.from_2d(atoms, (h + h.T) * .5).get_energies()
    return np.sort((energies.real - energies.imag) * (units._e / units._hplanck / 1e12))


def compare_values(teacher_cell, student_cell, teacher_freq, student_freq, lattice_fraction=.002,
                   phonon_fraction=.03, phonon_floor_THz=.1):
    limits = [lattice_fraction, phonon_fraction, phonon_floor_THz]
    if not np.isfinite(limits).all() or min(limits) < 0:
        raise ValueError('invalid fixed tolerances')
    ta, sa, tf, sf = map(np.asarray, (teacher_cell, student_cell, teacher_freq, student_freq))
    if ta.shape != (3,) or sa.shape != (3,) or tf.ndim != 1 or sf.shape != tf.shape:
        raise ValueError('incompatible lattice or phonon spectra')
    dcell, dfreq = sa / ta - 1, sf - tf
    threshold = np.maximum(phonon_floor_THz, phonon_fraction * abs(tf))
    return {'tolerance_source': ('design §4 provisional starting values, not calibrated' if limits == [.002,.03,.1]
                                 else 'user-specified fixed tolerances, not estimated from MD seed spread'),
            'lattice_fraction_limit': lattice_fraction, 'phonon_fraction_limit': phonon_fraction,
            'phonon_floor_THz': phonon_floor_THz, 'lattice_fraction_error': dcell.tolist(),
            'phonon_error_THz': dfreq.tolist(), 'phonon_limit_THz': threshold.tolist(),
            'lattice_pass': (abs(dcell) <= lattice_fraction).tolist(), 'phonon_pass': (abs(dfreq) <= threshold).tolist()}


def add_commands(sub):
    p = sub.add_parser('relax', help='bounded FIRE cell+ion relaxation, zero or supplied hydrostatic pressure')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='ASE structure; first frame')
    source.add_argument('--continue-relax', type=Path, help='continue an unconverged artifact with a fresh FIRE block')
    model_args(p)
    p.add_argument('--pressure-GPa', type=float, default=0.)
    p.add_argument('--fmax', type=float, default=.001, help='max filter force in eV/Å')
    p.add_argument('--steps', type=int, required=True, help='max iterations per fresh FIRE block; match both models')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('phonons', help='finite-difference force rows on a converged relaxed supercell')
    model_args(p)
    p.add_argument('--relaxation', type=Path, required=True)
    p.add_argument('--repeat', type=int, nargs=3, default=[1,1,1])
    p.add_argument('--delta-A', type=float, default=.01)
    p.add_argument('--start', type=int, default=0, help='first Cartesian DOF, inclusive')
    p.add_argument('--stop', type=int, required=True, help='last Cartesian DOF, exclusive')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('spectrum', help='CPU: assemble a complete set of force shards')
    p.add_argument('--relaxation', type=Path, required=True)
    p.add_argument('--shards', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('static-compare', help='CPU: compare relaxed cell lengths and signed supercell spectra')
    for role in ['teacher', 'student']:
        p.add_argument('--'+role, type=Path, required=True, help='spectrum JSON')
    p.add_argument('--lattice-fraction', type=float, default=.002)
    p.add_argument('--phonon-fraction', type=float, default=.03)
    p.add_argument('--phonon-floor-THz', type=float, default=.1)
    p.add_argument('--output', type=Path, required=True)


def run(opts):
    import loupmd as lm

    if opts.output.exists():
        raise FileExistsError(opts.output)
    cmd = opts.command
    if cmd == 'relax':
        old = record(opts.continue_relax, 'relax') if opts.continue_relax else None
        at = atoms_from(old['geometry']) if old else read(opts.input, index=0)
        ff, model, build = load_model(opts, at)
        protocol = {'system': old['protocol']['system'] if old else digest(geometry(at)),
                    'engine_build': build, 'fmax_eV_A': opts.fmax, 'pressure_GPa': opts.pressure_GPa,
                    'steps_per_FIRE_block': opts.steps}
        if old and (old['converged'] or old['role'] != opts.role or old['model'] != model or old['protocol'] != protocol):
            raise ValueError('cannot extend converged or mismatched relaxation')
        at.calc = lm.calc(ff)
        data = dict(model=model, protocol=protocol, role=opts.role, previous=digest(old) if old else None,
                    **relax(at, opts.pressure_GPa, opts.fmax, opts.steps))
    elif cmd in ('phonons', 'spectrum'):
        r = record(opts.relaxation, 'relax')
        if not r['converged']:
            raise ValueError('phonons require a converged relaxation')
        if cmd == 'phonons':
            if min(opts.repeat) < 1:
                raise ValueError('repeat must be positive')
            at = atoms_from(r['geometry']).repeat(opts.repeat)
            ff, model, build = load_model(opts, at)
            if (model, opts.role, build) != (r['model'], r['role'], r['protocol']['engine_build']):
                raise ValueError('model, role or build does not match relaxation')
            at.calc = lm.calc(ff)
            data = dict(relaxation=digest(r), repeat=opts.repeat, delta_A=opts.delta_A,
                        **columns(at, opts.delta_A, opts.start, opts.stop))
        else:
            shards = [record(p, 'phonons') for p in opts.shards]
            repeat, delta = shards[0]['repeat'], shards[0]['delta_A']
            if any((s['relaxation'], s['repeat'], s['delta_A']) != (digest(r), repeat, delta) for s in shards):
                raise ValueError('incompatible phonon shards')
            at = atoms_from(r['geometry'])
            data = dict(relaxation=digest(r), shards=[digest(s) for s in shards], role=r['role'], model=r['model'],
                        protocol=dict(**r['protocol'], repeat=repeat, delta_A=delta), cell_lengths_A=at.cell.lengths().tolist(),
                        signed_supercell_gamma_THz=frequencies(at.repeat(repeat), shards).tolist())
    else:
        t, s = (record(getattr(opts, role), 'spectrum') for role in ('teacher', 'student'))
        if t['role'] != 'teacher' or s['role'] != 'student' or t['protocol'] != s['protocol']:
            raise ValueError('deterministic protocols or roles do not match')
        data = dict(teacher=digest(t), student=digest(s), protocol=t['protocol'], teacher_model=t['model'],
                    student_model=s['model'], **compare_values(t['cell_lengths_A'], s['cell_lengths_A'],
                    t['signed_supercell_gamma_THz'], s['signed_supercell_gamma_THz'], opts.lattice_fraction,
                    opts.phonon_fraction, opts.phonon_floor_THz))
    publish(opts.output, dict(kind=cmd, **data))
