"""CPU estimators; ASE positions/cell in Å, time in fs and energy in eV."""
from itertools import combinations_with_replacement, islice

import numpy as np
from ase import units
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list


def partial_rdf(atoms, edges):
    """Unique pairs divided by their ideal-gas counts in spherical shells."""
    edges = np.asarray(edges, float)
    if edges.ndim != 1 or len(edges) < 2 or edges[0] != 0 or not (np.diff(edges) > 0).all():
        raise ValueError('RDF edges must increase from zero')
    heights = 1 / np.linalg.norm(np.linalg.inv(atoms.cell), axis=0)
    if not atoms.pbc.all() or 2 * edges[-1] > heights.min():
        raise ValueError('RDF cutoff must fit half of every periodic cell height at every frame')
    i, j, d = neighbor_list('ijd', atoms, edges[-1])
    symbols = np.array(atoms.get_chemical_symbols())
    counts = dict(zip(*np.unique(symbols, return_counts=True)))
    shells = (4 * np.pi / 3 / atoms.get_volume()) * np.diff(edges ** 3)
    result = {}
    for a, b in combinations_with_replacement(sorted(counts), 2):
        if a == b:
            possible = counts[a] * (counts[a] - 1) // 2
            mask = (i < j) & (symbols[i] == a) & (symbols[j] == a)
        else:
            possible = counts[a] * counts[b]
            mask = (symbols[i] == a) & (symbols[j] == b)
        if possible:
            result[f'{a}-{b}'] = np.histogram(d[mask], edges)[0] / (possible * shells)
    return result


def unwrapped_path(frames):
    """Cumulative minimum-image increments in the new cell (excludes affine NPT motion).

    As for any wrapped trajectory, the true per-sample displacement must be the
    nearest image. Shorten the sampling interval when that cannot be guaranteed.
    """
    first = frames[0]
    path = np.zeros((len(frames), len(first), 3))
    previous = first.get_scaled_positions(wrap=False)
    for t, frame in enumerate(frames[1:], 1):
        if not np.array_equal(first.numbers, frame.numbers) or not frame.pbc.all():
            raise ValueError('atom order and periodicity must be fixed throughout the trajectory')
        now = frame.get_scaled_positions(wrap=False)
        delta, _ = find_mic((now - previous) @ frame.cell, frame.cell, frame.pbc)
        path[t] = path[t - 1] + delta
        previous = now
    return path


def statistics(frames, sample_fs, block_frames, lags, edges, diffuse, npt):
    """Each equal, nonoverlapping block gives one estimate. Never pad a short tail."""
    lags = np.asarray(lags)
    if (sample_fs <= 0 or not np.isfinite(sample_fs) or block_frames < 2 or lags.ndim != 1
            or lags.dtype.kind not in 'iu' or not len(lags) or lags[0] < 1 or lags[-1] >= block_frames
            or not (np.diff(lags) > 0).all() or (diffuse and len(lags) < 2)):
        raise ValueError('need ordered positive lag counts inside a block (two or more for diffusion)')
    times = sample_fs * lags
    iterator, blocks, tail = iter(frames), [], 0
    ref = None
    while batch := list(islice(iterator, block_frames)):
        if len(batch) != block_frames:
            tail = len(batch)
            break
        first = batch[0]
        if ref is None:
            ref = first
        if any(not np.array_equal(f.numbers, ref.numbers) or not np.array_equal(f.get_masses(), ref.get_masses())
               or not f.pbc.all() for f in batch):
            raise ValueError('atom order, masses and periodicity must be fixed throughout the trajectory')
        symbols = np.array(first.get_chemical_symbols())
        if set(diffuse) - set(symbols):
            raise ValueError('requested diffusion species not present')
        values = {}
        for f in batch:
            for pair, rdf in partial_rdf(f, edges).items():
                key = 'rdf:' + pair
                values[key] = values.get(key, 0) + rdf / block_frames
        path = unwrapped_path(batch)
        for symbol in sorted(set(symbols)):
            x = path[:, symbols == symbol]
            msd = np.array([((x[lag:] - x[:-lag])**2).sum(axis=2).mean() for lag in lags])
            values['rms_displacement_A:' + symbol] = np.sqrt(msd)
            if symbol in diffuse:
                # <|r(t)-r(0)|²> = 6 D t in 3D. 1 Å²/fs = 0.1 cm²/s.
                values['diffusion_cm2_s:' + symbol] = np.polyfit(times, msd, 1)[0] * (.1 / 6)
        if npt:
            values['cell_lengths_A'] = np.mean([f.cell.lengths() for f in batch], axis=0)
            values['density_g_cm3'] = np.mean([f.get_masses().sum() / f.get_volume() for f in batch]) * units._amu * 1e27
        values['temperature_K'] = np.mean([f.get_kinetic_energy() * 2 / ((3*len(f)-3) * units.kB)
                                           for f in batch]) if len(first) > 1 else np.mean([f.get_temperature() for f in batch])
        blocks.append(values)
    if len(blocks) < 2:
        raise ValueError('need at least two complete blocks for block standard errors')
    result = {}
    for key in blocks[0]:
        estimates = np.stack([b[key] for b in blocks])
        result[key] = {'mean': estimates.mean(0).tolist(),
                       'block_sem': (estimates.std(0, ddof=1) / np.sqrt(len(blocks))).tolist()}
    return {'blocks': len(blocks), 'unused_tail_frames': tail, 'metrics': result}


def spread(teacher, student=()):
    """Absolute difference of two independent teacher seed estimates, not a confidence interval."""
    if len(teacher) != 2 or len(student) not in (0, 2):
        raise ValueError('need two teacher seeds and optionally both paired student seeds')
    keys = set(teacher[0]['metrics']) - {'temperature_K'}  # temperature is a condition diagnostic
    if any(keys != set(run['metrics']) - {'temperature_K'} for run in [*teacher, *student]):
        raise ValueError('different observables in compared trajectories')
    result = {}
    for key in sorted(keys):
        t = np.asarray([r['metrics'][key]['mean'] for r in teacher])
        tolerance = abs(t[1] - t[0])
        row = {'teacher_mean': t.mean(0).tolist(), 'teacher_seed_spread': tolerance.tolist()}
        if student:
            s = np.asarray([r['metrics'][key]['mean'] for r in student])
            if s.shape != t.shape:
                raise ValueError(f'different shape of {key}')
            delta = (s - t).mean(0)
            row.update(student_minus_teacher=delta.tolist(), within_teacher_spread=(abs(delta) <= tolerance).tolist())
        result[key] = row
    return result


def energy_metrics(time_fs, total_eV_atom):
    t, e = np.asarray(time_fs, float), np.asarray(total_eV_atom, float)
    if (t.ndim != 1 or e.shape != t.shape or len(t) < 2 or not np.isfinite(e).all()
            or not np.isfinite(t).all() or not (np.diff(t) > 0).all()):
        raise ValueError('need finite energies at increasing times')
    t = (t - t.mean()) / 1000
    e = (e - e.mean()) * 1000
    return {'rms_meV_atom': float(np.sqrt(np.mean(e**2))), 'drift_meV_atom_ps': float(np.dot(e, t) / np.dot(t, t))}
