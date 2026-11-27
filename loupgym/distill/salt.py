"""Generate the matched Gaussian arm, or measure fixed-cell molten-salt transport."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import iread


def paired(frames, seed, sigma, evaluate):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('need positive Gaussian standard deviation')
    count = Counter()
    seen = set()
    for a in frames:
        traj, fid = a.info['trajectory_id'], a.info['frame_id']
        if a.info['split'] != 'train' or fid in seen:
            raise ValueError('expected unique training parents')
        seen.add(fid)
        count[traj] += 1
        b = a.copy()
        if count[traj] % 2 == 0:
            key = hashlib.sha256(f'{seed}:{sigma.hex()}:{fid}'.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(key, 'big'))
            b.positions += rng.normal(0, sigma, b.positions.shape)
            b.info.update(parent_frame_id=fid, frame_id=f'{fid}:gaussian:{key.hex()}', gaussian_sigma_A=sigma, gaussian_seed=seed)
            evaluate(b)
        yield b
    if not count or any(n % 2 for n in count.values()):
        raise ValueError('each trajectory needs a positive even frame count')


def transport(frames):
    """Multi-time-origin self MSD; subtract mass-weighted COM, fit middle lags in ps."""
    first = frames[0]
    if len(frames) < 12 or any(not np.array_equal(a.numbers, first.numbers) or not np.allclose(a.cell, first.cell)
                                or a.info['trajectory_id'] != first.info['trajectory_id'] for a in frames):
        raise ValueError('need one fixed-cell trajectory sampled at least twelve times')
    times = np.array([a.info['step'] * a.info['timestep_fs'] / 1000 for a in frames])
    dt = np.diff(times)
    if dt[0] <= 0 or not np.allclose(dt, dt[0]):
        raise ValueError('nonuniform cadence')
    frac = np.diff([a.get_scaled_positions(wrap=False) for a in frames], axis=0)
    frac -= np.rint(frac)
    if np.abs(frac).max() > .45:
        raise ValueError('saved-frame separation is too large for reliable unwrapping')
    dx = frac @ first.cell.array
    dx -= np.average(dx, axis=1, weights=first.get_masses())[:, None, :]
    pos = np.concatenate((np.zeros_like(dx[:1]), dx.cumsum(axis=0)))
    lags = np.arange(1, 1 + len(pos)//2)
    fit = lags >= len(pos)//8
    rows = {}
    for sym in sorted(set(first.get_chemical_symbols())):
        x = pos[:, np.array(first.get_chemical_symbols()) == sym]
        msd = np.array([((x[k:] - x[:-k])**2).sum(axis=-1).mean() for k in lags])
        rows[sym] = dict(D_A2_ps=float(np.polyfit(lags[fit]*dt[0], msd[fit], 1)[0]/6), msd_A2=msd.tolist())
    return dict(lag_ps=(lags*dt[0]).tolist(), fit_start_ps=float(lags[fit][0]*dt[0]), species=rows,
                volume_A3=first.get_volume(), density_g_cm3=first.get_masses().sum()*1.66053906660/first.get_volume())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('mix'); m.add_argument('output', type=Path); m.add_argument('--seed', type=int, required=True)
    m.add_argument('--sigma', type=float, required=True)
    for parser in (m, sub.add_parser('transport')):
        parser.add_argument('input', type=Path)
    args = p.parse_args()
    frames = iread(args.input, index=':', format='extxyz')
    if args.command == 'transport':
        print(json.dumps(transport(list(frames)), indent=2, allow_nan=False))
    else:
        import loupmd as lm
        from .frames import _evaluate, _write_set
        teacher = lm.load('mh1-omat', head='omat_pbe')
        def evaluate(a):
            if (a.info['teacher_sha256'], a.info['teacher_head']) != (teacher.sha256, 'omat_pbe'):
                raise ValueError('teacher mismatch')
            _evaluate(a, teacher, 'REF')
        _write_set(args.output, paired(frames, args.seed, args.sigma, evaluate))


if __name__ == '__main__':
    main()
