"""Frozen oxide curvature experiment: selection, paired identities and units (no torch)."""
import hashlib
import json

import numpy as np
from ase.io import iread

if __package__:
    from .train import file_digest
else:
    from train import file_digest

TRAIN = 'cfcb2131082b98746dac1ad661b31497de9f6ce50b8616a83c04457eb1773bc4'
PARENT = '7fd0ab37079430c20750c1a1dfb62e863e3fb149d661dae8906ed23b00c74d9c'
TEACHER = 'a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47'
LIBRARY = 'd12e88dee68635570258ef678111637bd859966d942cae25627a09c3992ba048'
HEAD = 'omat_pbe'
SEED, PAIRS, EPS, INTERVAL, SHARE = 70491, 25, .01, 3, .1


def geometry_id(at):
    h = hashlib.sha256()
    for array, dtype in [(at.numbers, '<i4'), (at.pbc, 'u1'), (at.cell.array, '<f8'), (at.positions, '<f8')]:
        h.update(np.asarray(array, dtype=dtype).tobytes())
    return h.hexdigest()


def probes(source):
    if file_digest(source) != TRAIN:
        raise ValueError('requires exactly the corrected oxide train split')
    frames = list(iread(source, index=':', format='extxyz'))
    if len(frames) != 300 or any((f.info['split'], f.info['sampler'], f.info['teacher_sha256'], f.info['teacher_head'])
                                 != ('train', 'teacher', TEACHER, HEAD) for f in frames):
        raise ValueError('wrong training or teacher identity')
    # One without-replacement seed stream for centers; independent stream per center for directions.
    centers = np.random.default_rng(SEED).choice(len(frames), PAIRS, replace=False)
    force_ms = np.mean([np.mean(f.arrays['REF_forces'] ** 2) for f in frames])  # all have 320 atoms
    pairs = []
    for index in centers:
        at = frames[index]
        rng = np.random.default_rng([SEED, int(index), 1])
        v = rng.choice([-1., 1.], size=at.positions.shape)
        plus, minus = at.copy(), at.copy()
        plus.positions += EPS * v
        minus.positions -= EPS * v
        pairs.append((at, plus, minus))
    return pairs, float(force_ms)


def side_id(center, side, forces):
    return dict(frame=center.info['frame_id'], center=geometry_id(center), geometry=geometry_id(side),
                teacher_sha256=TEACHER, teacher_head=HEAD,
                force_sha256=hashlib.sha256(np.asarray(forces, dtype='<f8').tobytes()).hexdigest())


def difference(plus, minus):
    """Directional derivative of FORCE (negative Hessian-vector product), eV/Å²."""
    return (plus - minus) / (2 * EPS)


def checked(path, expected_sha, source):
    if file_digest(path) != expected_sha:
        raise ValueError('pair file SHA256 mismatch')
    pairs, force_ms = probes(source)
    with np.load(path, allow_pickle=False) as data:
        m = json.loads(str(data['manifest']))
        if (m['schema'], m['train_sha256'], m['seed'], m['eps_A'], m['library_sha256']) != (
                'oxide.curvature.v1', TRAIN, SEED, EPS, LIBRARY):
            raise ValueError('incorrect curvature release')
        positions, forces = data['positions'], data['forces']
        if (forces.shape != (PAIRS, 2, 320, 3) or forces.shape != positions.shape
                or not np.isfinite(forces).all() or len(m['sides']) != PAIRS):
            raise ValueError('invalid or incomplete paired arrays')
        for i, (center, plus, minus) in enumerate(pairs):
            for k, at in enumerate((plus, minus)):
                if (not np.array_equal(at.positions, positions[i, k]) or
                        side_id(center, at, forces[i, k]) != m['sides'][i][k]):
                    raise ValueError('displaced geometry, side or teacher identity mismatch')
        reps = m['stock_gate']
        if ([r['config'] for r in reps] != [geometry_id(at) for at in pairs[0][1:]] or
                not all(r['passed'] is True for r in reps) or not m['difference_gate']['passed']):
            raise ValueError('missing passing stock gate on signed representative pair')
    target = difference(forces[:, 0], forces[:, 1])
    hv_ms = float(np.mean(target**2))
    if not np.isfinite([force_ms, hv_ms]).all() or min(force_ms, hv_ms) <= 0:
        raise ValueError('invalid training-only curvature scales')
    return pairs, target, dict(pairs_sha256=expected_sha, pairs=PAIRS, probe_seed=SEED, eps_A=EPS,
                               pair_shuffle='PCG64([seed,epoch,2])', topology='independent per sign',
                               interval=INTERVAL, force_share=SHARE, force_ms=force_ms, difference_ms=hv_ms,
                               on_batch_weight_A2=100 * SHARE * INTERVAL * force_ms / hv_ms,
                               stock_gate=reps, difference_gate=m['difference_gate'])


def schedule(epoch):
    """One complete independently permuted pass over intact pairs in 75 ordinary updates."""
    order = np.random.default_rng([SEED, epoch, 2]).permutation(PAIRS)
    return [int(order[i // INTERVAL]) if (i+1) % INTERVAL == 0 else None for i in range(PAIRS * INTERVAL)]
