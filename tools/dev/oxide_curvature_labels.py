"""Under a short reservation, issue native force pairs from corrected training geometries only."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import loupmd as lm
from loupgym.distill import curvature as c
from loupgym.distill.frames import _labels
from loupmd.validate import gate, references


def issue(source, dest):
    library = lm.library_path().resolve()
    if c.file_digest(library) != c.LIBRARY:
        raise ValueError('not the corrected native teacher library')
    pairs, _ = c.probes(source)
    teacher = lm.load('mh1', head=c.HEAD)
    if teacher.sha256 != c.TEACHER or teacher.head != c.HEAD:
        raise ValueError('wrong teacher')
    configs = [(c.geometry_id(a), a) for a in pairs[0][1:]]
    r64, r32, stress_ref = references.compute(teacher.path, configs, c.HEAD, 'cuda', None, None)
    sides, positions, forces, stock = [], [], [], []
    for i, (center, plus, minus) in enumerate(pairs):
        side, values = [], []
        for k, at in enumerate((plus, minus)):
            before = c.geometry_id(at)
            result = _labels(at, 'native', *teacher.evaluate(at, stress=True))
            ident = c.side_id(center, at, result[1])
            if before != ident['geometry']:
                raise ValueError('geometry changed during native evaluation')
            if i == 0:
                row = gate.compare(ident['geometry'], {}, len(at), result, r64[k], r32[k], stress_ref[k])
                if not row.passed:
                    raise ValueError(f'failed individual stock gate: {row}')
                stock.append(asdict(row))
            side.append(ident)
            values.append(result[1].copy())
        sides.append(side)
        forces.append(values)
        positions.append([plus.positions, minus.positions])
    hv64 = c.difference(r64[0][1], r64[1][1])
    noise = float(np.max(abs(c.difference(r32[0][1].astype(float), r32[1][1].astype(float)) - hv64)))
    err = float(np.max(abs(c.difference(*forces[0]) - hv64)))
    # A worst-case difference of two per-side force floors divided by 2 eps.
    limit = max(3 * noise, 1e-5 / c.EPS)
    if not np.isfinite([err, noise]).all() or err > limit:
        raise ValueError(f'failed differenced stock gate: {err} > {limit}')
    m = dict(schema='oxide.curvature.v1', train_sha256=c.TRAIN, library_sha256=c.file_digest(library),
             bundle_sha256=c.file_digest(teacher.bundle), seed=c.SEED, eps_A=c.EPS, sides=sides,
             stock_gate=stock, reference=references.device_tag('cuda'), versions=references.versions(),
             difference_gate=dict(passed=True, native_max_eV_A2=err, stock32_max_eV_A2=noise, limit_eV_A2=limit))
    dest.mkdir(exist_ok=False, parents=True)
    with TemporaryDirectory(dir=dest) as tmp:
        output = Path(tmp) / 'pairs.npz'
        np.savez_compressed(output, manifest=json.dumps(m, allow_nan=False), positions=positions,
                            forces=np.asarray(forces, dtype=np.float64))
        digest = c.file_digest(output)
        _, _, summary = c.checked(output, digest, source)
        os.link(output, dest / output.name)
        (dest / 'pairs.sha256').write_text(f'{digest}  pairs.npz\n')
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=Path)
    parser.add_argument('new_directory', type=Path)
    args = parser.parse_args()
    issue(args.train, args.new_directory)
