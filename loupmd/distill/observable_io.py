"""Content-identified records shared by Stage 0 observable jobs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from ase import Atoms


def digest(value):
    return hashlib.sha256(json.dumps(value, allow_nan=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def publish(path, value):
    """Publish an entire record or nothing. Never unlink an existing destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps({'sha256': digest(value), 'record': value}, allow_nan=False, indent=2)
    with NamedTemporaryFile(dir=path.parent, delete=False, mode='w', encoding='utf-8') as f:
        tmp = Path(f.name)
        try:
            f.write(data + '\n')
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            f.close()
            tmp.unlink()
            raise
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def record(path, kind=None):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    value = doc['record']
    if digest(value) != doc['sha256'] or kind is not None and value.get('kind') != kind:
        raise ValueError(f'{path}: record identity or kind mismatch')
    return value


def geometry(atoms):
    if len(atoms) < 1 or not atoms.pbc.all() or atoms.cell.rank != 3 or atoms.constraints:
        raise ValueError('need an unconstrained, nonempty, 3D periodic structure')
    values = dict(numbers=atoms.numbers.tolist(), masses=atoms.get_masses().tolist(),
                  cell=atoms.cell.array.tolist(), positions=atoms.positions.tolist())
    if (not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell).all()
            or not np.isfinite(atoms.get_masses()).all() or min(atoms.get_masses()) <= 0):
        raise ValueError('invalid atomic geometry or masses')
    return values


def atoms_from(values):
    return Atoms(**values, pbc=True)


def model_args(p):
    p.add_argument('--model', required=True, help='LoupMD model label, or a checkpoint with --sha256')
    p.add_argument('--sha256')
    p.add_argument('--head')
    p.add_argument('--role', required=True, choices=['teacher', 'student'])


def load_model(opts, atoms):
    import loupmd as lm

    ff = lm.load(opts.model, head=opts.head, sha256=opts.sha256)
    ff.check_elements(atoms.numbers)
    head = ff.head or (ff.heads[0] if ff.heads and len(ff.heads) == 1 else 'default')
    return ff, {'sha256': ff.sha256, 'head': head, 'name': ff.name}, lm.library_version()
