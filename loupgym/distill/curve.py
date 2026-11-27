"""Assemble a nested learning-curve prefix from the frozen baseline and numbered teacher shards.

Pin train.xyz with --base-sha256. Pin valid.xyz and teacher-test.xyz with the corresponding
options or trusted .sha256 sidecars in --base. Shards always require sha256sum sidecars.
Only the requested prefix must be complete. Nothing is published until every check passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from io import StringIO
from itertools import chain, islice
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase.io.extxyz import XYZError, read_xyz


BASE_FRAMES = 300
SHARD_FRAMES = 25
LEVELS = (1000, 3000, 10000)
_SHA = re.compile(r'[0-9a-fA-F]{64}')


def _sidecar(path):
    companion = path.with_suffix('.sha256')
    match = re.fullmatch(r'([0-9a-fA-F]{64})  ([^\r\n]+)\n', companion.read_text(encoding='utf-8'))
    # Allow a path relative to the writer's working directory, but never open that path.
    if not match or Path(match[2]).name != path.name:
        raise ValueError(f'{companion}: expected one sha256sum record for {path.name}')
    return match[1].lower()


def _shards(directories, needed):
    files = {}
    max_id = (max(LEVELS) - BASE_FRAMES) // SHARD_FRAMES - 1
    for directory in directories:
        for path in directory.iterdir():
            if not path.name.startswith('part-') or path.suffix not in ('.xyz', '.sha256'):
                continue  # claims, in-flight attempts, logs, etc.
            number = path.stem.removeprefix('part-')
            if not re.fullmatch(r'[0-9]+', number):
                raise ValueError(f'{path}: invalid shard name')
            i = int(number)
            if i > max_id or path.stem != f'part-{i:02d}':
                raise ValueError(f'{path}: invalid shard id or noncanonical name (range 0..{max_id})')
            if path.name in files:
                raise ValueError(f'ambiguous shard: both {files[path.name]} and {path}')
            files[path.name] = path
    paths = []
    for i in range(needed):
        name = f'part-{i:02d}.xyz'
        if name not in files:
            raise FileNotFoundError(f'missing shard {name}')
        paths.append(files[name])
    return paths


def _text(info, key):
    value = info.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'missing or invalid {key}')
    return value


def _integer(info, key):
    value = info.get(key)
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
        raise ValueError(f'missing or invalid {key}')
    return int(value)


class _Inventory:
    """Keep only IDs, geometry digests and counts once a frame has been checked."""

    def __init__(self):
        self.teacher = None
        self.composition = None
        self.frame_ids = set()
        self.trajectories = {}
        self.seeds = {}
        self.geometries = set()
        self.new_temperatures = Counter()

    def accept(self, a, split, replica):
        info = a.info
        if info.get('split') != split or info.get('sampler') != 'teacher':
            raise ValueError(f'expected teacher-sampled {split} split')
        teacher = (_text(info, 'teacher_sha256'), _text(info, 'teacher_head'))
        if not _SHA.fullmatch(teacher[0]) or self.teacher not in (None, teacher):
            raise ValueError('inconsistent teacher sha256 or head')
        self.teacher = teacher
        composition = tuple(np.bincount(a.numbers))  # also pins the number of atoms, not their ordering
        if self.composition not in (None, composition):
            raise ValueError('atom count or species composition differs from baseline')
        self.composition = composition

        for key, value, shape in [('REF_energy', info.get('REF_energy'), ()),
                                  ('REF_forces', a.arrays.get('REF_forces'), (len(a), 3)),
                                  ('REF_stress', info.get('REF_stress'), (3, 3))]:
            array = np.asarray(value)
            if array.shape != shape or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
                raise ValueError(f'missing or invalid {key}: expected finite values of shape {shape}')
        if not a.pbc.all() or not np.isfinite(a.cell).all() or not np.isfinite(a.positions).all() or a.cell.rank != 3:
            raise ValueError('invalid periodic geometry')
        geometry = hashlib.sha256(a.numbers.tobytes())
        geometry.update(a.cell.array.tobytes())
        geometry.update(a.positions.tobytes())
        geometry = geometry.digest()
        if geometry in self.geometries:
            raise ValueError('duplicate geometry')
        self.geometries.add(geometry)

        frame = _text(info, 'frame_id')
        trajectory = _text(info, 'trajectory_id')
        seed = _integer(info, 'md_seed')
        if frame in self.frame_ids:
            raise ValueError(f'duplicate frame_id {frame}')
        self.frame_ids.add(frame)
        # Only original time slices within one split may share their trajectory and its RNG stream.
        if trajectory in self.trajectories and (replica is not None or self.trajectories[trajectory] != (split, seed)):
            raise ValueError(f'duplicate or inconsistent trajectory_id {trajectory}')
        if seed in self.seeds and (replica is not None or self.seeds[seed] != trajectory):
            raise ValueError(f'duplicate md_seed {seed}')
        self.trajectories[trajectory] = split, seed
        self.seeds[seed] = trajectory
        if replica is not None:
            if _integer(info, 'replica') != replica:
                raise ValueError(f'expected replica {replica}')
            stride = _integer(info, 'stride')
            if not stride or _integer(info, 'step') != _integer(info, 'burn_in') + stride:
                raise ValueError('expected one terminal frame at burn_in + stride')
            temperature = info.get('temperature_K')
            if np.asarray(temperature).dtype.kind not in 'fiu' or np.ndim(temperature) != 0 or not np.isfinite(temperature) or temperature <= 0:
                raise ValueError('invalid temperature_K')
            t = float(temperature)
            self.new_temperatures[int(t) if t.is_integer() else t] += 1


def _copy(path, dest, inventory, split, pin, *, count=None, first_replica=None, train_digest=None):
    """Read the source once: the exact bytes parsed here are hashed and appended to staging."""
    digest = hashlib.sha256()
    frames = 0
    with path.open('rb') as stream:
        for first in stream:
            try:
                if count is not None and frames == count:
                    raise ValueError(f'expected only {count} frames')
                n = int(first)
                if n <= 0 or (inventory.composition is not None and n != sum(inventory.composition)):
                    raise ValueError(f'unexpected atom count {n} for baseline composition')
                raw = b''.join(chain((first,), islice(stream, n + 1)))
                # A missing newline would splice the next input into the last atom's row.
                if raw.count(b'\n') != n + 2:
                    raise ValueError('incomplete or unterminated extxyz frame')
                a = next(read_xyz(StringIO(raw.decode('utf-8')), index=0))
                replica = first_replica + frames if first_replica is not None else None
                inventory.accept(a, split, replica)
            except (ValueError, KeyError, TypeError, XYZError) as exc:
                raise ValueError(f'{path}, frame {frames + 1}: {exc}') from exc
            dest.write(raw)
            digest.update(raw)
            if train_digest is not None:
                train_digest.update(raw)
            frames += 1
    if not frames or (count is not None and frames != count):
        raise ValueError(f'{path}: {frames} frames, expected {count or "nonempty split"}')
    sha = digest.hexdigest()
    if sha != pin.lower():
        raise ValueError(f'{path}: sha256 mismatch (expected {pin}, read {sha})')
    return sha, frames


def _assemble(args):
    out = args.output
    if os.path.lexists(out):
        raise FileExistsError(f'refusing to replace {out}')
    paths = _shards(args.shards, (args.count - BASE_FRAMES) // SHARD_FRAMES)
    pins = {'train': args.base_sha256, 'valid': args.valid_sha256, 'teacher-test': args.teacher_test_sha256}
    inventory = _Inventory()
    sources, counts = {}, {}
    train_digest = hashlib.sha256()
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=f'.{out.name}.') as scratch:
        stage = Path(scratch) / 'dataset'
        stage.mkdir()
        for split, pin in pins.items():
            path = args.base / f'{split}.xyz'
            with (stage / path.name).open('xb') as dest:
                sources[path.name], counts[split] = _copy(
                    path, dest, inventory, split, pin if pin is not None else _sidecar(path),
                    count=BASE_FRAMES if split == 'train' else None,
                    train_digest=train_digest if split == 'train' else None)
        with (stage / 'train.xyz').open('ab') as dest:
            for i, path in enumerate(paths):
                sources[path.name], frames = _copy(path, dest, inventory, 'train', _sidecar(path), count=SHARD_FRAMES,
                                                   first_replica=i * SHARD_FRAMES, train_digest=train_digest)
                counts['train'] += frames
        record = dict(frames=counts['train'], new_independent_trajectories=counts['train'] - BASE_FRAMES,
                      split_counts=counts, new_temperature_counts=inventory.new_temperatures,
                      teacher_sha256=inventory.teacher[0], teacher_head=inventory.teacher[1],
                      input_sha256=sources, train_sha256=train_digest.hexdigest())
        (stage / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        if os.path.lexists(out):
            raise FileExistsError(f'refusing to replace {out}')
        stage.rename(out)  # atomic on the shared filesystem; completed directories are never empty
    return record


def verify_training_release(base, data, count):
    """Bind the transferred qualification receipt to the exact cold-start inputs."""
    from .train import file_digest

    receipt = json.loads((data / 'release.json').read_bytes())
    manifest = json.loads((data / 'manifest.json').read_bytes())
    base_receipt = base / 'manifest.json'
    if (count not in LEVELS or receipt.get('schema') != 'loupmd.curve-release.v2' or
            receipt.get('label_release') != 'curve-labels-v2' or
            receipt.get('base_manifest_sha256') != file_digest(base_receipt)):
        raise ValueError('wrong learning-curve release or baseline')
    splits = json.loads(base_receipt.read_bytes())['splits']
    counts = {k: count if k == 'train' else splits[k]['frames'] for k in ('train', 'valid', 'teacher-test')}
    groups = receipt.get('groups', {})
    # The v2 issuer qualifies four contiguous 25-frame shards as a single group.
    starts = range(0, (count - BASE_FRAMES) // SHARD_FRAMES, 4)
    if (receipt.get('split_counts') != counts or manifest['split_counts'] != counts or
            set(groups) != {str(i) for i in starts} or
            any(not _SHA.fullmatch(g.get(k, '')) for g in groups.values() for k in ('gate_sha256', 'receipt_sha256'))):
        raise ValueError('incomplete learning-curve release coverage')
    if not receipt.get('train_sha256') == manifest['train_sha256'] == file_digest(data / 'train.xyz'):
        raise ValueError('training data differ from the qualified release')
    for role in counts:
        name = role + '.xyz'
        pin = splits[role]['output_sha256']
        if (manifest['input_sha256'][name] != pin or file_digest(base / name) != pin or
                (role != 'train' and file_digest(data / name) != pin)):
            raise ValueError(f'{name}: release uses different baseline data')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True, help='frozen train.xyz, valid.xyz and teacher-test.xyz')
    p.add_argument('--base-sha256', required=True, help='trusted train.xyz digest')
    p.add_argument('--valid-sha256', help='trusted valid.xyz digest (default: valid.sha256)')
    p.add_argument('--teacher-test-sha256', help='trusted teacher-test.xyz digest (default: teacher-test.sha256)')
    p.add_argument('--shards', type=Path, action='append', required=True, help='shard directory; repeat as needed')
    p.add_argument('--count', type=int, choices=LEVELS, required=True, help='total training frames, including baseline')
    p.add_argument('--output', type=Path, required=True, help='new directory for the complete dataset and manifest')
    record = _assemble(p.parse_args(argv))
    print(json.dumps(record))


if __name__ == '__main__':
    main()
