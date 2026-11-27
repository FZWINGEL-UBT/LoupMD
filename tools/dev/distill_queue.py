"""Coordinate finite shard IDs on shared POSIX storage with cross-host flock and close-to-open consistency.

Run at most one active worker per stable OWNER. Claims never expire. A restarted
worker reclaims its ID and token, and may replay publish with the same checked data.
The .sha256 sidecar is the public completion marker; queue.json retains producer receipts.
"""
import argparse
import fcntl
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path


def _atomic_write(path, data):
    # Stage on the destination filesystem. A killed process can leave a hidden
    # temp, but no reader can see a partial final file or mistake it for a shard.
    with tempfile.NamedTemporaryFile(prefix=f'.{path.name}.', dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        tmp.unlink(missing_ok=True)


def _save(path, state):
    _atomic_write(path, (json.dumps(state, indent=2) + '\n').encode())


def _publish(data, expected, dest, sidecar, receipt):
    # A shard is ~1.4 MB. Hash and write the *same bytes*, including when DATA is
    # on another mount. Never rename, unlink, or otherwise modify the caller's input.
    payload = data.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected.lower():
        raise ValueError(f'{data}: sha256 mismatch')
    if receipt is not None and digest != receipt['sha256']:
        raise ValueError('different digest already published for this claim')
    marker = f'{digest}  {dest.name}\n'.encode()
    have_data, have_marker = dest.exists(), sidecar.exists()
    # Refuse conflicts before changing any public file. On retry, either rename
    # may already have happened; leave matching files in place.
    if have_data and dest.read_bytes() != payload:
        raise ValueError(f'{dest}: conflicting existing shard')
    if have_marker and sidecar.read_bytes() != marker:
        raise ValueError(f'{sidecar}: conflicting existing sha256')
    if not have_data:
        _atomic_write(dest, payload)
    if not have_marker:
        _atomic_write(sidecar, marker)
    return digest


def _locked(args):
    """Read/mutate metadata only while holding the lock on its own persistent inode."""
    root = args.root
    path = root / 'queue.json'
    shards = root / 'shards'
    if args.command == 'init' and not path.exists():
        shards.mkdir(exist_ok=True)
        _save(path, {'first': args.first, 'last': args.last, 'claims': {}, 'receipts': {}})
        return
    state = json.loads(path.read_bytes())
    if args.command == 'init':
        if (args.first, args.last) != (state['first'], state['last']):
            raise ValueError('queue already initialized with different bounds')
        return

    claims, receipts = state['claims'], state['receipts']
    if args.command == 'claim':
        for key, record in claims.items():
            if record['owner'] == args.owner:
                return f"{key} {record['token']}"
        for shard in range(state['first'], state['last'] + 1):
            key = str(shard)
            if key not in claims and key not in receipts:
                token = uuid.uuid4().hex
                claims[key] = {'owner': args.owner, 'token': token}
                _save(path, state)
                return f'{key} {token}'
        return 'WAIT' if claims else 'DONE'

    key = str(args.id)
    receipt = receipts.get(key)
    record = claims.get(key) or receipt
    if record is None or (args.owner, args.token) != (record['owner'], record['token']):
        raise ValueError('no matching owner, shard and token')
    dest = shards / f'part-{args.id:02d}.xyz'
    sidecar = dest.with_suffix('.sha256')
    if args.command == 'release':
        if receipt is not None or dest.exists() or sidecar.exists():
            raise ValueError('publication already started; cannot release shard')
    else:
        digest = _publish(args.data, args.sha256, dest, sidecar, receipt)
        if receipt is not None:  # acknowledgement was lost; immutable receipt already recorded
            return
        receipts[key] = {**record, 'sha256': digest}
    del claims[key]
    _save(path, state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init')
    init.add_argument('first', type=int)
    init.add_argument('last', type=int)
    for name in ('claim', 'publish', 'release'):
        p = commands.add_parser(name)
        p.add_argument('owner')
        if name != 'claim':
            p.add_argument('id', type=int)
            p.add_argument('token')
        if name == 'publish':
            p.add_argument('data', type=Path)
            p.add_argument('sha256', help='hex digest of the validated input data')
    args = parser.parse_args(argv)
    try:
        initializing = args.command == 'init'
        if initializing:
            if not 0 <= args.first <= args.last:
                raise ValueError('require 0 <= FIRST <= LAST')
            args.root.mkdir(parents=True, exist_ok=True)
        elif not args.owner:
            raise ValueError('OWNER must be nonempty')
        # Never remove, truncate, or replace the lock file. Non-init operations
        # must not create queue files in an uninitialized directory.
        with (args.root / '.queue.lock').open('a+b' if initializing else 'r+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = _locked(args)
        if result is not None:
            print(result)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{parser.prog}: {exc}\n')


if __name__ == '__main__':
    main()
