"""Exercise the NFS queue protocol using independent CLI processes and real files."""
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip('fcntl', reason='the shard coordinator requires POSIX file locking')
from tools.dev import distill_queue


SCRIPT = Path(distill_queue.__file__)


def cli(root, *args, ok=True):
    result = subprocess.run([sys.executable, str(SCRIPT), str(root), *map(str, args)],
                            capture_output=True, text=True, timeout=30)
    if ok:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
    return result.stdout.strip()


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_concurrent_claims_restart_and_wait(tmp_path):
    root = tmp_path / 'queue'
    owners = ['worker-' + str(i) for i in range(8)]
    cli(root, 'init', 28, 33)
    with ThreadPoolExecutor(max_workers=len(owners)) as pool:
        replies = dict(zip(owners, pool.map(lambda owner: cli(root, 'claim', owner), owners)))
    held = {owner: reply.split() for owner, reply in replies.items() if reply != 'WAIT'}
    assert sorted(int(i) for i, token in held.values()) == list(range(28, 34))
    assert len({token for i, token in held.values()}) == 6
    cli(root, 'init', 28, 33)
    cli(root, 'init', 28, 34, ok=False)
    for owner, reply in replies.items():
        assert cli(root, 'claim', owner) == reply


def test_unowned_and_stale_tokens_cannot_change_queue(tmp_path):
    root = tmp_path / 'queue'
    cli(root, 'init', 9, 9)
    shard, old = cli(root, 'claim', 'worker').split()
    data = tmp_path / 'upload'
    data.write_bytes(b'checked frame data')
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    before = snapshot(root)
    for owner, token in [('intruder', old), ('worker', 'wrong')]:
        cli(root, 'release', owner, shard, token, ok=False)
        cli(root, 'publish', owner, shard, token, data, digest, ok=False)
        assert snapshot(root) == before
    cli(root, 'release', 'worker', shard, old)
    same_shard, new = cli(root, 'claim', 'worker').split()
    assert same_shard == shard and new != old
    reclaimed = snapshot(root)
    cli(root, 'publish', 'worker', shard, old, data, digest, ok=False)
    cli(root, 'release', 'worker', shard, old, ok=False)
    cli(root, 'publish', 'worker', shard, new, data, '0' * 64, ok=False)
    assert snapshot(root) == reclaimed
    assert cli(root, 'claim', 'worker') == f'{shard} {new}'


@pytest.mark.parametrize('shard', [9, 100])
def test_publish_copies_input_and_preserves_receipt_on_replay(tmp_path, shard):
    root = tmp_path / 'queue'
    cli(root, 'init', shard, shard + 1)
    i, token = cli(root, 'claim', 'producer').split()
    assert int(i) == shard
    upload = tmp_path / 'unique.inflight'
    data = b'validated bytes\n'
    upload.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    cli(root, 'publish', 'producer', i, token, upload, digest)
    dest = root / 'shards' / f'part-{shard:02d}.xyz'
    assert upload.read_bytes() == dest.read_bytes() == data
    assert dest.with_suffix('.sha256').read_text() == f'{digest}  {dest.name}\n'
    receipt = json.loads((root / 'queue.json').read_text())['receipts'][i]
    assert receipt['owner'] == 'producer' and receipt['sha256'] == digest
    j, token2 = cli(root, 'claim', 'producer').split()
    assert int(j) == shard + 1
    before = snapshot(root)
    cli(root, 'publish', 'producer', i, token, upload, digest)
    cli(root, 'release', 'producer', i, token, ok=False)
    upload.write_bytes(b'different bytes')
    cli(root, 'publish', 'producer', i, token, upload, digest, ok=False)
    cli(root, 'publish', 'producer', i, token, upload, hashlib.sha256(upload.read_bytes()).hexdigest(), ok=False)
    assert snapshot(root) == before
    cli(root, 'publish', 'producer', j, token2, upload, hashlib.sha256(upload.read_bytes()).hexdigest())
    assert cli(root, 'claim', 'other') == 'DONE'


@pytest.mark.parametrize('stop_after', ['.xyz', '.sha256'])
def test_interrupted_publication_resumes_without_reassignment(tmp_path, monkeypatch, stop_after):
    root = tmp_path / 'queue'
    cli(root, 'init', 28, 28)
    shard, token = cli(root, 'claim', 'local').split()
    data = tmp_path / 'uploaded'
    data.write_bytes(b'validated frames')
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    replace = os.replace

    def interrupt(src, dst):
        replace(src, dst)
        if Path(dst).suffix == stop_after:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(os, 'replace', interrupt)
        with pytest.raises(KeyboardInterrupt):
            distill_queue.main([str(root), 'publish', 'local', shard, token, str(data), digest])
    dest = root / 'shards' / 'part-28.xyz'
    assert dest.read_bytes() == data.read_bytes()
    assert dest.with_suffix('.sha256').exists() == (stop_after == '.sha256')
    partial = snapshot(root)
    cli(root, 'release', 'local', shard, token, ok=False)
    conflicting = tmp_path / 'conflicting'
    conflicting.write_bytes(b'other valid bytes')
    cli(root, 'publish', 'local', shard, token, conflicting, hashlib.sha256(conflicting.read_bytes()).hexdigest(), ok=False)
    assert snapshot(root) == partial
    assert cli(root, 'claim', 'other') == 'WAIT'
    assert cli(root, 'claim', 'local') == f'{shard} {token}'
    cli(root, 'publish', 'local', shard, token, data, digest)
    assert dest.with_suffix('.sha256').read_text() == f'{digest}  part-28.xyz\n'
    assert cli(root, 'claim', 'other') == 'DONE'
