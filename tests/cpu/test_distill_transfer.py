"""An interrupted or corrupt transfer must never publish a training split."""
import hashlib
import subprocess

import pytest

from loupmd.distill.transfer import assemble, chunks
from tools.dev import colab_distill


def test_chunk_boundary_and_atomic_digest_check(tmp_path):
    data = b'123456789'
    source = tmp_path / 'train.xyz'
    source.write_bytes(data)
    parts = list(chunks(source, size=4))
    assert parts == [b'1234', b'5678', b'9']
    names = []
    for i, part in enumerate(parts):
        name = tmp_path / str(i)
        name.write_bytes(part)
        names.append(name)
    dest = tmp_path / 'remote.xyz'
    digest = hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError, match='sha256'):
        assemble(names[::-1], dest, digest)
    assert not dest.exists()
    assemble(names, dest, digest)
    assert dest.read_bytes() == data
    names[-1].write_bytes(b'x')
    with pytest.raises(ValueError, match='sha256'):
        assemble(names, dest, digest)
    assert dest.read_bytes() == data


def test_status_timeout_does_not_discard_a_detached_job(monkeypatch):
    replies = iter([subprocess.TimeoutExpired('colab exec', 300), 'running', '17'])
    monkeypatch.setattr(colab_distill.time, 'sleep', lambda seconds: None)

    def poll(timeout):
        response = next(replies)
        if isinstance(response, Exception):
            raise response
        return response

    # Preserve the real remote exit code so even a failed job's log can be downloaded.
    assert colab_distill._wait_for_exit(poll, colab_distill.time.monotonic() + 900) == 17


def test_unreachable_status_cannot_run_past_paid_deadline(monkeypatch):
    now = [100.]
    monkeypatch.setattr(colab_distill.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(colab_distill.time, 'sleep', lambda seconds: now.__setitem__(0, now[0] + seconds))

    def poll(timeout):
        now[0] += timeout
        raise subprocess.TimeoutExpired('colab exec', timeout)

    with pytest.raises(TimeoutError, match='job deadline'):
        colab_distill._wait_for_exit(poll, 450.)
    assert now[0] == 450.
