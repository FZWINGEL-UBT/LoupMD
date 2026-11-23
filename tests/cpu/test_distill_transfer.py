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


@pytest.mark.parametrize('error', [subprocess.TimeoutExpired('exec', 90), colab_distill._ConnectionLost('lost')])
def test_completed_setup_survives_lost_reply_without_resubmitting(tmp_path, error):
    # A non-repeatable action exposes a duplicate. The receipt is written on the
    # remote side, not by the test's download implementation.
    calls = []

    def execute(code, **kwargs):
        calls.append('exec')
        exec(code, {'Path': type(tmp_path)})
        raise error

    def download(remote, local):
        local.write_bytes(type(tmp_path)(remote).read_bytes())

    colab_distill._confirmed_setup(
        f'Path({str(tmp_path / "started")!r}).touch(exist_ok=False)',
        tmp_path / 'receipt', str(tmp_path), execute, download,
    )
    assert (tmp_path / 'started').is_file()
    assert calls == ['exec']


def test_unconfirmed_setup_cannot_advance_to_next_upload(tmp_path):
    def execute(code, **kwargs):
        raise subprocess.TimeoutExpired('exec', 90)

    def download(remote, local):
        local.write_text('receipt for a different operation')

    with pytest.raises(RuntimeError, match='receipt'):
        colab_distill._confirmed_setup('', tmp_path / 'receipt', '/content/loup-s', execute, download)


def test_ambiguous_launch_recovers_real_pid_after_file_channel_outage(tmp_path, monkeypatch):
    # Simulate a lost exec acknowledgement and two failed downloads. Run the guarded
    # command on disk: a replay must not replace the first launch's PID.
    clock = [0.]
    monkeypatch.setattr(colab_distill.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(colab_distill.time, 'sleep', lambda s: clock.__setitem__(0, clock[0] + s))
    pid = tmp_path / 'pid'
    attempts = []

    def launch(timeout):
        if not pid.exists():
            pid.write_text('12345')
        raise subprocess.TimeoutExpired('colab exec', timeout)

    def read_pid(timeout):
        attempts.append(timeout)
        if len(attempts) <= 2:
            raise RuntimeError('Download failed: File or directory not found')
        return pid.read_text()

    colab_distill._confirm_launch(launch, read_pid, 400.)
    assert len(attempts) == 3


def test_ambiguous_launch_keeps_original_paid_deadline(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(colab_distill.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(colab_distill.time, 'sleep', lambda s: clock.__setitem__(0, clock[0] + s))

    def unavailable(timeout):
        clock[0] += timeout
        raise subprocess.TimeoutExpired('colab', timeout)

    with pytest.raises(TimeoutError, match='deadline'):
        colab_distill._confirm_launch(unavailable, unavailable, 125.)
    assert clock[0] == 125.
