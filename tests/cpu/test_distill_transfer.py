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
    completed = 17, hashlib.sha256(b'failed job and its log').hexdigest()
    replies = iter([subprocess.TimeoutExpired('colab download', 45), None, completed])
    monkeypatch.setattr(colab_distill.time, 'sleep', lambda seconds: None)

    def poll(timeout):
        response = next(replies)
        if isinstance(response, Exception):
            raise response
        return response

    # Preserve the real remote exit code so even a failed job's log can be downloaded.
    assert colab_distill._wait_for_exit(poll, colab_distill.time.monotonic() + 900) == completed


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


@pytest.mark.parametrize('exit_code', [0, 23])
def test_detached_payload_can_be_collected_without_exec(tmp_path, exit_code):
    import tarfile

    remote = tmp_path / 'vm'
    remote.mkdir()
    (remote / 'out').mkdir()
    (remote / 'train.sh').write_text(f'echo "epoch 149" >out/train.log\nexit {exit_code}\n')
    # Run the actual detached shell, without a live Colab kernel for status or retrieval.
    subprocess.run(['bash', '-c', colab_distill._PAYLOAD, 'job', 'train.sh'], cwd=remote)
    fetch = lambda name, dest: dest.write_bytes((remote / name).read_bytes())
    record = colab_distill._file_completion(tmp_path / 'receipt', fetch)
    assert record[0] == exit_code
    saved = tmp_path / 'result.tar.gz'
    assemble([remote / 'result.tar.gz'], saved, record[1])
    with tarfile.open(saved) as archive:
        assert archive.extractfile('out/train.log').read() == b'epoch 149\n'


def test_unpublished_completion_distinguishes_running_from_deleted_session(tmp_path):
    def missing_file(name, dest):
        raise RuntimeError(f'Download failed: File or directory not found: /content/loup-s/{name}')

    def missing_session(name, dest):
        raise RuntimeError("Session 'my-run' not found.")

    receipt = tmp_path / 'receipt'
    assert colab_distill._file_completion(receipt, missing_file) is None
    with pytest.raises(RuntimeError, match='Session'):
        colab_distill._file_completion(receipt, missing_session)


def test_status_outage_preserves_detached_result_until_file_api_recovers(tmp_path, monkeypatch):
    receipt = tmp_path / 'receipt'
    digest = hashlib.sha256(b'complete archive').hexdigest()
    replies = iter([RuntimeError('Download failed: HTTP 503 Service Unavailable'), f'0 {digest}\n'])
    monkeypatch.setattr(colab_distill.time, 'sleep', lambda _: None)

    def fetch(name, dest):
        result = next(replies)
        if isinstance(result, Exception):
            raise result
        dest.write_text(result)

    assert colab_distill._wait_for_exit(
        lambda _: colab_distill._file_completion(receipt, fetch),
        colab_distill.time.monotonic() + 90,
    ) == (0, digest)
