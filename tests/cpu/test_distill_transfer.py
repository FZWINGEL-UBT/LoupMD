"""An interrupted or corrupt transfer must never publish a training split."""
import hashlib
import subprocess

import pytest

from loupmd.distill.transfer import assemble, chunks
from tools.dev import colab_distill


@pytest.mark.parametrize('disconnect', [False, True])
def test_execution_cell_owns_real_work_but_disconnection_does_not_kill_it(tmp_path, disconnect):
    import os
    import shutil
    import sys
    import time
    from pathlib import Path

    vm = tmp_path / 'vm'
    job = tmp_path / 'train.sh'
    job.write_text('echo finished >out/model')
    cell = subprocess.Popen([sys.executable, '-c', colab_distill._launch_program(str(vm), job.name)])
    try:
        deadline = time.monotonic() + 15
        while not (vm / 'pid').exists() and time.monotonic() < deadline:
            time.sleep(.05)
        pid = int((vm / 'pid').read_text())
        assert os.getsid(pid) == pid
        # The verified inputs are not available yet: the legitimate computation
        # is still running and the execution cell must not have returned.
        with pytest.raises(subprocess.TimeoutExpired):
            cell.wait(timeout=.2)
        if disconnect:
            cell.terminate()
            cell.wait(timeout=3)
        colab_distill._upload_inputs([job], tmp_path,
                                    lambda src, dst: shutil.copyfile(src, vm / 'staging' / Path(dst).name))
        if not disconnect:
            assert cell.wait(timeout=15) == 0
        deadline = time.monotonic() + 15
        while not (vm / 'completion').exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert (vm / 'completion').read_text().split()[0] == '0'
        assert (vm / 'out/model').read_text().strip() == 'finished'
    finally:
        if cell.poll() is None:
            cell.kill()
            cell.wait()


@pytest.mark.parametrize('corrupt', [False, True])
def test_detached_job_verifies_every_input_without_a_second_kernel_request(tmp_path, corrupt):
    from pathlib import Path
    import shutil

    vm = tmp_path / 'vm'
    stage = vm / 'staging'
    stage.mkdir(parents=True)
    (vm / 'out').mkdir()
    files = [tmp_path / name for name in ('train.sh', 'train.xyz', 'valid.xyz')]
    for file, data in zip(files, (b'cat train.xyz valid.xyz >out/trained', b'train frames', b'valid frames')):
        file.write_bytes(data)
    # Only the file channel is available after launching the detached verifier.
    program = colab_distill._input_verifier()
    proc = subprocess.Popen(['bash', '-c', colab_distill._PAYLOAD, 'job', 'train.sh', program], cwd=vm)
    try:
        def upload(local, remote):
            dest = stage / Path(remote).name
            shutil.copyfile(local, dest)
            if corrupt and dest.read_bytes() == b'valid frames':
                dest.write_bytes(b'wrong frames')

        received = colab_distill._upload_inputs(files, tmp_path, upload)
        code = proc.wait(timeout=15)
        assert received == {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        if corrupt:
            assert code != 0 and not (vm / 'out/trained').exists()
        else:
            assert code == 0 and (vm / 'out/trained').read_bytes() == b'train framesvalid frames'
        assert int((vm / 'completion').read_text().split()[0]) == code
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


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


def test_final_download_recovers_after_partial_file_with_published_digest(tmp_path, monkeypatch):
    contents = b'completed training archive'
    digest = hashlib.sha256(contents).hexdigest()
    tries = []
    monkeypatch.setattr(colab_distill.time, 'sleep', lambda _: None)

    def fetch(dest, timeout):
        assert not dest.exists() and 0 < timeout <= 900
        tries.append(timeout)
        if len(tries) == 1:
            dest.write_bytes(contents[:4])
            raise RuntimeError('Download failed: HTTP 503 Service Unavailable')
        dest.write_bytes(contents)

    dest = tmp_path / 'result.tar.gz'
    colab_distill._collect_archive(dest, digest, fetch, colab_distill.time.monotonic() + 900)
    assert dest.read_bytes() == contents


def test_final_download_never_publishes_corrupt_or_missing_vm_result(tmp_path):
    dest = tmp_path / 'result.tar.gz'
    digest = hashlib.sha256(b'trained artifact').hexdigest()
    deadline = colab_distill.time.monotonic() + 900

    def gone(path, timeout):
        raise RuntimeError("Session 'my-run' not found.")

    with pytest.raises(RuntimeError, match="Session 'my-run' not found"):
        colab_distill._collect_archive(dest, digest, gone, deadline)
    with pytest.raises(ValueError, match='sha256'):
        colab_distill._collect_archive(dest, digest, lambda p, _: p.write_bytes(b'truncated'), deadline)
    assert not dest.exists()


def test_unadmitted_worker_cannot_consume_a_full_training_allocation(monkeypatch):
    now = [0.]
    monkeypatch.setattr(colab_distill.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(colab_distill.time, 'sleep', lambda s: now.__setitem__(0, now[0] + s))

    def no_pid(_):
        raise RuntimeError('File or directory not found: /content/loup-s/pid')

    with pytest.raises(TimeoutError, match='launch'):
        colab_distill._confirm_launch(lambda _: None, no_pid, 5 * 3600.)
    assert now[0] == 300.
