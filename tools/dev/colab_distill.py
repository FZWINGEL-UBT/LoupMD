"""Run one distillation job using the operator's already authenticated WSL Colab CLI.

Pass a bash job and its inputs; the job runs in /content/loup-s and puts deliverables in out/.
A new, unique session owns exactly this job. The local controller must stay alive until stop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from loupmd.distill import transfer

REMOTE = '/content/loup-s'

# Do not depend on a healthy notebook kernel after launching the detached job.
# Publish the receipt last, including on failure so the trainer's log is retrievable.
_PAYLOAD = r'''python -c "$2" && bash "$1"
rc=$?
if tar -czf result.tar.gz out && sha=$(sha256sum result.tar.gz); then
    printf '%s %s\n' "$rc" "${sha%% *}" >completion.part
else
    echo 'result packaging failed' >completion.part
fi
mv completion.part completion
exit "$rc"
'''


def _missing_session(exc):
    return "Session '" in str(exc) and "' not found" in str(exc)


def _file_completion(local, fetch):
    local.unlink(missing_ok=True)
    try:
        fetch('completion', local)
    except RuntimeError as exc:
        detail = str(exc)
        if f'File or directory not found: {REMOTE}/completion' in detail:
            return None
        if _missing_session(exc):
            raise  # confirmed deallocation, not just an unreachable Contents API
        raise _ConnectionLost(detail) from exc
    fields = local.read_text().split()
    if (len(fields) != 2 or not fields[0].isdecimal() or not 0 <= int(fields[0]) <= 255
            or len(fields[1]) != 64 or any(c not in '0123456789abcdef' for c in fields[1])):
        raise RuntimeError(f'invalid Colab completion receipt: {fields!r}')
    return int(fields[0]), fields[1]


class _ConnectionLost(RuntimeError):
    """No reply; the remote job may still be healthy."""


def _wait_for_exit(poll, deadline):
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            result = poll(min(45, remaining))
            if result is not None:
                return result
        except (subprocess.TimeoutExpired, _ConnectionLost) as exc:
            print(f'Status unavailable; keeping detached job: {exc}', flush=True)
        time.sleep(max(0, min(30, deadline - time.monotonic())))
    raise TimeoutError('remote job deadline exceeded')


def _collect_archive(dest, digest, fetch, deadline):
    """After completion, a transient download failure must not discard a whole run."""
    partial = dest.with_suffix('.part')
    try:
        while (remaining := deadline - time.monotonic()) > 0:
            partial.unlink(missing_ok=True)
            try:
                fetch(partial, min(900, remaining))
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                if _missing_session(exc):
                    raise
                print(f'Archive download unavailable; retaining completed job: {exc}', flush=True)
                time.sleep(max(0, min(5, deadline - time.monotonic())))
                continue
            sha = hashlib.sha256()
            for block in transfer.chunks(partial):
                sha.update(block)
            if sha.hexdigest() != digest:
                raise ValueError(f'sha256 mismatch: {dest}')
            partial.replace(dest)
            return
        raise TimeoutError('completed archive download deadline exceeded')
    finally:
        partial.unlink(missing_ok=True)


def cli(*args, retry=True, timeout=300):
    exe = os.environ.get('COLAB_CLI', str(Path.home() / '.local/bin/colab'))
    for attempt in range(3):
        p = subprocess.run([exe, *map(str, args)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout, stdin=subprocess.DEVNULL, check=False)
        if 'connection was lost' not in p.stdout.lower():
            if p.returncode:
                raise RuntimeError(p.stdout)
            return p.stdout
        if not retry or attempt == 2:
            raise _ConnectionLost(p.stdout)
        time.sleep(5)


def _input_verifier():
    return Path(transfer.__file__).read_text() + "\nreceive_inputs('staging')\n"


def _launch_program(root, job):
    # The execution cell owns the real computation until it finishes. A lost
    # notebook connection must neither kill the worker nor start it a second time.
    return f'''import os, subprocess
from pathlib import Path
Path({root!r}).mkdir(exist_ok=True)
os.chdir({root!r})
Path("out").mkdir(exist_ok=True)
Path("staging").mkdir(exist_ok=True)
if not Path("started").exists():
    with open("started", "x"), open("out/job.log", "ab") as log:
        proc = subprocess.Popen(["bash", "-c", {_PAYLOAD!r}, "job", {job!r}, {_input_verifier()!r}],
                                start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        Path("pid.part").write_text(str(proc.pid))
        Path("pid.part").replace("pid")
        proc.wait()
if not Path("pid").is_file():
    raise RuntimeError("incomplete detached launch; refusing to start twice")
'''


def _upload_inputs(files, tmp, upload):
    """Only the file channel is needed after launch; never reuse another input's parts."""
    stage = f'{REMOTE}/staging'
    chunk = Path(tmp) / 'chunk'
    manifest = []
    for i, file in enumerate(files):
        digest = hashlib.sha256()
        count = 0
        for block in transfer.chunks(file):
            chunk.write_bytes(block)
            digest.update(block)
            upload(chunk, f'{stage}/{i}.{count}')
            count += 1
        manifest.append((file.name, count, digest.hexdigest()))
    encoded = json.dumps(manifest).encode()
    chunk.write_bytes(encoded)
    upload(chunk, f'{stage}/manifest')
    chunk.write_text(hashlib.sha256(encoded).hexdigest() + '\n')
    upload(chunk, f'{stage}/ready')
    return {name: digest for name, _, digest in manifest}


def _confirm_launch(launch, read_pid, deadline):
    """A detached process may outlive both the exec reply and the file channel."""
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            launch(min(90, remaining))
        except (subprocess.TimeoutExpired, _ConnectionLost):
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            pid = read_pid(min(45, remaining)).strip()
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            if _missing_session(exc):
                raise
            print(f'Launch status unavailable; retaining session: {exc}', flush=True)
        else:
            if not pid.isascii() or not pid.isdecimal() or int(pid) <= 0:
                raise ValueError('invalid detached PID')
            return
        time.sleep(max(0, min(30, deadline - time.monotonic())))
    raise TimeoutError('detached job deadline exceeded while confirming launch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True, choices=('L4', 'T4'))
    parser.add_argument('--account', required=True, choices=('1', '2'), help='bookkeeping only; never logs in')
    parser.add_argument('--output', required=True, type=Path, help='a new local run directory')
    parser.add_argument('--hours', type=float, default=4, help='maximum detached job duration')
    parser.add_argument('job', type=Path)
    parser.add_argument('input', type=Path, nargs='*')
    args = parser.parse_args()
    files = [args.job, *args.input]
    if args.hours <= 0 or not all(f.is_file() for f in files) or len({f.name for f in files}) != len(files):
        parser.error('provide existing files with distinct basenames and a positive deadline')
    args.output.mkdir(parents=True, exist_ok=False)
    session = 'ls0-' + uuid.uuid4().hex[:12]
    record = {'session': session, 'account': args.account, 'gpu': args.gpu, 'start': time.time(),
              'usage_before': cli('usage'), 'input_sha256': {}}

    def save():
        pending = args.output / 'allocation.part'
        pending.write_text(json.dumps(record, indent=2) + '\n')
        pending.replace(args.output / 'allocation.json')

    save()
    print(f'{session}: {args.gpu} account {args.account}', flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        program = Path(tmp) / 'exec.py'
        program.write_text(_launch_program(REMOTE, args.job.name))
        client = None

        try:
            # If the response is lost, do not request a second allocation; finally targets only this unique name.
            print(cli('new', '-s', session, '--gpu', args.gpu, retry=False, timeout=450), flush=True)
            record['allocated'] = time.time()
            save()
            record['usage_during'] = cli('usage')
            save()
            deadline = time.monotonic() + 3600 * args.hours

            def launch(_timeout):
                nonlocal client
                if client is None or client.poll() is not None:
                    with (args.output / 'exec.log').open('ab') as log:
                        client = subprocess.Popen(
                            [os.environ.get('COLAB_CLI', str(Path.home() / '.local/bin/colab')), 'exec', '-s',
                             session, '-f', str(program), '--timeout', str(max(1, deadline - time.monotonic()))],
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)

            def read_pid(timeout):
                pid = Path(tmp) / 'pid'
                pid.unlink(missing_ok=True)
                cli('download', '-s', session, REMOTE + '/pid', pid, retry=False, timeout=timeout)
                return pid.read_text()

            _confirm_launch(launch, read_pid, deadline)
            record['input_sha256'] = _upload_inputs(
                files, tmp, lambda src, dst: cli('upload', '-s', session, src, dst, timeout=480))
            save()

            def poll(timeout):
                return _file_completion(Path(tmp) / 'completion', lambda name, dest:
                                        cli('download', '-s', session, f'{REMOTE}/{name}', dest,
                                            retry=False, timeout=timeout))

            rc, digest = _wait_for_exit(poll, deadline)
            record['remote_completion'] = {'exit_code': rc, 'sha256': digest}
            save()  # Not a local result until its verified bytes are published.
            _collect_archive(args.output / 'result.tar.gz', digest, lambda path, timeout:
                             cli('download', '-s', session, REMOTE + '/result.tar.gz', path,
                                 retry=False, timeout=timeout), time.monotonic() + 900)
            record.update(exit_code=int(rc), result_sha256=digest)
            if int(rc):
                raise RuntimeError(f'{session}: job exit {rc}; see out/job.log in result.tar.gz')
        finally:
            if client is not None and client.poll() is None:
                client.terminate()
                client.wait()
            record['stop_requested'] = time.time()
            save()
            print(cli('stop', '-s', session), flush=True)
            record['stopped'] = time.time()
            save()
            record['usage_after'] = cli('usage')
            save()


if __name__ == '__main__':
    main()
