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
_PAYLOAD = r'''bash "$1"
rc=$?
if tar -czf result.tar.gz out && sha=$(sha256sum result.tar.gz); then
    printf '%s %s\n' "$rc" "${sha%% *}" >completion.part
else
    echo 'result packaging failed' >completion.part
fi
mv completion.part completion
exit "$rc"
'''


def _file_completion(local, fetch):
    local.unlink(missing_ok=True)
    try:
        fetch('completion', local)
    except RuntimeError as exc:
        if 'File or directory not found: /content/loup-s/completion' in str(exc):
            return None
        raise
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


def _confirmed_setup(body, local_receipt, directory, execute, download):
    """Recover a lost exec reply over Colab's independent file channel."""
    nonce = uuid.uuid4().hex
    receipt = f'{directory}/.ok-{nonce}'
    code = body + f'\nPath({receipt!r}).write_text({nonce!r})'
    for attempt in range(2):
        try:
            execute(code, retry=False, timeout=90)
            return
        except (subprocess.TimeoutExpired, _ConnectionLost):
            local_receipt.unlink(missing_ok=True)
            try:
                download(receipt, local_receipt)
            except (subprocess.TimeoutExpired, RuntimeError):
                if attempt:
                    raise
                continue  # only mkdir and checked assembly may be replayed here
            if local_receipt.read_text() != nonce:
                raise RuntimeError('wrong Colab setup receipt')
            return


def _confirm_launch(launch, read_pid, deadline):
    """A detached process may outlive both the exec reply and the file channel."""
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            launch(min(90, remaining))
            return
        except (subprocess.TimeoutExpired, _ConnectionLost):
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            pid = read_pid(min(45, remaining)).strip()
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
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
        (args.output / 'allocation.json').write_text(json.dumps(record, indent=2) + '\n')

    save()
    print(f'{session}: {args.gpu} account {args.account}', flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        def remote(body, **cli_options):
            program = Path(tmp) / 'exec.py'
            program.write_text('import os, json, subprocess\nfrom pathlib import Path\n' + body + '\nprint("LS0_OK")\n')
            result = cli('exec', '-s', session, '-f', program, '--timeout', '180', **cli_options)
            if 'LS0_OK' not in result.splitlines():
                raise RuntimeError(result)
            return result

        def setup(body):
            _confirmed_setup(body, Path(tmp) / 'receipt', REMOTE, remote,
                             lambda src, dst: cli('download', '-s', session, src, dst, timeout=45))

        try:
            # If the response is lost, do not request a second allocation; finally targets only this unique name.
            print(cli('new', '-s', session, '--gpu', args.gpu, retry=False, timeout=450), flush=True)
            record.update(allocated=time.time(), usage_during=cli('usage'))
            save()
            setup(f'Path({REMOTE!r}).mkdir(exist_ok=True)\nPath({(REMOTE + "/out")!r}).mkdir(exist_ok=True)')
            for file in files:
                parts = []
                checksum = hashlib.sha256()
                for i, block in enumerate(transfer.chunks(file)):
                    chunk = Path(tmp) / 'chunk'
                    chunk.write_bytes(block)
                    checksum.update(block)
                    name = f'{REMOTE}/.part{i}'
                    cli('upload', '-s', session, chunk, name, timeout=480)
                    parts.append(name)
                digest = checksum.hexdigest()
                # Leave the pieces until the next upload overwrites them: repeating assemble after a lost
                # response must still work. The VM is released at the end of the job.
                setup(Path(transfer.__file__).read_text()
                      + f'\nassemble({parts!r}, {REMOTE + "/" + file.name!r}, {digest!r})')
                record['input_sha256'][file.name] = digest
                save()
            # Exclusive launch marker makes replay after a lost response idempotent.
            launch_code = f'''os.chdir({REMOTE!r})
if not Path("started").exists():
    with open("started", "x"), open("out/job.log", "ab") as log:
        proc = subprocess.Popen(["bash", "-c", {_PAYLOAD!r}, "job", {args.job.name!r}], start_new_session=True,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        Path("pid.part").write_text(str(proc.pid))
        Path("pid.part").replace("pid")
if not Path("pid").is_file():
    raise RuntimeError("incomplete detached launch; refusing to start twice")'''

            def read_pid(timeout):
                pid = Path(tmp) / 'pid'
                pid.unlink(missing_ok=True)
                cli('download', '-s', session, REMOTE + '/pid', pid, retry=False, timeout=timeout)
                return pid.read_text()

            deadline = time.monotonic() + 3600 * args.hours
            _confirm_launch(lambda timeout: remote(launch_code, retry=False, timeout=timeout), read_pid, deadline)

            def poll(timeout):
                return _file_completion(Path(tmp) / 'completion', lambda name, dest:
                                        cli('download', '-s', session, f'{REMOTE}/{name}', dest,
                                            retry=False, timeout=timeout))

            rc, digest = _wait_for_exit(poll, deadline)
            download = args.output / 'result.part'
            cli('download', '-s', session, REMOTE + '/result.tar.gz', download, timeout=900)
            transfer.assemble([download], args.output / 'result.tar.gz', digest)
            download.unlink()
            record.update(exit_code=int(rc), result_sha256=digest)
            if int(rc):
                raise RuntimeError(f'{session}: job exit {rc}; see out/job.log in result.tar.gz')
        finally:
            record['stop_requested'] = time.time()
            save()
            print(cli('stop', '-s', session), flush=True)
            record.update(stopped=time.time(), usage_after=cli('usage'))
            save()


if __name__ == '__main__':
    main()
