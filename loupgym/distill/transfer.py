"""Stream Colab inputs in <=16 MB pieces; publish only the verified byte stream."""
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

CHUNK_BYTES = 16_000_000


def chunks(path, size=CHUNK_BYTES):
    if not 0 < size <= CHUNK_BYTES:
        raise ValueError('invalid Colab chunk size')
    with Path(path).open('rb') as stream:
        while block := stream.read(size):
            yield block


def assemble(parts, destination, digest):
    destination = Path(destination)
    check = sha256()
    with NamedTemporaryFile(dir=destination.parent, delete=False) as output:
        pending = Path(output.name)
        try:
            for part in parts:
                for block in chunks(part):
                    check.update(block)
                    output.write(block)
            if check.hexdigest() != digest:
                raise ValueError(f'sha256 mismatch: {destination}')
            output.close()
            pending.replace(destination)
        finally:
            output.close()
            pending.unlink(missing_ok=True)


def receive_inputs(staging):
    """One detached verifier: wait for the last upload, then publish checked inputs."""
    import json
    import time

    staging = Path(staging)
    marker = staging / 'ready'
    while not marker.exists() or not (digest := marker.read_bytes()).endswith(b'\n'):
        time.sleep(1)
    manifest = (staging / 'manifest').read_bytes()
    if digest != (sha256(manifest).hexdigest() + '\n').encode():
        raise ValueError('sha256 mismatch: input manifest')
    for i, (name, count, expected) in enumerate(json.loads(manifest)):
        if not name or Path(name).name != name:
            raise ValueError('expected an input basename')
        assemble((staging / f'{i}.{j}' for j in range(count)), staging.parent / name, expected)
