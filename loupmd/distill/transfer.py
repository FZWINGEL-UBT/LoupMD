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
