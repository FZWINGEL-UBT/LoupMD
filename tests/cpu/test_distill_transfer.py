"""An interrupted or corrupt transfer must never publish a training split."""
import hashlib

import pytest

from loupmd.distill.transfer import assemble, chunks


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
