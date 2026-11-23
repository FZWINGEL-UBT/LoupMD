"""A partially transferred or mismatched Colab result must not reach evaluation."""
import hashlib
import io
import json
import tarfile

import pytest

from loupmd.distill.results import unpack


def archive(tmp_path, *, corrupt=False, swap_test=False, failed=False):
    base = tmp_path / 'splits'
    base.mkdir()
    sources = []
    for role in ('train', 'valid', 'teacher-test'):
        payload = role.encode()
        (base / f'{role}.xyz').write_bytes(payload)
        sources.append({'role': role, 'sha256': hashlib.sha256(payload).hexdigest()})
    m = dict(recipe={'epochs': 150, 'stage_two_start': 111}, student_epoch=149, pre_stage_two_epoch=110,
             student_sha256=hashlib.sha256(b'student').hexdigest(),
             pre_stage_two_sha256=hashlib.sha256(b'checkpoint').hexdigest(), sources=sources)
    contents = {'manifest.json': json.dumps(m).encode(), 'student.model': b'student', 'train.log': b'log',
                'job.log': b'job', 'plan.json': b'{}', 'pre_stage_two.pt': b'bad' if corrupt else b'checkpoint'}
    path = tmp_path / 'result.tar.gz'
    with tarfile.open(path, 'w:gz') as tar:
        for name, payload in contents.items():
            info = tarfile.TarInfo('out/' + name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    (tmp_path / 'allocation.json').write_text(json.dumps(dict(exit_code=int(failed),
          result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    if swap_test:
        (base / 'teacher-test.xyz').write_bytes(b'different test labels')
    return base


@pytest.mark.parametrize('problem', ['corrupt', 'swap_test', 'failed'])
def test_unqualified_result_does_not_publish(tmp_path, problem):
    base = archive(tmp_path, **{problem: True})
    with pytest.raises(ValueError):
        unpack(tmp_path, base)
    assert not (tmp_path / 'checked').exists()


@pytest.mark.parametrize('recovery', [False, True])
def test_boundary_and_final_survive_both_collection_paths(tmp_path, recovery):
    base = archive(tmp_path)
    if recovery:
        recovered = tmp_path / 'recovery'
        recovered.mkdir()
        (tmp_path / 'result.tar.gz').rename(recovered / 'final.tar.gz')
        sha = hashlib.sha256((recovered / 'final.tar.gz').read_bytes()).hexdigest()
        (recovered / 'final.sha256').write_text(f'{sha}  recovery/final.tar.gz\n')
        (tmp_path / 'allocation.json').write_text('{}')
    unpack(tmp_path, base)
    assert (tmp_path / 'checked' / 'pre_stage_two.pt').read_bytes() == b'checkpoint'
    assert (tmp_path / 'checked' / 'student.model').read_bytes() == b'student'
