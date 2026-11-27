"""Recover a completed distillation run only for the exact base splits that trained it."""
import argparse
import json
import tarfile
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory

from .train import BASE_SPLITS, file_digest

_FILES = {'student.model', 'pre_stage_two.pt', 'manifest.json', 'plan.json', 'train.log', 'job.log'}


def unpack(run, splits):
    output = run / 'checked'
    if output.exists():
        raise FileExistsError(output)
    archive = run / 'result.tar.gz'
    if archive.is_file():
        record = json.loads((run / 'allocation.json').read_text())
        if record.get('exit_code') != 0:
            raise ValueError('no successful training exit recorded')
        digest = record['result_sha256']
    else:
        # File-channel recovery on pre-receipt controllers checked remote exit=0
        # and both manifest hashes before atomically publishing this archive.
        archive = run / 'recovery/final.tar.gz'
        digest, name = (run / 'recovery/final.sha256').read_text().split()
        if name != 'recovery/final.tar.gz':
            raise ValueError('unexpected recovery archive name')
    if file_digest(archive) != digest:
        raise ValueError('training archive SHA256 mismatch')
    with TemporaryDirectory(dir=run) as tmp:
        ready = Path(tmp) / 'checked'
        ready.mkdir()
        wanted = {'out/' + n: n for n in _FILES}
        with tarfile.open(archive, 'r|gz') as contents:
            for entry in contents:
                if entry.name not in wanted:
                    continue
                dest = ready / wanted[entry.name]
                if not entry.isreg() or dest.exists():
                    raise ValueError(f'invalid training archive entry: {entry.name}')
                with contents.extractfile(entry) as src, dest.open('xb') as dst:
                    copyfileobj(src, dst)
        if {p.name for p in ready.iterdir()} != _FILES:
            raise ValueError('incomplete training artifact set')
        m = json.loads((ready / 'manifest.json').read_text())
        if (m['student_epoch'] != m['recipe']['epochs'] - 1 or
                m['pre_stage_two_epoch'] != m['recipe']['stage_two_start'] - 1):
            raise ValueError('wrong final or phase-boundary epoch')
        for file, key in [('student.model', 'student_sha256'), ('pre_stage_two.pt', 'pre_stage_two_sha256')]:
            if file_digest(ready / file) != m[key]:
                raise ValueError(f'{file}: training manifest SHA256 mismatch')
        expected = [(role, file_digest(splits / (role + '.xyz'))) for role in BASE_SPLITS]
        if [(s['role'], s['sha256']) for s in m['sources'][:3]] != expected:
            raise ValueError('training manifest base splits do not match evaluation inputs')
        ready.rename(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('splits', type=Path)
    args = parser.parse_args()
    print(unpack(args.run, args.splits))
