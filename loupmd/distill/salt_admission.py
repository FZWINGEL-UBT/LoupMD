"""Bind one released salt student to its exact gated checkpoint and native arm plan."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def admitted(root, arm):
    plan_name = {'trajectory': 'train-plan.json', 'mix': 'train-mix-plan.json'}[arm]
    job = root / f'{arm}-s1-v1'; release = job / 'release'
    plan = json.loads((root/plan_name).read_text())
    manifest = json.loads((job/'checked/manifest.json').read_text())
    gate = json.loads((release/'gate.json').read_text())
    sha = digest(job/'checked/student.model')
    if not (gate['schema'] == 'loupmd.gate/1' and gate['passed'] is True and gate['rows'] and
            all(row['passed'] is True for row in gate['rows']) and sha == gate['model']['sha256'] == manifest['student_sha256']):
        raise ValueError('checkpoint does not match a wholly passing native gate and training manifest')
    if (release/'student.loup.sha256').read_text().split() != [digest(release/'student.loup'), 'student.loup']:
        raise ValueError('released bundle does not match its named sidecar')
    def sources(record):
        return [(s['role'], s['sha256'], s['frames']) for s in record['sources']]
    if ([s['role'] for s in plan['sources']] != ['train','valid','teacher-test'] or sources(plan) != sources(manifest)
            or any(plan[k] != manifest[k] for k in ('recipe','mace_torch','teacher_head','teacher_sha256','resume'))):
        raise ValueError('release does not match this salt arm and frozen training recipe')
    return sha


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=['trajectory','mix'])
    print(admitted(Path.cwd(), parser.parse_args().arm))
