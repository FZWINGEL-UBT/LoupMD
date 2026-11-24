import hashlib
import json
import shutil

import pytest

from loupmd.distill.salt_admission import admitted


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def issue(root, arm):
    j = root / (arm+'-s1-v1'); (j/'checked').mkdir(parents=True); (j/'release').mkdir()
    blob = arm.encode(); (j/'checked/student.model').write_bytes(blob)
    sources = [dict(role=k, sha256=sha((arm if k=='train' else k).encode()), frames=12) for k in ('train','valid','teacher-test')]
    plan = dict(sources=sources, recipe={'seed':1}, teacher_sha256='a'*64, teacher_head='omat_pbe', mace_torch='0.3.16', resume=None)
    (root/('train-plan.json' if arm=='trajectory' else 'train-mix-plan.json')).write_text(json.dumps(plan))
    (j/'checked/manifest.json').write_text(json.dumps(dict(plan, student_sha256=sha(blob))))
    gate = dict(schema='loupmd.gate/1', model={'sha256':sha(blob)}, passed=True, rows=[{'passed':True}]); (j/'release/gate.json').write_text(json.dumps(gate))
    (j/'release/student.loup').write_bytes(blob+b'bundle'); (j/'release/student.loup.sha256').write_text(sha(blob+b'bundle')+'  student.loup\n'); return j


@pytest.mark.parametrize('fault', ['other_gate','model','row','bundle','sidecar','other_arm','none'])
def test_admission(tmp_path, fault):
    j=issue(tmp_path,'trajectory'); other=issue(tmp_path,'mix')
    if fault=='other_gate':
        assert admitted(tmp_path,'mix') == sha(b'mix')
        shutil.copy(other/'release/gate.json',j/'release/gate.json')
    elif fault=='model': (j/'checked/student.model').write_bytes(b'changed')
    elif fault=='row':
        p=j/'release/gate.json'; g=json.loads(p.read_text()); g['rows'][0]['passed']=False; p.write_text(json.dumps(g))
    elif fault=='bundle': (j/'release/student.loup').write_bytes(b'changed')
    elif fault=='sidecar': (j/'release/student.loup.sha256').write_text(sha(b'x')+'  other.loup\n')
    elif fault=='other_arm':
        for directory in ('checked','release'): shutil.copytree(other/directory,j/directory,dirs_exist_ok=True)
    if fault=='none': assert admitted(tmp_path,'trajectory')==sha(b'trajectory')
    else:
        with pytest.raises(ValueError): admitted(tmp_path,'trajectory')
