"""The corrected learning curve must never trust old labels or a mismatched gate."""
import hashlib
import io
import json

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from loupmd.distill import reissue
from tools.dev import issue_curve_labels as release


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def test_staging_ignores_no_unreceipted_bytes(tmp_path):
    path = tmp_path / 'part-00.xyz'
    path.write_bytes(b'payload')
    path.with_suffix('.sha256').write_text(f'{sha(b"payload")}  part-00.xyz\n')
    assert release.pinned_shard(path, sha(b'payload')) == b'payload'
    with pytest.raises(ValueError):
        release.pinned_shard(path, sha(b'unreceipted'))
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        release.pinned_shard(path, sha(b'payload'))


@pytest.fixture
def group(tmp_path, monkeypatch):
    for name in ('work', 'issued', 'gates'):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(release, 'SHARD_FRAMES', 1)
    monkeypatch.setattr(release, 'GROUP', 1)
    a = Atoms('SrTiO3', positions=np.arange(15).reshape(5, 3), cell=np.eye(3)*20, pbc=True)
    a.set_momenta(a.positions/20)
    a.new_array('REF_forces', np.zeros((5, 3)))
    a.info.update(frame_id='frame', trajectory_id='traj', md_seed=341, split='train', sampler='teacher',
                  teacher_sha256=reissue.TEACHER_SHA256, teacher_head='omat_pbe', sampler_head='omat_pbe',
                  sampler_sha256=reissue.TEACHER_SHA256, replica=0, seed=7901, burn_in=1000, stride=1,
                  step=1001, regime='target', temperature_K=300, integrator='langevin', engine_build='frozen',
                  REF_energy=0., REF_stress=np.eye(3))
    data = io.StringIO(); write(data, a, format='extxyz'); payload = data.getvalue().encode()
    (tmp_path / 'work/group-0.xyz').write_bytes(payload)
    item = dict(sha256=sha(payload), sampler='frozen')
    cat = dict(groups={'0': sha(payload)}, shards={'0': item})
    (tmp_path / 'catalog.json').write_text(json.dumps(cat))
    metric = dict(atoms=5, passed=True, failed=[])
    row = dict(frame_id='frame', sequential=metric, fresh=metric, historical=dict(passed=False),
               sequential_vs_fresh={k: 0. for k in reissue.NATIVE_AGREEMENT_LIMITS},
               labels=dict(REF_energy=1.234567890123, REF_forces=a.positions.tolist(), REF_stress=np.eye(3).tolist()))
    report = dict(schema='loupmd.frozen-cell-gate.v1', passed=True, frames=1, source_sha256=sha(payload),
                  teacher_sha256=reissue.TEACHER_SHA256, teacher_head='omat_pbe', library_version='fixed',
                  library_sha256=release.LIBS['leeloo'], bundle_sha256=release.BUNDLE, rows=[row])
    return tmp_path, report


@pytest.mark.parametrize('fault', ('none', 'source', 'library', 'bundle', 'history', 'individual', 'overall'))
def test_positive_gate_required_for_each_issued_group(group, fault):
    root, report = group
    if fault == 'source': report['source_sha256'] = '0'*64
    if fault == 'library': report['library_sha256'] = '0'*64
    if fault == 'bundle': report['bundle_sha256'] = '0'*64
    if fault == 'history': report['rows'][0]['sequential_vs_fresh']['force_max_abs_eV_A'] = 1.
    if fault == 'individual': report['rows'][0]['fresh']['passed'] = False
    if fault == 'overall': report['passed'] = False
    (root / 'gates/group-0.json').write_text(json.dumps(report))
    if fault != 'none':
        with pytest.raises(ValueError): release.issue(root, 0)
        assert not (root / 'issued/group-0').exists()
        return
    release.issue(root, 0)
    part = root / 'issued/group-0/part-00.xyz'
    result = read(part)
    assert result.info['label_release'] == release.RELEASE
    assert result.info['engine_build'] == 'frozen'
    assert result.info['REF_energy'] == report['rows'][0]['labels']['REF_energy']
    assert sha(part.read_bytes()) in part.with_suffix('.sha256').read_text()
    with pytest.raises(FileExistsError): release.issue(root, 0)


@pytest.mark.parametrize('fault', ('none', 'wrong-pin', 'unexpected-env', 'different-source', 'not-rejected'))
def test_exact_rerun_links_original_failure_and_certified_build(group, fault):
    root, gate = group
    old = dict(gate, passed=False)
    if fault == 'different-source':
        old['source_sha256'] = '9'*64
    if fault == 'not-rejected':
        old['passed'] = True
    negative = root/'work/rejected-0.json'
    negative.write_text(json.dumps(old))
    lib = root/'exact.so'
    lib.write_bytes(b'exact table library')
    gate['library_sha256'] = sha(lib.read_bytes())
    gate['native_environment'] = {'LOUP_NO_HYBRID': '1'} if fault == 'unexpected-env' else {}
    (root/'gates/group-0.json').write_text(json.dumps(gate))
    evidence = dict(gate, source_sha256=release.DIAGNOSTIC_SHA, frames=2,
                    rows=[dict(gate['rows'][0], replica=i) for i in (364, 365)])
    evidence['native_environment'] = {}
    evidence_path = root/'diagnostic.json'
    evidence_path.write_text(json.dumps(evidence))
    if fault == 'wrong-pin':
        with pytest.raises(ValueError):
            release.register_exact(root, 'leeloo', lib, '9'*64, evidence_path)
        return
    release.register_exact(root, 'leeloo', lib, sha(lib.read_bytes()), evidence_path)
    if fault != 'none':
        with pytest.raises(ValueError):
            release.issue(root, 0)
        return
    release.issue(root, 0)
    result = json.loads((root/'issued/group-0/receipt.json').read_bytes())
    assert result['label_mode'] == 'exact'
    assert result['default_rejection_sha256'] == sha(negative.read_bytes())
    assert result['native_environment'] == {}
    assert result['conv_table'] == 'fp32'
    with pytest.raises(FileExistsError):
        release.register_exact(root, 'leeloo', lib, sha(lib.read_bytes()), evidence_path)


@pytest.mark.parametrize('workers', (0, 5))
def test_fresh_worker_fanout_is_bounded_before_gpu_access(workers):
    from tools.dev.qualify_frozen_cells import qualify
    with pytest.raises(ValueError, match='fresh workers'):
        qualify(None, None, workers)


def test_only_known_automatic_native_cache_paths_are_allowed(tmp_path):
    cache = tmp_path.parent/'curve1/cache-leeloo'
    env = {'LOUP_CACHE_DIR': str(cache), 'LOUP_KERNEL_PATH': str(cache/'lib')}
    assert release.no_arithmetic_override(tmp_path, 'leeloo', env)
    assert not release.no_arithmetic_override(tmp_path, 'coraline', env)
    assert not release.no_arithmetic_override(tmp_path, 'leeloo', env | {'LOUP_OWN_EXT': '0'})
    assert not release.no_arithmetic_override(tmp_path, 'leeloo', {'LOUP_KERNEL_PATH': '/untrusted'})
