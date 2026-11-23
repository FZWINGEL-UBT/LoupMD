import json

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from loupmd.distill.bar_calibration import calibrate
from loupmd.distill.train import file_digest


def inputs(tmp_path, *, split='valid', frozen=False, reissue=False):
    xyz, gate, release = (tmp_path / name for name in ('frames.xyz', 'gate.json', 'manifest.json'))
    frames, rows = [], []
    for k in (1, 2):
        tid, fid = f't{k}', f'f{k}'
        a = Atoms('HO', [(0, 0, 0), (1, 0, 0)], cell=[4, 4, 4], pbc=True)
        a.info.update(split=split, sampler='teacher', teacher_head='default', sampler_head='default',
                      teacher_sha256='teacher', sampler_sha256='teacher', md_seed=k, regime='target',
                      temperature_K=300, trajectory_id=tid, frame_id=fid)
        if reissue:
            a.info.update(label_release='base-labels-v2', label_library_sha256='library')
        f = np.array([[k, k, k], [0, 0, 0]], dtype=float)
        a.set_array('REF_forces', f)
        frames.append(a)
        metric = dict(config=fid, atoms=2, passed=True, df=2e-5, df32=1e-5)
        if frozen:
            rows.append(dict(frame_id=fid, regime='target', trajectory_id=tid, sequential=metric,
                             historical=dict(metric, df=1.5e-5), sequential_vs_fresh={'force_max_abs_eV_A': 1e-6},
                             labels={'REF_forces': f.tolist()}))
        else:
            rows.append(metric)
    write(xyz, frames, format='extxyz')
    digest = file_digest(xyz)
    rec = dict(schema='loupmd.frozen-cell-gate.v1' if frozen else 'loupmd.water-stored-label-gate.v1',
               teacher_sha256='teacher', teacher_head='default', frames=2, rows=rows, passed=True,
               source_sha256='old' if reissue else digest, library_sha256='library', reference_device='cpu')
    gate.write_text(json.dumps(rec))
    if reissue:
        release.write_text(json.dumps(dict(schema='loupmd.stage0-label-release.v2', release='base-labels-v2',
                                           teacher_sha256='teacher', teacher_head='default', library_sha256='library',
                                           splits={'valid': dict(frames=2, output_sha256=digest, source_sha256='old',
                                                                 gate_sha256=file_digest(gate))})))
    return xyz, gate, release if reissue else None


@pytest.mark.parametrize(('frozen', 'reissue', 'native_df'), [(False, False, 2e-5), (True, False, 1.5e-5),
                                                               (True, True, 2e-5)])
def test_gate_paths_and_force_scales(tmp_path, frozen, reissue, native_df):
    args = inputs(tmp_path, frozen=frozen, reissue=reissue)
    target = calibrate(*args)['regimes']['target']
    p = target['pooled']
    assert len(target['by_trajectory']) == p['trajectories'] == 2
    assert p['overall']['atom_observations'] == 4
    assert p['by_species']['H']['nominal_upper_0_1pct_count'] == .002
    assert p['by_species']['O']['teacher_norm_p999_over_component_rms'] is None
    assert p['by_species']['H']['teacher_component_rms'] == pytest.approx(np.sqrt(2.5))
    assert p['overall']['teacher_norm_p999'] == pytest.approx(1.997 * np.sqrt(3))
    assert p['pooled_tail_cap'] == pytest.approx(.1 * np.sqrt(1.25))
    assert p['audit_threshold_range'] == pytest.approx([.04 / np.sqrt(2), .08 / np.sqrt(2)])
    assert p['tail_noise_bound_fraction'] == pytest.approx(np.sqrt(3) * native_df / p['pooled_tail_cap'])
    assert p['max_audit_noise_bound_fraction'] == pytest.approx(native_df / p['audit_threshold_range'][0])


@pytest.mark.parametrize('split', ['teacher-test', 'fixed-student-test', 'train'])
def test_no_performance_or_training_frames(tmp_path, split):
    with pytest.raises(ValueError, match='VALID only'):
        calibrate(*inputs(tmp_path, split=split))


def test_no_unbound_reissued_labels(tmp_path):
    xyz, gate, release = inputs(tmp_path, frozen=True, reissue=True)
    with pytest.raises(ValueError, match='release manifest'):
        calibrate(xyz, gate)
    gate.write_text(gate.read_text() + '\n')
    with pytest.raises(ValueError, match='does not bind'):
        calibrate(xyz, gate, release)


def test_rejects_failed_stored_labels_even_if_rerun_passed(tmp_path):
    xyz, gate, _ = inputs(tmp_path, frozen=True)
    rec = json.loads(gate.read_text())
    rec['rows'][1]['historical']['passed'] = False
    gate.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match='stored-label gate failed'):
        calibrate(xyz, gate)
