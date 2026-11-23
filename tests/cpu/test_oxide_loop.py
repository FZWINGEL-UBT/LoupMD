"""Closed-loop lineage checks independent of a CUDA device."""
import pytest
from ase import Atoms

from loupmd.distill.oxide_loop import audit_streams, decision, schedule, verify_parent


def frame(i, trajectory, seed):
    return Atoms('O', positions=[[i, 0, 0]], cell=[9, 9, 9], pbc=True,
                 info=dict(frame_id=str(i), trajectory_id=trajectory, md_seed=seed))


@pytest.mark.parametrize('collision', ['md_seed', 'trajectory_id', 'frame_id', 'geometry'])
def test_leakage_across_roles(collision):
    first, second = frame(1, 'a', 1), frame(2, 'b', 2)
    if collision == 'geometry':
        second.positions[:] = first.positions
    else:
        second.info[collision] = first.info[collision]
    with pytest.raises(ValueError, match=collision):
        audit_streams([('base', [first]), ('harvest', [second])])


def test_allows_separate_time_slices():
    assert audit_streams([('own', [frame(1, 'a', 1), frame(2, 'a', 1)])]) == {'own': 2}


def test_common_initial_stream_only_between_fixed_tests():
    pair = [('own:0', [frame(1, 'a', 1)]), ('own:1', [frame(2, 'b', 1)])]
    assert audit_streams(pair) == {'own:0': 1, 'own:1': 1}
    pair[1][1].append(frame(3, 'c', 1))
    with pytest.raises(ValueError, match='md_seed'):
        audit_streams(pair)


def test_upload_receipt_binds_actual_child_payload(tmp_path):
    import json
    from loupmd.distill.oxide_loop import verify_upload
    (tmp_path / 'paired-inputs.sha256').write_text('a' * 64 + '  /paired/pre_stage_two.pt\n')
    run = tmp_path / 'training'
    run.mkdir()
    def receipt(value):
        (run/'allocation.json').write_text(json.dumps({'input_sha256': {'pre_stage_two.pt': value}}))
    receipt('a' * 64)
    verify_upload(run)
    receipt('b' * 64)
    with pytest.raises(ValueError, match='uploaded'):
        verify_upload(run)


def test_native_replay_pass_does_not_excuse_bad_stored_training_label():
    from loupmd.distill.oxide_loop import check_stored, LIB_SHA, TEACHER_SHA
    row = dict(frame_id='harvest-a', historical={'passed': True})
    gate = dict(passed=True, library_sha256=LIB_SHA, teacher_sha256=TEACHER_SHA, teacher_head='omat_pbe',
                frames=1, source_sha256='data-hash', rows=[row])
    check_stored(gate, 'data-hash', ['harvest-a'])
    row['historical']['passed'] = False
    with pytest.raises(ValueError, match='stored'):
        check_stored(gate, 'data-hash', ['harvest-a'])


def test_protocol_streams_and_conditional_hot_round():
    seeds = [schedule(i, 'harvest')['seed'] for i in range(3)]
    assert len(set(seeds)) == 3 and 3101 not in seeds
    assert schedule(2, 'harvest')['hot_K'] > schedule(1, 'harvest')['hot_K']
    assert decision(1, True) == 'not_evaluated'
    assert decision(2, False) == 'closed'
    assert decision(2, True) == 'third_hot_round'
    assert decision(3, True) == 'stop_open'


def test_parent_rejects_obsolete_or_wrong_precision(tmp_path):
    import json
    from loupmd.distill.oxide_loop import BASE, RECIPE
    from loupmd.distill.train import file_digest
    (tmp_path / 'student.model').write_bytes(b'student')
    (tmp_path / 'pre_stage_two.pt').write_bytes(b'ema')
    sources = [dict(role=k, frames=v[0], sha256=v[1]) for k, v in BASE.items()]
    manifest = dict(schema=1, mace_torch='0.3.16', recipe={**RECIPE, 'dtype':'float64'}, sources=sources,
                    student_epoch=149, pre_stage_two_epoch=110, resume=None,
                    student_sha256=file_digest(tmp_path / 'student.model'),
                    pre_stage_two_sha256=file_digest(tmp_path / 'pre_stage_two.pt'))
    def save():
        (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    save()
    assert verify_parent(tmp_path, 'float64', []) == manifest
    with pytest.raises(ValueError, match='recipe'):
        verify_parent(tmp_path, 'float32', [])
    manifest['sources'][0]['sha256'] = 'obsolete'
    save()
    with pytest.raises(ValueError, match='sources'):
        verify_parent(tmp_path, 'float64', [])
