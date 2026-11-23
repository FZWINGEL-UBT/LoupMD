from copy import deepcopy
from hashlib import sha256

import pytest

from tools.dev.oxide_precision import compare


def evidence(seed, dtype, error):
    identity = f'{dtype}:{seed}'
    recipe = {'epochs': 150, 'stage_two_start': 111, 'seed': seed, 'dtype': dtype,
              'width': 64, 'radial_head': 64, 'cutoff': 5., 'batch_size': 4}
    sources = [{'role': r, 'sha256': r + '-v2'} for r in ('train', 'valid', 'teacher-test')]
    # Training's order-sensitive ID excludes dtype but includes the seed.
    pair_id = sha256(str((sources, {k: v for k, v in recipe.items() if k != 'dtype'})).encode()).hexdigest()
    manifest = {'recipe': recipe, 'sources': sources, 'student_sha256': identity, 'student_epoch': 149,
                'pre_stage_two_epoch': 110, 'teacher_sha256': 'mh1', 'teacher_head': 'omat_pbe',
                'centered_before_cast': dtype == 'float32', 'resume': None, 'pair_id': pair_id}
    prov = {'teacher_sha256': 'mh1', 'teacher_head': 'omat_pbe', 'prediction_sha256': identity,
            'regimes': ['target'], 'split': 'teacher-test'}
    test = {'provenance': prov, 'frames': [{'frame_id': 'fixed-target:50'}], 'label_metrics_passed': True,
            'forces': {'overall': {'rmse': error}}}
    own = deepcopy(test)
    own['provenance']['split'] = 'fixed-student-test'
    own['student_visited_label_bar'] = 'pass'
    gate = {'model': {'sha256': identity}, 'rows': [{'passed': True}], 'passed': True}
    return manifest, test, own, gate


def trio():
    return evidence(1, 'float32', .009), evidence(1, 'float64', .008), evidence(2, 'float64', .010)


def test_seed_interval_is_inclusive_but_never_overrides_absolute_bar():
    candidate, first, second = trio()
    candidate[1]['forces']['overall']['rmse'] = .010
    assert compare(candidate, first, second)['precision_screen_passed']
    candidate[2]['student_visited_label_bar'] = 'fail'
    assert not compare(candidate, first, second)['precision_screen_passed']
    candidate[2]['student_visited_label_bar'] = 'pass'
    candidate[1]['forces']['overall']['rmse'] = .0101
    assert not compare(candidate, first, second)['within_fp64_seed_spread']


@pytest.mark.parametrize('fault', ['test_set', 'labels', 'checkpoint', 'same_seed', 'graph_shape', 'failed_gate', 'pair'])
def test_incomparable_precision_evidence_is_rejected(fault):
    candidate, first, second = trio()
    if fault == 'test_set':
        second[1]['frames'][0]['frame_id'] = 'different geometry'
    elif fault == 'labels':
        first[0]['sources'][0]['sha256'] = 'obsolete'
    elif fault == 'checkpoint':
        candidate[3]['model']['sha256'] = 'other'
    elif fault == 'same_seed':
        second[0]['recipe']['seed'] = 1
    elif fault == 'graph_shape':
        first[0]['recipe']['radial_head'] = 8
    elif fault == 'failed_gate':
        second[3]['rows'][0]['passed'] = False
    elif fault == 'pair':
        first[0]['pair_id'] = 'different same-seed run'
    with pytest.raises(ValueError):
        compare(candidate, first, second)
