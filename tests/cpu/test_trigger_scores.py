"""Stage 0 ranking gates on numeric features; no model, torch or MD involved."""
import json

import numpy as np
import pytest

from loupmd.distill.trigger_scores import evaluate, fit, load_frame, main, ranking


def frame(directory, split, h, *, idx=0, energy=None, numbers=None, error=None,
          hidden=None, edges=None, distance=None, **metadata):
    directory.mkdir(exist_ok=True)
    h = np.atleast_2d(h).astype(float)
    n = len(h)
    ids = dict(frame_id=f'{split}:{idx}', trajectory_id=split, split=split,
               sampler='student' if split == 'fixed-student-test' else 'teacher',
               regime='target', prediction_sha256='a' * 64, teacher_sha256='b' * 64,
               prediction_head='Default', teacher_head='omat_pbe')
    ids.update(metadata)
    arrays = dict(h=h, numbers=np.ones(n, int) if numbers is None else numbers,
                  hidden=np.zeros((n, 16)) if hidden is None else hidden,
                  energy=np.zeros(n) if energy is None else energy,
                  edge_index=np.empty((2, 0), int) if edges is None else edges,
                  distance=np.array([]) if distance is None else distance,
                  cutoff=5., cutoff_p=5, ref_forces=np.zeros((n, 3)), **ids)
    if error is not None:
        arrays['force_error'] = error
    path = directory / f'{idx}.npz'
    np.savez(path, **arrays)
    return path


def test_gaussians_leverage_and_species_ranges(tmp_path):
    # Two occupied sites; their midpoint is the *single* Gaussian's exact mean.
    train = frame(tmp_path, 'train', [[x, 0] for x in (-10, -9.9, 9, 11, 0, 2)],
                  numbers=[1, 1, 1, 1, 8, 8], energy=[-2, -1, -1, -2, 5, 7])
    model = fit([load_frame(train, 'train')], clusters=2)
    hidden = np.zeros((7, 16)); hidden[3, 0] = 5
    path = frame(tmp_path, 'valid', [[.025, 0], [-9.75, 0], [10.2, 0], [10.2, 0],
                                     [9, 4], [1, 0], [15, 0]], idx=1,
                 hidden=hidden, error=np.zeros(7), numbers=[1, 1, 1, 1, 1, 8, 1],
                 energy=[1, -2, -1, -2, -2, 1, -1])
    scores = model.scores(load_frame(path, 'valid'))
    np.testing.assert_array_equal(scores['energy_range'], [2, 0, 0, 0, 0, 4, 0])
    assert scores['mahal_h'][0] < scores['mahal_h'][1]
    assert scores['cluster_h'][0] > scores['cluster_h'][1] > scores['cluster_h'][2]
    assert scores['llpr_hidden'][3] > scores['llpr_hidden'][4] == 0
    assert scores['llpr_h_hidden'][4] > scores['llpr_h_hidden'][2] > 0
    assert scores['norm_range'][0] > 0 and scores['norm_range'][6] > 0
    assert scores['norm_range'][2] == 0
    for b in ('0.5', '1'):
        for method in ('mahal', 'cluster'):
            key = f'{method}_h_b{b}'
            assert scores[key][4] > scores[key][2]
            assert scores[key + '_or_norm'][6] >= scores['norm_range'][6]


def test_weights_direction_shortest_and_isolated(tmp_path):
    train = frame(tmp_path, 'train', [[0]])
    model = fit([load_frame(train, 'train')])
    test = frame(tmp_path, 'valid', [[0]] * 4, idx=1, error=[0] * 4, energy=[1, 9, 2, 4],
                 edges=[[0, 1, 2], [2, 2, 1]], distance=[1., 4., 4.])
    scores = model.scores(load_frame(test, 'valid'))
    def envelope(x):
        return 1 - 21*x**5 + 35*x**6 - 15*x**7
    a, b = envelope(.2), envelope(.8)
    assert scores['neighbour_energy_range'] == pytest.approx([1, 2, (a + 9*b)/(a+b), 4])
    assert scores['short_distance'][2] > scores['short_distance'][1] > scores['short_distance'][3]


def test_equal_scores_are_one_decision_and_no_positive_is_not_perfect():
    scores, labels = np.array([3., 3., 1., 0.]), np.array([True, False, True, False])
    for order in (np.arange(4), [1, 0, 2, 3]):
        result = ranking(scores[order], labels[order], threshold=3.)
        assert result['average_precision'] == pytest.approx(.5 * .5 + .5 * 2/3)
        assert result['operating_point']['precision'] == result['operating_point']['recall'] == .5
    result = ranking(scores, np.zeros(4, bool), threshold=4.)
    assert result['average_precision'] is None
    assert result['operating_point']['recall'] is None
    assert result['operating_point']['precision'] is None


def sets(root):
    dirs = [root / split for split in ('train', 'valid', 'fixed-student-test')]
    frame(dirs[0], 'train', [[0], [1]])
    frame(dirs[1], 'valid', [[0], [1], [2]], energy=[1, 2, 4], error=[0, 0, .2])
    frame(dirs[2], 'fixed-student-test', [[0], [1], [2]], energy=[2, 4, 3], error=[0, 0, .2], regime='hot')
    return dirs


def test_fixed_labels_cannot_choose_operating_point(tmp_path):
    dirs = sets(tmp_path)
    output = tmp_path / 'report.json'
    main(['--train', str(dirs[0]), '--valid', str(dirs[1]), '--fixed', str(dirs[2]),
          '--error-eV-A', '.1', '--output', str(output)])
    report = json.loads(output.read_text())
    energy = report['candidates']['energy_range']
    assert energy['threshold'] == 4
    assert energy['fixed']['average_precision'] == .5
    point = energy['fixed']['operating_point']
    assert (point['tp'], point['fp'], point['fn']) == (0, 1, 1)
    assert energy['fixed_by_regime']['hot'] == energy['fixed']
    frame(dirs[2], 'fixed-student-test', [[0], [1], [2]], energy=[2, 4, 3], error=[0, 0, 0])
    changed = evaluate(*dirs, error_eV_A=.1)['candidates']
    assert all(changed[k]['threshold'] == v['threshold'] for k, v in report['candidates'].items())
    assert changed['energy_range']['fixed']['average_precision'] is None


def test_no_validation_positives_has_no_calibrated_threshold(tmp_path):
    dirs = sets(tmp_path)
    frame(dirs[1], 'valid', [[0]], error=[0])
    entry = evaluate(*dirs, error_eV_A=.1)['candidates']['energy_range']
    assert entry['threshold'] is None and entry['fixed']['operating_point'] is None
    assert entry['fixed']['average_precision'] == .5


@pytest.mark.parametrize('bad,match', [({'trajectory_id': 'valid'}, 'trajectory'),
                                      ({'frame_id': 'train:0'}, 'frame_id'),
                                      ({'sampler': 'teacher'}, 'sampler'),
                                      ({'prediction_sha256': 'c' * 64}, 'prediction_sha256'),
                                      ({'teacher_head': 'wrong'}, 'teacher_head')])
def test_split_and_model_provenance(tmp_path, bad, match):
    dirs = sets(tmp_path)
    frame(dirs[2], 'fixed-student-test', [[0]], error=[0], **bad)
    with pytest.raises(ValueError, match=match):
        evaluate(*dirs, error_eV_A=.1)
    with pytest.raises(ValueError, match='split'):
        evaluate(dirs[0], dirs[0], dirs[2], error_eV_A=.1)


def test_llpr_uses_full_uncentred_training_gram_not_cluster_sample(tmp_path):
    h = np.arange(1., 7.)[:, None]
    hidden = np.zeros((6, 16)); hidden[:, 0] = [0., 1., 2., 2., 1., 0.]
    train = frame(tmp_path, 'train', h, hidden=hidden, energy=[0, 2, 1, 8, 4, 3])
    model = fit([load_frame(train, 'train')], clusters=2, cluster_samples=2, ridge=.25)
    test = frame(tmp_path, 'valid', h[:3], idx=1, hidden=hidden[:3], energy=[0, 8, 7], error=[0]*3)
    scores = model.scores(load_frame(test, 'valid'))
    for key, x in [('llpr_hidden', hidden), ('llpr_h_hidden', np.column_stack((h, hidden)))]:
        expected = np.diag(x[:3] @ np.linalg.solve(x.T @ x + .25*np.eye(x.shape[1]), x[:3].T))
        np.testing.assert_allclose(scores[key], expected)
    np.testing.assert_array_equal(scores['energy_range'], np.zeros(3))
    assert model.fit_counts['1']['cluster_atoms'] == 2
    with pytest.raises(ValueError, match='train'):
        fit([load_frame(test, 'valid')])
