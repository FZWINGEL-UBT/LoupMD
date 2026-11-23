import importlib.util
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms


path = Path(__file__).resolve().parents[2] / 'tools/dev/score_stage0_triggers.py'
spec = importlib.util.spec_from_file_location('stage0_trigger_recipe', path)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


def frame():
    a = Atoms('HO', positions=[[0, 0, 0], [0, 0, 1]], cell=[5]*3, pbc=True)
    a.info.update(frame_id='valid:1', trajectory_id='v', md_seed=21, split='valid', sampler='teacher', regime='target',
                  teacher_head='teacher', teacher_sha256='a'*64, prediction_head='Default', prediction_sha256='b'*64)
    a.arrays['REF_forces'] = np.ones((2, 3)); a.arrays['STUDENT_forces'] = np.zeros((2, 3))
    return a


def test_capture_must_match_actual_scored_frame():
    a = frame(); cap = {**a.info, 'force_error': np.full(2, np.sqrt(3)), 'numbers': a.numbers}
    np.testing.assert_array_equal(recipe.check_frame(a, cap, 'valid', 'teacher')[0], cap['force_error'])
    for k, bad in [('frame_id', 'other'), ('prediction_sha256', 'c'*64), ('force_error', np.zeros(2))]:
        with pytest.raises(ValueError, match=k):
            recipe.check_frame(a, {**cap, k: bad}, 'valid', 'teacher')


def test_reject_geometry_or_seed_shared_across_populations():
    a = frame(); used = recipe.Identity(); used.accept(a, 'valid'); b = a.copy()
    b.info.update(frame_id='fixed:1', trajectory_id='f', md_seed=22)
    with pytest.raises(ValueError, match='geometry'): used.accept(b, 'hard')
    b.positions += 0.25; b.info.update(md_seed=21, frame_id='fixed:2')
    with pytest.raises(ValueError, match='md_seed'): used.accept(b, 'hard')


def test_no_positives_do_not_select_and_ties_stable():
    rows = [('v', 'target', np.array([0., .01]), {'energy_range': np.zeros(2), 'mahal_h': np.ones(2)}, 1., .01)]
    name, reports = recipe.calibrate(rows, .1)
    assert name is None and reports['mahal_h']['threshold'] is None
    assert recipe.calibrate(rows, .005)[0] == 'energy_range'


def test_single_trajectory_no_bootstrap_precision_claim():
    rows = [('v', 'target', np.ones(2), {}, 1., .01)]
    assert list(recipe.resamples(rows, 10, 3)) == []
    assert recipe.interval([])['quantiles'] is None


def test_all_validation_atoms_positive_is_uninformative():
    rows = [('v', 'target', np.ones(2), {'energy_range': np.zeros(2)}, 1., .01)]
    name, reports = recipe.calibrate(rows, .1)
    assert name is None and reports['energy_range']['prevalence'] == 1.
