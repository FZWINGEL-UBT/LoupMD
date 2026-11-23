"""Stage 0 precision screen on three complete runs. Does not certify MD observables."""
import argparse
import json
import math
from pathlib import Path


def read_run(path):
    return tuple(json.loads((path / f).read_text()) for f in (
        'checked/manifest.json', 'teacher-score.json', 'fidelity.json', 'release/gate.json'))


def compare(candidate, first, second):
    if candidate[0]['pair_id'] != first[0]['pair_id']:
        raise ValueError('unmatched same-seed precision pair')
    errors, bars, models = [], [], []
    reference = None
    for (manifest, test, own, gate), seed, dtype in zip((candidate, first, second), (1, 1, 2),
                                                       ('float32', 'float64', 'float64')):
        model = manifest['student_sha256']
        recipe = manifest['recipe']
        if (recipe['dtype'] != dtype or recipe['seed'] != seed or recipe['width'] != 64 or recipe['radial_head'] != 64
                or manifest['resume'] is not None or manifest['centered_before_cast'] != (dtype == 'float32')
                or manifest['student_epoch'] != recipe['epochs'] - 1
                or manifest['pre_stage_two_epoch'] != recipe['stage_two_start'] - 1):
            raise ValueError('expected the completed centered-FP32 and two independent stock FP64 s64l0 controls')
        sources = [(s['role'], s['sha256']) for s in manifest['sources']]
        if [s[0] for s in sources] != ['train', 'valid', 'teacher-test']:
            raise ValueError('expected cold starts on the three frozen splits')
        teacher = manifest['teacher_sha256'], manifest['teacher_head']
        frame_ids = [r['frame_id'] for r in test['frames']]
        identity = sources, teacher, {k: v for k, v in recipe.items() if k not in ('seed', 'dtype')}, frame_ids
        if not frame_ids or not all(frame_ids) or len(set(frame_ids)) != len(frame_ids):
            raise ValueError('missing or repeated teacher-test frame IDs')
        if reference is not None and identity != reference:
            raise ValueError('incomparable training labels, schedule or teacher-test frames')
        reference = identity
        for report, split in [(test, 'teacher-test'), (own, 'fixed-student-test')]:
            p = report['provenance']
            if (p['split'] != split or p['regimes'] != ['target'] or p['prediction_sha256'] != model
                    or (p['teacher_sha256'], p['teacher_head']) != teacher):
                raise ValueError(f'{split} report is not for this candidate and target teacher')
        if gate['model']['sha256'] != model or not (gate['passed'] and gate['rows'] and all(r['passed'] for r in gate['rows'])):
            raise ValueError('missing a passing 3x export gate for this exact checkpoint')
        error = test['forces']['overall']['rmse']
        if not math.isfinite(error) or error < 0:
            raise ValueError('invalid held-out force RMSE')
        errors.append(error)
        models.append(model)
        bars.append(test['label_metrics_passed'] and own['student_visited_label_bar'] == 'pass')
    low, high = sorted(errors[1:])
    within = low <= errors[0] <= high
    return {'metric': 'target teacher-test pooled Cartesian force RMSE (eV/Angstrom)',
            'order': ['centered FP32 seed 1', 'FP64 seed 1', 'FP64 seed 2'], 'models': models,
            'force_rmse': errors, 'fp64_seed_interval': [low, high], 'within_fp64_seed_spread': within,
            'teacher_and_own_label_bars': bars, 'precision_screen_passed': within and bars[0],
            'scope': 'numeric spread and label bars only; independent MD observables and time-to-fidelity still required'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('centered_fp32', 'fp64_seed1', 'fp64_seed2'):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(*(read_run(run) for run in (args.centered_fp32, args.fp64_seed1, args.fp64_seed2))), indent=2))
