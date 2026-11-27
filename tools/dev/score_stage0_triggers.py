"""CPU-only Stage 0 trigger qualification on one model and one system.

Use trigger_features separately on train, scored valid, and target+hard in that
order. This command never acquires a GPU. See --help for inputs. No test labels
enter a fit, candidate choice, score threshold, force cutoff or audit default.
"""
import argparse
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase.io import iread
from loupgym.distill.trigger_scores import _Provenance, fit, ranking


class Identity:
    def __init__(self):
        self.frames = set(); self.geometries = set(); self.groups = {}

    def accept(self, atoms, group):
        info = atoms.info
        digest = hashlib.sha256()
        for arr in (atoms.numbers, atoms.cell.array, atoms.positions):
            digest.update(np.ascontiguousarray(arr, dtype='<f8').tobytes())
        for key, value, seen in [('frame_id', info['frame_id'], self.frames), ('geometry', digest.digest(), self.geometries)]:
            if value in seen:
                raise ValueError(f'duplicate {key} in {group}')
            seen.add(value)
        for key in ('trajectory_id', 'md_seed'):
            if self.groups.setdefault((key, info[key]), group) != group:
                raise ValueError(f'{key} crosses populations at {group}')


def check_frame(a, cap, split, sampler):
    info = a.info
    if info['split'] != split or info['sampler'] != sampler:
        raise ValueError(f'expected {split}/{sampler} frames')
    ref = a.arrays['REF_forces']
    if ref.shape != (len(a), 3) or not np.isfinite(ref).all():
        raise ValueError('invalid REF_forces')
    if cap is not None:
        keys = ['frame_id', 'trajectory_id', 'regime', 'teacher_sha256', 'teacher_head']
        if split != 'train':
            keys += ['prediction_sha256', 'prediction_head']
        for k in keys:
            if info[k] != cap[k]:
                raise ValueError(f'{k} differs between capture and extxyz')
        if not np.array_equal(a.numbers, cap['numbers']):
            raise ValueError('numbers differ between capture and extxyz')
    if split == 'train':
        return
    err = a.arrays['STUDENT_forces'] - ref
    norm = np.linalg.norm(err, axis=1)
    if err.shape != ref.shape or not np.isfinite(norm).all():
        raise ValueError('invalid STUDENT_forces')
    if cap is not None and not np.array_equal(norm, cap['force_error']):
        raise ValueError('force_error differs between capture and extxyz')
    ref_ss = float(np.vdot(ref, ref))
    if ref_ss == 0:
        raise ValueError('zero reference RMS: relative audit undefined')
    return norm, ref_ss, float(np.sqrt(np.vdot(err, err) / ref_ss))


def vec(rows, name=None):
    return np.concatenate([r[2] if name is None else r[3][name] for r in rows])


def calibrate(rows, cut):
    y = vec(rows) > cut
    reports = {k: ranking(vec(rows, k), y, calibrate=True) for k in rows[0][3]}
    if not y.any() or y.all():
        return None, reports
    return max(reports, key=lambda k: reports[k]['average_precision']), reports


def resamples(rows, count, seed):
    clusters = [[r for r in rows if r[0] == i] for i in sorted({r[0] for r in rows})]
    n = len(clusters)
    if n < 2:
        return
    indices = ((0, 0), (0, 1), (1, 0), (1, 1)) if n == 2 else np.random.default_rng(seed).integers(n, size=(count, n))
    for ix in indices:
        yield [r for j in ix for r in clusters[j]]


def leave_one(rows):
    for i in sorted({r[0] for r in rows}):
        sample = [r for r in rows if r[0] != i]
        if sample:
            yield sample


def interval(values):
    good = [v for v in values if v is not None]
    return dict(quantiles=np.quantile(good, [.025, .5, .975]).tolist() if good else None, defined=len(good), draws=len(values))


def residuals(rows):
    norms = vec(rows); bar = .02 * math.sqrt(sum(r[4] for r in rows) / (3 * len(norms)))
    p999 = float(np.quantile(norms, .999))
    return dict(frames=len(rows), trajectories=len({r[0] for r in rows}), force_bar=bar, p999=p999,
                maximum=float(norms.max()), tail_factor=p999/bar, max_frame_relative_rmse=max(r[5] for r in rows))


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def original_frames(raw, scored):
    for a, b in zip_longest(iread(raw, index=':', format='extxyz'), iread(scored, index=':', format='extxyz')):
        if a is None or b is None:
            raise ValueError('rescored frame count differs from frozen input')
        if any(a.info[k] != b.info[k] for k in ('frame_id','trajectory_id','md_seed','split','regime','teacher_sha256','teacher_head')):
            raise ValueError('rescored identities differ from frozen input')
        if any(not np.array_equal(u,v) for u,v in ((a.positions,b.positions),(a.cell,b.cell),(a.numbers,b.numbers),
               (a.info['REF_energy'],b.info['REF_energy']),(a.info['REF_stress'],b.info['REF_stress']))):
            raise ValueError('rescored geometry or reference differs from frozen input')
        if not np.allclose(a.arrays['REF_forces'],b.arrays['REF_forces'],rtol=0,atol=5.1e-9):
            raise ValueError('rescored reference force differs beyond extxyz rounding')
        yield b


def run(args, out):
    prov = _Provenance(); seen = Identity(); groups = {}; pins = {}
    manifest = json.loads(args.training_manifest.read_text()); gate = json.loads(args.student_gate.read_text())
    if not gate['passed'] or not gate['rows'] or gate['model']['sha256'] != manifest['student_sha256']:
        raise ValueError('missing passing gate for the training checkpoint')
    sources = {s['role']:s['sha256'] for s in manifest['sources']}
    for role, path in [('train',args.train),('valid',args.original_valid),('teacher-test',args.original_teacher)]:
        if file_sha(path) != sources[role]: raise ValueError(f'{role}: source does not match training manifest')
    pins.update(training_manifest=file_sha(args.training_manifest),student_gate=file_sha(args.student_gate),
                original_valid=sources['valid'],original_teacher=sources['teacher-test'])
    def frames(group):
        path = getattr(args, group); pins[group] = file_sha(path)
        original = {'valid':args.original_valid,'teacher':args.original_teacher}.get(group)
        source = original_frames(original, path) if original else iread(path, index=':', format='extxyz')
        for a in source:
            seen.accept(a, group)
            yield a

    def matched(group, captures, split, sampler):
        for a, cap in zip_longest(frames(group), captures):
            if a is None or cap is None:
                raise ValueError(f'{group}: capture count mismatch')
            stats = check_frame(a, cap, split, sampler)
            if cap['prediction_sha256'] != manifest['student_sha256'] or cap['teacher_sha256'] != manifest['teacher_sha256'] or cap['teacher_head'] != manifest['teacher_head']:
                raise ValueError('capture identity does not match trusted training manifest')
            yield cap, stats

    model = fit((f for f, _ in matched('train', prov.frames(args.train_features, 'train'), 'train', 'teacher')))
    def collect(group, split, sampler):
        rows = []
        for cap, (norm, ref_ss, ratio) in matched(group, prov.frames(getattr(args, group+'_features'), split), split, sampler):
            rows.append((cap['trajectory_id'], cap['regime'], norm, model.scores(cap), ref_ss, ratio))
        return rows
    valid = collect('valid', 'valid', 'teacher')
    clean = [r for r in valid if r[1] == 'target']
    if not clean:
        raise ValueError('need target validation frames to calibrate')
    bar = residuals(clean)['force_bar']
    cuts = args.error_eV_A or [3*bar, 5*bar]
    if len(set(cuts)) != len(cuts) or any(not np.isfinite(c) or c <= 0 for c in cuts):
        raise ValueError('event cutoffs must be distinct finite positive eV/A')
    lock = {}; chosen = {}; cal = {}
    for c in cuts:
        k, v = calibrate(valid, c); chosen[c] = k; cal[c] = v
        frame_q = lambda rs: float(np.quantile([r[3][k].max() for r in rs], .95))
        lock[c] = dict(candidate=k, valid=v, frame_max_q95=frame_q(clean) if k else None,
                       frame_max_leave_one=[frame_q(rs) for rs in leave_one(clean)] if k else [],
                       threshold_leave_one=[ranking(vec(rs,k),vec(rs)>c,calibrate=True)['threshold'] for rs in leave_one(valid)] if k else [])
    audit = math.ceil(100 * max(r[5] for r in valid)) / 100
    defaults = dict(audit_relative_rmse=audit, audit_leave_one=[math.ceil(100*max(r[5] for r in v))/100 for v in leave_one(valid)],
                    target_tail_multiplier=math.ceil(residuals(clean)['tail_factor']), target_force_bar_eV_A=bar)
    def dump(name, obj): (out/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    dump('locked-validation.json', dict(system=args.system, cutoffs=cuts, defaults=defaults, locks=lock, sha256=pins))
    # One sorted feature directory covering concatenated target then hard files.
    # Source membership, not the string 'hot', determines the hard population.
    def fixed_inputs():
        for g in ('target','hard'):
            for a in frames(g): yield g, a
    groups.update(valid=valid, valid_target=clean, target=[], hard=[], teacher=[])
    for record, cap in zip_longest(fixed_inputs(), prov.frames(args.fixed_features, 'fixed-student-test')):
        if record is None or cap is None: raise ValueError('fixed capture count mismatch')
        group, a = record; norm, ss, ratio = check_frame(a, cap, 'fixed-student-test', 'student')
        if any(a.info.get('sampler_'+k) != prov.contract['prediction_'+k] for k in ('sha256','head')):
            raise ValueError('fixed frames not visited by evaluated student')
        groups[group].append((cap['trajectory_id'], cap['regime'], norm, model.scores(cap), ss, ratio))
    for a in frames('teacher'):
        if any(a.info.get(k) != prov.contract[k] for k in ('teacher_sha256','teacher_head','prediction_sha256','prediction_head')):
            raise ValueError('teacher-test evaluated different models')
        norm, ss, ratio = check_frame(a, None, 'teacher-test', 'teacher')
        groups['teacher'].append((a.info['trajectory_id'], a.info['regime'], norm, {}, ss, ratio))
    if not all(groups.values()): raise ValueError('empty fixed or teacher population')
    groups['pooled'] = groups['target'] + groups['hard']
    reports = {}; uncertainty = {}; monitor = {}
    for c in cuts:
        k = chosen[c]; reports[c] = {}; uncertainty[c] = {}; monitor[c] = {}
        for g in ('target','hard','pooled'):
            rs = groups[g]; y = vec(rs)>c
            reports[c][g] = {n:ranking(vec(rs,n),y,threshold=v['threshold']) for n,v in cal[c].items()}
            if k is None or g=='pooled': continue
            draws = []
            for sample in resamples(rs, args.draws, args.seed):
                yb = vec(sample)>c; pred = ranking(vec(sample,k), yb, threshold=cal[c][k]['threshold'])
                ap = pred['average_precision']; baseline = ranking(vec(sample,'energy_range'), yb)['average_precision']; op = pred['operating_point']
                draws.append((ap-baseline if ap is not None else None, op['precision'], op['recall']))
            uncertainty[c][g] = {name:interval([d[i] for d in draws]) for i,name in enumerate(('paired_AP_delta','precision','recall'))}
            flags = np.array([r[3][k].max() >= lock[c]['frame_max_q95'] for r in rs]); bad = np.array([r[5]>audit for r in rs]); monitor[c][g] = dict(frames=len(rs), flagged=int(flags.sum()), audit_exceedances=int(bad.sum()), both=int((flags & bad).sum()))
    tails={}
    for g,rs in groups.items():
        if g=='pooled': continue
        d=residuals(rs); d.update(frames_over_audit=sum(r[5]>audit for r in rs), frames_over_4pct=sum(r[5]>.04 for r in rs),
            tail_factor_trajectory_interval=interval([residuals(q)['tail_factor'] for q in resamples(rs,args.draws,args.seed)])); tails[g]=d
    dump('evaluation.json', dict(system=args.system, sha256=pins, model=prov.contract, fit_by_Z=model.fit_counts,
         cutoffs_eV_A=cuts, candidates=reports, selected=chosen, uncertainty=uncertainty, frame_monitor=monitor,
         defaults=defaults, tails=tails, trajectory_resampling=dict(draws=args.draws, seed=args.seed, two='exact four draws', one='no interval'),
         scope='Offline single-system comparisons; pooled means target + hard within this system. No measured deployment cadence, overhead or cross-system transfer.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',required=True)
    for n in ('train','valid','teacher','target','hard'):
        p.add_argument('--'+n,type=Path,required=True,help='extxyz; scored by the evaluated student except train')
    for name in ('training-manifest','student-gate','original-valid','original-teacher'):
        p.add_argument('--'+name,type=Path,required=True)
    for n in ('train','valid','fixed'):
        p.add_argument('--'+n+'-features',type=Path,required=True)
    p.add_argument('--error-eV-A',type=float,action='append',help='repeat for primary/secondary events; default 3x and 5x 2%% target validation teacher RMS component')
    p.add_argument('--draws',type=int,default=300);p.add_argument('--seed',type=int,default=20261124)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.draws < 1 or args.seed < 0: p.error('invalid bootstrap controls')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        dest=Path(tmp)/'result'; dest.mkdir(); run(args,dest); dest.rename(args.output)


if __name__ == '__main__': main()
