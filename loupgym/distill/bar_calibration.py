"""Teacher-only force scales, numerical resolution and p99.9 support on VALID.

Usage: python -m loupmd.distill.bar_calibration VALID.xyz VALID-gate.json OUTPUT.json
For reissued labels also pass --release-manifest base-labels-v2/manifest.json.
Never searches thresholds or measures student errors. The existing §4/§8
policies below are scale references, not fitted parameters. Student/observable
fidelity, including molecular force and torque, must be evaluated separately.
"""
import argparse
import json
from pathlib import Path

from ase.io import iread
import numpy as np

from loupmd.config import GATE_FACTOR, GATE_FORCE_FLOOR_EV_PER_A
from .train import file_digest

FORCE_FRACTION = 0.02
TAIL_MULTIPLIER = 5.0  # p99.9 atom-vector norm / pooled overall teacher component RMS bar
AUDIT_MULTIPLIER = 2.0  # frame component RMSE / same-frame teacher component RMS bar


def _scales(forces):
    sqnorm = np.einsum('ij,ij->i', forces, forces)
    n = len(sqnorm)
    rms = float(np.sqrt(sqnorm.mean() / 3))
    norms = np.sqrt(sqnorm)
    p999 = float(np.quantile(norms, .999, method='linear'))
    return dict(atom_observations=n, nominal_upper_0_1pct_count=n / 1000,
                teacher_component_rms=rms, component_bar=FORCE_FRACTION * rms,
                teacher_norm_p999=p999, teacher_norm_max=float(norms.max()),
                teacher_norm_p999_over_component_rms=p999 / rms if rms else None)


def _summary(frames, *, detail=False):
    forces = np.concatenate([a.arrays['REF_forces'] for a, _ in frames])
    species = np.concatenate([a.get_chemical_symbols() for a, _ in frames])
    total = _scales(forces)
    cap = TAIL_MULTIPLIER * total['component_bar']
    frame_rows = []
    for a, resolution in frames:
        rms = float(np.sqrt(np.mean(np.square(a.arrays['REF_forces']))))
        audit = AUDIT_MULTIPLIER * FORCE_FRACTION * rms
        df = resolution['native_df']
        frame_rows.append(dict(frame_id=a.info['frame_id'], teacher_component_rms=rms, audit_threshold=audit,
                               audit_noise_bound_fraction=df / audit if audit else None, **resolution))
    # Each row's df bounds *every* component on that frame, hence also its
    # component RMSE. sqrt(3) df bounds each atom norm; it does not change the cap.
    df_max = max(r['native_df'] for r in frame_rows)
    audits = [r['audit_threshold'] for r in frame_rows]
    audit_bounds = [r['audit_noise_bound_fraction'] for r in frame_rows]
    return dict(frames=len(frames), trajectories=len({a.info['trajectory_id'] for a, _ in frames}),
                overall=total, by_species={s: _scales(forces[species == s]) for s in np.unique(species)},
                pooled_tail_cap=cap, tail_noise_bound_fraction=np.sqrt(3) * df_max / cap if cap else None,
                audit_threshold_range=[min(audits), max(audits)],
                max_audit_noise_bound_fraction=max(audit_bounds) if None not in audit_bounds else None,
                **({'frame_scales': frame_rows} if detail else {}))


def calibrate(source, gate_path, release_manifest=None):
    """Bind the *stored* labels to the all-frame gate, including reissue lineage."""
    source_sha, gate_sha = file_digest(source), file_digest(gate_path)
    gate = json.loads(gate_path.read_text())
    frozen = gate['schema'] == 'loupmd.frozen-cell-gate.v1'
    if (not frozen and gate['schema'] != 'loupmd.water-stored-label-gate.v1') or gate['passed'] is not True:
        raise ValueError('need a passing frozen-cell or water stored-label gate')
    manifest_sha = None
    if release_manifest:
        manifest = json.loads(release_manifest.read_text())
        split = manifest['splits']['valid']
        if (not frozen or manifest['schema'] != 'loupmd.stage0-label-release.v2'
                or manifest['release'] != 'base-labels-v2'
                or (split['output_sha256'], split['gate_sha256'], split['source_sha256'], split['frames'])
                != (source_sha, gate_sha, gate['source_sha256'], gate['frames'])
                or any(manifest[k] != gate[k] for k in ('teacher_sha256', 'teacher_head', 'library_sha256'))):
            raise ValueError('release does not bind VALID to this gate')
        manifest_sha = file_digest(release_manifest)
    elif source_sha != gate['source_sha256']:
        raise ValueError('gate not bound to this source; reissued labels need their release manifest')

    if gate['frames'] != len(gate['rows']) or not gate['frames']:
        raise ValueError('gate must cover each VALID frame')
    groups, seen, traj_meta = {}, set(), {}
    for a, row in zip(iread(source, index=':', format='extxyz'), gate['rows'], strict=True):
        i = a.info
        if i['split'] != 'valid' or i['sampler'] != 'teacher' or 'STUDENT_forces' in a.arrays:
            raise ValueError('use unscored teacher VALID only')
        regime, tid, fid = i['regime'], i['trajectory_id'], i['frame_id']
        if (regime not in ('target', 'hot') or not tid or not fid or fid in seen
                or i['teacher_sha256'] != gate['teacher_sha256'] or i['sampler_sha256'] != i['teacher_sha256']
                or i['teacher_head'] != (gate['teacher_head'] if frozen else 'default')
                or i['sampler_head'] != i['teacher_head']):
            raise ValueError('invalid calibration provenance')
        seen.add(fid)
        meta = (regime, float(i['temperature_K']), int(i['md_seed']))
        if not np.isfinite(meta[1]) or meta[1] <= 0 or traj_meta.setdefault(tid, meta) != meta:
            raise ValueError('inconsistent trajectory provenance')
        ref = a.arrays['REF_forces']
        if not len(a) or ref.shape != (len(a), 3) or not np.isfinite(ref).all():
            raise ValueError('invalid teacher forces')
        repeat = None
        if frozen:
            if (row['frame_id'], row['trajectory_id'], row['regime']) != (fid, tid, regime):
                raise ValueError('gate geometry identity mismatch')
            if release_manifest:
                if (i.get('label_release') != manifest['release'] or i.get('label_library_sha256') != gate['library_sha256']
                        or not np.array_equal(ref, row['labels']['REF_forces'])):
                    raise ValueError('reissued reference differs from qualified labels')
                metric = row['sequential']
            else:
                metric = row['historical']  # directly serialized labels, not the newly evaluated path
            repeat = row['sequential_vs_fresh']['force_max_abs_eV_A']
            if not np.isfinite(repeat) or not 0 <= repeat <= GATE_FORCE_FLOOR_EV_PER_A:
                raise ValueError('native paths disagree')
        else:
            metric = row
        df, df32 = metric['df'], metric['df32']
        limit = max(GATE_FORCE_FLOOR_EV_PER_A, GATE_FACTOR * df32)
        if (metric['config'] != fid or metric['atoms'] != len(a) or metric['passed'] is not True
                or not np.isfinite([df, df32]).all() or min(df, df32) < 0 or df > limit):
            raise ValueError('stored-label gate failed or covers different data')
        groups.setdefault(regime, []).append((a, dict(native_df=df, stock_df32=df32, gate_limit=limit,
                                                       sequential_fresh_df=repeat)))
    if 'target' not in groups or len({m[2] for m in traj_meta.values()}) != len(traj_meta):
        raise ValueError('need target data and disjoint trajectory seeds')
    out = {}
    for regime, rows in groups.items():
        temperatures = {traj_meta[a.info['trajectory_id']][1] for a, _ in rows}
        if len(temperatures) != 1:
            raise ValueError('split distinct temperatures rather than pooling')
        out[regime] = dict(temperature_K=temperatures.pop(), pooled=_summary(rows, detail=True), by_trajectory={
            tid: _summary([r for r in rows if r[0].info['trajectory_id'] == tid])
            for tid in sorted({a.info['trajectory_id'] for a, _ in rows})})
    return dict(schema='loupmd.validation-force-resolution.v1', unit='eV/Angstrom',
                source_sha256=source_sha, gate_sha256=gate_sha, release_manifest_sha256=manifest_sha,
                teacher_sha256=gate['teacher_sha256'], reference_device=gate['reference_device'],
                force_fraction=FORCE_FRACTION, tail_multiplier=TAIL_MULTIPLIER, audit_multiplier=AUDIT_MULTIPLIER,
                note='Teacher norm quantiles are physical scales, not student errors. Counts above nominal quantile '
                     'are resolution estimates, not independent samples. Only the overall tail cap is a gate. '
                     'df fields are maxima over Cartesian components, not measured RMS errors.',
                regimes=out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'gate', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--release-manifest', type=Path)
    args = p.parse_args()
    result = calibrate(args.source, args.gate, args.release_manifest)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
