"""Rank Stage 0 per-atom triggers on NPZ captures, without loading either model.

python -m loupmd.distill.trigger_scores --train features/train --valid features/valid \
    --fixed features/fixed-student-test --error-eV-A 0.1 --output trigger.json

The error cutoff is a scientific input, not estimated from the test distribution.
Each validation F1 operating point is frozen before any fixed frame is read. Only
cluster fits subsample training atoms, via a uniform seeded reservoir per species.
Ranking is exact: temporary scalar score columns on disk, one sort per candidate
and population in memory, never an in-memory corpus of features or edge lists.
"""
import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np


_SPLITS = {'train': 'teacher', 'valid': 'teacher', 'fixed-student-test': 'student'}
_IDENT = ('teacher_sha256', 'teacher_head', 'prediction_sha256', 'prediction_head')
_FORMS = ('h', 'h_b0.5', 'h_b1')
_EPS = 1e-12


def load_frame(path, split):
    """Read one capture with pickle disabled. Never read training force errors."""
    with np.load(path, allow_pickle=False) as data:
        def get(key):
            if key not in data:
                raise ValueError(f'{path}: missing {key}')
            return data[key]

        frame = {}
        for key in ('frame_id', 'trajectory_id', 'regime', 'split', 'sampler', *_IDENT):
            value = get(key)
            if value.shape != () or value.dtype.kind != 'U' or not value.item().strip():
                raise ValueError(f'{path}: expected nonempty scalar string {key}')
            frame[key] = value.item()
        if split not in _SPLITS or frame['split'] != split:
            raise ValueError(f"{path}: wrong split {frame['split']!r}, expected {split!r}")
        if frame['sampler'] != _SPLITS[split]:
            raise ValueError(f'{path}: wrong sampler for {split}')
        for key in ('teacher_sha256', 'prediction_sha256'):
            if not re.fullmatch(r'[0-9a-fA-F]{64}', frame[key]):
                raise ValueError(f'{path}: invalid {key}')
            frame[key] = frame[key].lower()

        def array(key, shape, integer=False):
            a = get(key)
            if (a.dtype.kind not in ('iu' if integer else 'fiu') or a.ndim != len(shape)
                    or any(s is not None and actual != s for actual, s in zip(a.shape, shape))
                    or not np.isfinite(a).all()):
                raise ValueError(f'{path}: invalid {key} (expected shape {shape}, finite values)')
            return np.asarray(a, dtype=np.int64 if integer else np.float64)

        z = array('numbers', (None,), True)
        n = len(z)
        h = array('h', (n, None))
        if not n or np.any(z < 1) or not h.shape[1]:
            raise ValueError(f'{path}: empty features or invalid atomic numbers')
        edges = array('edge_index', (2, None), True)
        distance = array('distance', (edges.shape[1],))
        cutoff = float(array('cutoff', ()))
        p = int(array('cutoff_p', (), True))
        if cutoff <= 0 or p < 1 or np.any(distance < 0) or np.any(edges < 0) or np.any(edges >= n):
            raise ValueError(f'{path}: invalid graph or cutoff')
        frame.update(numbers=z, h=h, hidden=array('hidden', (n, 16)), energy=array('energy', (n,)),
                     edge_index=edges, distance=distance, cutoff=cutoff, cutoff_p=p)
        array('ref_forces', (n, 3))  # validate units' carrier; not a predictor or calibration target
        if split != 'train':
            error = array('force_error', (n,))
            if np.any(error < 0):
                raise ValueError(f'{path}: negative force_error norm')
            frame['force_error'] = error
        return frame


def _variants(h, norm):
    yield h
    yield h / (np.sqrt(norm)[:, None] + _EPS)
    yield h / (norm[:, None] + _EPS)


class _Moments:
    """Merge centred batch scatters; stable even for features with large means."""
    def __init__(self, width):
        self.n = 0
        self.mean = np.zeros(width)
        self.scatter = np.zeros((width, width))

    def update(self, x):
        delta = x.mean(axis=0) - self.mean
        new_n = self.n + len(x)
        centred = x - (self.mean + delta)
        self.scatter += centred.T @ centred + np.outer(delta, delta) * (self.n * len(x) / new_n)
        self.mean += delta * (len(x) / new_n)
        self.n = new_n


def _whitener(matrix, ridge):
    eigen, axes = np.linalg.eigh(matrix)
    return axes / np.sqrt(np.maximum(eigen, 0.) + ridge)


def _quadratic(x, whitener):
    projected = x @ whitener
    return np.einsum('ij,ij->i', projected, projected)


@dataclass
class _Gaussian:
    mean: np.ndarray
    whitener: np.ndarray

    @classmethod
    def from_stats(cls, moments, ridge):
        return cls(moments.mean, _whitener(moments.scatter / moments.n, ridge))

    def distance(self, x):
        return np.sqrt(_quadratic(x - self.mean, self.whitener))


def _nearest(x, centres):
    # Do not materialize an atoms x clusters x channels tensor.
    def squared(c):
        d = x - c
        return np.einsum('ij,ij->i', d, d)
    best = squared(centres[0]); labels = np.zeros(len(x), dtype=int)
    for k, centre in enumerate(centres[1:], 1):
        dist = squared(centre)
        closer = dist < best
        labels[closer] = k
        np.minimum(best, dist, out=best)
    return labels, best


class _Clusters:
    def __init__(self, x, count, ridge):
        # Farthest-first seeding followed by Lloyd updates. Repeated/identical sites
        # produce fewer centres, rather than a singular empty-cluster covariance.
        centres = [x[_nearest(x, [x.mean(axis=0)])[1].argmin()]]
        _, distances = _nearest(x, centres)
        for _ in range(1, min(len(x), count)):
            far = distances.argmax()
            if distances[far] == 0.:
                break
            centres.append(x[far])
            np.minimum(distances, _nearest(x, centres[-1:])[1], out=distances)
        for _ in range(50):
            labels, _ = _nearest(x, centres)
            updated = [x[labels == k].mean(axis=0) for k in np.unique(labels)]
            if np.array_equal(centres, updated):
                break
            centres = updated
        labels, _ = _nearest(x, centres)
        self.sites = []
        for k in np.unique(labels):
            site = x[labels == k]
            moments = _Moments(x.shape[1]); moments.update(site)
            gaussian = _Gaussian.from_stats(moments, ridge)
            # The same trigger level means the same multiple of a site's own P95,
            # not the radius of a different coordination state of the same species.
            spread = max(float(np.quantile(gaussian.distance(site), .95)), _EPS)
            self.sites.append((gaussian, spread))

    def distance(self, x):
        labels, _ = _nearest(x, [g.mean for g, _ in self.sites])
        result = np.empty(len(x))
        for k in np.unique(labels):
            mask = labels == k
            gaussian, spread = self.sites[k]
            result[mask] = gaussian.distance(x[mask]) / spread
        return result


class _SpeciesStats:
    def __init__(self, width, seed):
        self.moments = [_Moments(width) for _ in _FORMS]
        self.norm = _Moments(1)
        self.lower = np.full(2, np.inf); self.upper = -self.lower
        self.sample = np.empty((0, width)); self.keys = np.empty(0)
        self.rng = np.random.default_rng(seed)

    def update(self, h, energy, cap):
        norm = np.linalg.norm(h, axis=1)
        np.minimum(self.lower, [energy.min(), norm.min()], out=self.lower)
        np.maximum(self.upper, [energy.max(), norm.max()], out=self.upper)
        self.norm.update(norm[:, None])
        for moments, x in zip(self.moments, _variants(h, norm)):
            moments.update(x)
        # Bottom-k iid priorities are a uniform sample of all atoms seen so far.
        keys = self.rng.random(len(h))
        if len(self.keys) == cap:
            take = keys < self.keys.max()
            if not take.any():
                return
            h, keys = h[take], keys[take]
        self.sample = np.concatenate((self.sample, h)); self.keys = np.concatenate((self.keys, keys))
        if len(self.keys) > cap:
            keep = np.argpartition(self.keys, cap - 1)[:cap]
            self.sample, self.keys = self.sample[keep], self.keys[keep]


@dataclass
class _SpeciesModel:
    lower: np.ndarray
    upper: np.ndarray
    norm_sd: float
    gaussians: list
    clusters: list

    def scores(self, h, energy):
        def excess(x, index):
            return np.maximum(0., np.maximum(self.lower[index] - x, x - self.upper[index]))
        norm = np.linalg.norm(h, axis=1)
        norm_score = excess(norm, 1) / self.norm_sd
        yield 'energy_range', excess(energy, 0)
        yield 'norm_range', norm_score
        for form, x, gaussian, clusters in zip(_FORMS, _variants(h, norm), self.gaussians, self.clusters):
            single = gaussian.distance(x)
            multiple = clusters.distance(x)
            yield 'mahal_' + form, single
            yield 'cluster_' + form, multiple
            if form != 'h':
                # Two independently interpretable alarms: RMS whitened coordinates
                # (or site P95 units), versus training SDs outside the norm range.
                yield 'mahal_' + form + '_or_norm', np.maximum(single / np.sqrt(x.shape[1]), norm_score)
                yield 'cluster_' + form + '_or_norm', np.maximum(multiple, norm_score)


@dataclass
class _Model:
    species: dict
    leverage: np.ndarray
    hidden_leverage: np.ndarray
    fit_counts: dict

    def scores(self, frame):
        h, z, hidden = (frame[k] for k in ('h', 'numbers', 'hidden'))
        n = len(z)
        raw = {'llpr_h_hidden': _quadratic(np.column_stack((h, hidden)), self.leverage),
               'llpr_hidden': _quadratic(hidden, self.hidden_leverage)}
        for species in np.unique(z):
            if species not in self.species:
                raise ValueError(f'{frame["frame_id"]}: species {species} not in training set')
            mask = z == species
            for key, values in self.species[species].scores(h[mask], frame['energy'][mask]):
                if key not in raw:
                    raw[key] = np.empty(n)
                raw[key][mask] = values

        send, recv = frame['edge_index']; cutoff = frame['cutoff']; p = frame['cutoff_p']
        distance = np.minimum(frame['distance'], cutoff)
        shortest = np.full(n, cutoff)
        np.minimum.at(shortest, recv, distance)
        raw['short_distance'] = cutoff - shortest  # larger always means more anomalous
        x = distance / cutoff
        # MACE polynomial envelope in factored form to avoid cancellation near r_c.
        coefficients = [(i + 1) * (i + 2) / 2 for i in range(p)]
        w = (1 - x)**3 * np.polynomial.polynomial.polyval(x, coefficients)
        total = np.bincount(recv, weights=w, minlength=n)
        averaged = {}
        for key, values in raw.items():
            numerator = np.bincount(recv, weights=w * values[send], minlength=n)
            averaged['neighbour_' + key] = np.divide(numerator, total, out=values.copy(), where=total > 0)
        return {**raw, **averaged}


def fit(frames, *, clusters=4, cluster_samples=4096, ridge=1e-6, seed=0):
    """All-atom range, covariance and LLPR Gram fits; bounded cluster sample only."""
    if not 1 <= clusters <= cluster_samples or not np.isfinite(ridge) or ridge <= 0 or seed < 0:
        raise ValueError('need 1 <= clusters <= cluster_samples, positive finite ridge, nonnegative seed')
    stats = {}; gram = None
    for frame in frames:
        if frame['split'] != 'train':
            raise ValueError('fit accepts only the train split')
        h, z = frame['h'], frame['numbers']
        f = np.column_stack((h, frame['hidden']))
        if gram is None:
            gram = np.zeros((f.shape[1], f.shape[1]))
        gram += f.T @ f
        for species in np.unique(z):
            if species not in stats:
                stats[species] = _SpeciesStats(h.shape[1], [seed, int(species)])
            mask = z == species
            stats[species].update(h[mask], frame['energy'][mask], cluster_samples)
    if gram is None:
        raise ValueError('empty train split')
    models = {}; counts = {}
    for species, s in stats.items():
        norms = np.linalg.norm(s.sample, axis=1)
        local = [_Clusters(x, clusters, ridge) for x in _variants(s.sample, norms)]
        models[species] = _SpeciesModel(s.lower, s.upper, max(np.sqrt(s.norm.scatter[0, 0] / s.norm.n), _EPS),
                                        [_Gaussian.from_stats(m, ridge) for m in s.moments], local)
        counts[str(species)] = dict(training_atoms=s.norm.n, cluster_atoms=len(norms),
                                    centres={form: len(c.sites) for form, c in zip(_FORMS, local)})
    return _Model(models, _whitener(gram, ridge), _whitener(gram[-16:, -16:], ridge), counts)


def _point(tp, fp, positives, n):
    tp, fp = int(tp), int(fp)
    return dict(tp=tp, fp=fp, fn=positives - tp, tn=n - positives - fp,
                precision=tp/(tp + fp) if tp + fp else None, recall=tp/positives if positives else None,
                f1=2*tp/(tp + fp + positives) if tp + fp + positives else None)


def ranking(scores, labels, *, threshold=None, calibrate=False):
    """Non-interpolated AP with all equal scores entering as one threshold block."""
    scores = np.asarray(scores); labels = np.asarray(labels, dtype=bool)
    n = len(scores)
    if scores.shape != (n,) or labels.shape != (n,) or not n or not np.isfinite(scores).all():
        raise ValueError('ranking needs matching finite, nonempty atom-score and target vectors')
    if calibrate and threshold is not None:
        raise ValueError('cannot both supply and calibrate a threshold')
    positives = int(labels.sum()); ap = None; curve = []
    if positives:
        order = np.argsort(scores)[::-1]
        sorted_scores = scores[order]
        ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), n - 1]
        flagged = ends + 1
        hits = np.cumsum(labels[order], dtype=np.int64)[ends]
        precision = hits / flagged
        ap = float(np.dot(np.diff(hits, prepend=0) / positives, precision))
        for request in (.25, .5, .75, .9, 1.):
            i = np.searchsorted(hits, request * positives)
            curve.append(dict(requested_recall=request, recall=float(hits[i] / positives),
                              precision=float(precision[i]), threshold=float(sorted_scores[ends[i]])))
        if calibrate:
            best = np.argmax(2 * hits / (flagged + positives))  # ties: highest threshold
            threshold = float(sorted_scores[ends[best]])
    point = None
    if threshold is not None:
        flagged = scores >= threshold
        tp = np.count_nonzero(labels & flagged)
        point = _point(tp, flagged.sum() - tp, positives, n)
    return dict(atoms=n, positives=positives, prevalence=positives/n, average_precision=ap,
                precision_at_recall=curve, threshold=threshold, operating_point=point)


class _Provenance:
    def __init__(self):
        self.ids = set(); self.trajectories = {}; self.contract = {}; self.populations = {}

    def frames(self, directory, split):
        paths = sorted(Path(directory).glob('*.npz'))
        if not paths:
            raise ValueError(f'empty {split} directory: {directory}')
        sizes = {}
        for path in paths:
            f = load_frame(path, split)
            if f['frame_id'] in self.ids:
                raise ValueError(f'duplicate frame_id: {f["frame_id"]}')
            self.ids.add(f['frame_id'])
            if self.trajectories.setdefault(f['trajectory_id'], split) != split:
                raise ValueError(f'trajectory_id crosses splits: {f["trajectory_id"]}')
            contract = {k: f[k] for k in (*_IDENT, 'cutoff', 'cutoff_p')}
            contract['h_width'] = f['h'].shape[1]
            for k, value in contract.items():
                if self.contract.setdefault(k, value) != value:
                    raise ValueError(f'{path}: inconsistent {k} across captures')
            row = sizes.setdefault(f['regime'], dict(frames=0, atoms=0))
            row['frames'] += 1; row['atoms'] += len(f['numbers'])
            yield f
        self.populations[split] = sizes


class _Columns:
    """Spool scalar columns, not training features or complete per-frame score dicts."""
    def __init__(self, root, frames, model, error_eV_A):
        root.mkdir()
        self.root = root; self.regimes = {}; self.names = []
        with ExitStack() as stack:
            files = {}
            def append(name, array):
                if name not in files:
                    files[name] = stack.enter_context((root / name).open('wb'))
                array.tofile(files[name])
            for frame in frames:
                y = frame['force_error'] > error_eV_A
                append('labels', y)
                code = self.regimes.setdefault(frame['regime'], len(self.regimes))
                append('regimes', np.full(len(y), code, np.int32))
                scores = model.scores(frame)
                for name, values in scores.items():
                    append(name, values)
            self.names = [k for k in files if k not in ('labels', 'regimes')]

    def get(self, name, dtype=np.float64):
        return np.memmap(self.root / name, mode='r', dtype=dtype)

    def calibrate(self):
        y = self.get('labels', np.bool_)
        return {name: ranking(self.get(name), y, calibrate=True) for name in self.names}

    def fixed(self, valid):
        y = self.get('labels', np.bool_); regimes = self.get('regimes', np.int32)
        out = {}
        for name in self.names:
            threshold = valid[name]['threshold']
            scores = self.get(name)
            fixed = ranking(scores, y, threshold=threshold)
            by_regime = {}
            for regime, code in self.regimes.items():
                if len(self.regimes) > 1:
                    mask = regimes == code
                    by_regime[regime] = ranking(scores[mask], y[mask], threshold=threshold)
                else:
                    by_regime[regime] = fixed
            out[name] = dict(threshold=threshold, valid=valid[name], fixed=fixed, fixed_by_regime=by_regime)
        return out


def evaluate(train, valid, fixed, *, error_eV_A, clusters=4, cluster_samples=4096, ridge=1e-6, seed=0, scratch=None):
    """Fit -> calibrate -> evaluate, refusing cross-split frame/trajectory reuse."""
    if not np.isfinite(error_eV_A) or error_eV_A <= 0:
        raise ValueError('error_eV_A must be a positive finite absolute force-error norm in eV/Angstrom')
    provenance = _Provenance()
    options = dict(clusters=clusters, cluster_samples=cluster_samples, ridge=ridge, seed=seed)
    model = fit(provenance.frames(train, 'train'), **options)
    # Separate contexts also release the validation columns before the fixed set.
    with TemporaryDirectory(dir=scratch) as tmp:
        thresholds = _Columns(Path(tmp) / 'valid', provenance.frames(valid, 'valid'), model, error_eV_A).calibrate()
    with TemporaryDirectory(dir=scratch) as tmp:
        columns = _Columns(Path(tmp) / 'fixed', provenance.frames(fixed, 'fixed-student-test'), model, error_eV_A)
        candidates = columns.fixed(thresholds)
    return dict(target=dict(quantity='||F_student - F_teacher||_2 per atom', comparison='>',
                            cutoff=error_eV_A, unit='eV/Angstrom'),
                calibration='score >= threshold; maximum pooled valid F1, strictest tied threshold; null if no positives',
                ranking='pooled atoms; non-interpolated AP at distinct scores; AP/recall undefined without positives',
                score_definitions=dict(energy_range='eV outside per-species train [min,max]; zero inside',
                    short_distance='cutoff minus shortest incoming distance in Angstrom; zero for isolated atom',
                    mahal='per-species Mahalanobis distance with population covariance + ridge I',
                    cluster='nearest Euclidean site; local Mahalanobis divided by local train P95 (floor 1e-12)',
                    norm_range='two-sided L2 norm range excess divided by species train norm SD (floor 1e-12)',
                    normalized='h / (||h||^b + 1e-12); or_norm takes max with norm_range, dividing single Gaussian by sqrt(width)',
                    llpr='f^T (F^T F + ridge I)^-1 f; uncentred all-species Gram of train atoms',
                    neighbour='MACE polynomial-cutoff-weighted sender mean per receiver; own score if no positive weight'),
                provenance=provenance.contract, populations=provenance.populations, fit=options,
                fit_by_atomic_number=model.fit_counts, candidates=candidates)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for split in ('train', 'valid', 'fixed'):
        parser.add_argument('--' + split, type=Path, required=True, help='directory of per-frame NPZ captures')
    parser.add_argument('--error-eV-A', dest='error_eV_A', type=float, required=True,
                        help='positive if per-atom ||student force - teacher force|| exceeds this many eV/Angstrom')
    parser.add_argument('--clusters', type=int, default=4, help='maximum sites per species and feature form')
    parser.add_argument('--cluster-samples', type=int, default=4096, help='uniform training atoms per species for sites')
    parser.add_argument('--ridge', type=float, default=1e-6, help='positive diagonal added to covariances and LLPR Gram')
    parser.add_argument('--seed', type=int, default=0, help='training cluster reservoir seed')
    parser.add_argument('--scratch', type=Path, help='parent directory for temporary scalar score columns')
    parser.add_argument('--output', type=Path, required=True, help='JSON report')
    args = vars(parser.parse_args(argv)); output = args.pop('output')
    report = evaluate(**args)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
