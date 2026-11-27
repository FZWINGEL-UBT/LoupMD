"""Publish the frozen Stage 0 base splits with gate-qualified native MH-1 labels (CPU only).

Inputs are the three original byte-pinned files and the three *trusted, complete* JSON reports
from tools/dev/qualify_frozen_cells.py, named {train,valid,teacher-test}.json. Use the same
corrected library file that produced the reports. No model is fitted, selected, or evaluated
here: the issued E/F/S are the exact sequential values that passed on the serialized geometries.
The original engine_build describes the *sampler*; do not replace it with the label engine.
"""
import argparse
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
from ase.io.extxyz import read_xyz

from loupmd import config
from .curve import _Inventory
from .train import file_digest


RELEASE = 'base-labels-v2'
TEACHER_SHA256 = 'a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47'
FROZEN = {
    'train': (300, 'ea8a334655efee6c5512bf171f3aa2fd38e7e39d6d5228d350154ffaf013085f'),
    'valid': (30, '13cef8d03e7e79d4d7d6c919d7f14cac26834ad6aec5f9f613ddc114b98a7924'),
    'teacher-test': (60, 'd0bc3eb8a27673c6611b13a26e861f1746ccaf8b7a655cb584cd4b205b783f0d'),
}
NATIVE_AGREEMENT_LIMITS = {
    'energy_abs_eV_atom': config.GATE_ENERGY_FLOOR_MEV_PER_ATOM / 1000,
    'force_max_abs_eV_A': config.GATE_FORCE_FLOOR_EV_PER_A,
    'stress_max_abs_eV_A3': config.GATE_STRESS_FLOOR_EV_PER_A3,
}


def native_paths_agree(delta):
    """Native repeatability uses the absolute gate floors, not stock FP32's error."""
    return isinstance(delta, dict) and all(
        type(delta.get(k)) in (int, float) and 0 <= delta[k] <= limit for k, limit in NATIVE_AGREEMENT_LIMITS.items())


def _receipt(raw, split, library_sha, count, pin):
    gate = json.loads(raw)
    for key, expected in dict(schema='loupmd.frozen-cell-gate.v1', source_sha256=pin, frames=count,
                              teacher_head='omat_pbe', teacher_sha256=TEACHER_SHA256,
                              library_sha256=library_sha).items():
        if gate.get(key) != expected:
            raise ValueError(f'{split} gate: wrong {key}')
    if gate.get('passed') is not True or not isinstance(gate.get('rows'), list) or len(gate['rows']) != count:
        raise ValueError(f'{split} gate: expected {count} passing rows')
    if not re.fullmatch(r'[a-f0-9]{64}', gate.get('bundle_sha256', '')):
        raise ValueError(f'{split} gate: missing bundle digest')
    if not isinstance(gate.get('library_version'), str) or not gate['library_version'].strip():
        raise ValueError(f'{split} gate: missing library version')
    return gate


def _issue_frame(source, row, library_sha, release):
    # Pinned baseline and curve extxyz use exactly this layout. Preserve their
    # geometry, momenta and sampling metadata without reserializing any of them;
    # ASE's standard writer would round the newly qualified per-atom forces to 8 decimal places.
    count_line = source.readline()
    n = int(count_line)
    header = source.readline().rstrip('\n')
    if ' Properties=species:S:1:pos:R:3:momenta:R:3:REF_forces:R:3 ' not in header:
        raise ValueError('unexpected frozen Properties layout')
    for kind in ('sequential', 'fresh'):
        result = row[kind]
        if result.get('passed') is not True or result.get('failed') != [] or result.get('atoms') != n:
            raise ValueError(f'{row["frame_id"]}: {kind} did not pass')
    if not native_paths_agree(row.get('sequential_vs_fresh')):
        raise ValueError(f'{row["frame_id"]}: sequential vs fresh exceeds native agreement limits or is missing')
    labels = row['labels']
    values = []
    for key, shape in [('REF_energy', ()), ('REF_forces', (n, 3)), ('REF_stress', (3, 3))]:
        a = np.asarray(labels[key])
        if a.shape != shape or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
            raise ValueError(f'{row["frame_id"]}: invalid {key}')
        values.append(a)
    energy, forces, stress = values
    if not np.allclose(stress, stress.T, rtol=1e-7, atol=1e-10):
        raise ValueError(f'{row["frame_id"]}: non-symmetric REF_stress')
    for pattern, replacement in ((r' REF_energy=\S+', ' REF_energy=' + repr(float(energy))),
                                 (r' REF_stress="_JSON [^"]*"', ' REF_stress="_JSON ' + json.dumps(stress.tolist()) + '"')):
        header, substitutions = re.subn(pattern, replacement, header)
        if substitutions != 1:
            raise ValueError('unexpected frozen REF metadata')
    text = [count_line, f'{header} label_release={release} label_library_sha256={library_sha}\n']
    for force in forces:
        original, _, _, _ = source.readline().rsplit(None, 3)
        text.append(original + ' ' + ' '.join(repr(float(x)) for x in force) + '\n')
    payload = ''.join(text)
    frame = next(read_xyz(StringIO(payload), index=0))
    if (frame.info.get('frame_id') != row['frame_id'] or not np.array_equal(frame.info['REF_energy'], energy)
            or not np.array_equal(frame.arrays['REF_forces'], forces) or not np.array_equal(frame.info['REF_stress'], stress)):
        raise ValueError('frame identity or qualified labels did not round-trip')
    return frame, payload.encode('utf-8')


def _publish(base, reports, library, out):
    if out.name != RELEASE:
        raise ValueError(f'new output directory must be named {RELEASE}')
    if os.path.lexists(out):
        raise FileExistsError(f'refusing to replace {out}')
    library_sha = file_digest(library)
    inventory = _Inventory()
    record = dict(schema='loupmd.stage0-label-release.v2', release=RELEASE, teacher_sha256=TEACHER_SHA256,
                  teacher_head='omat_pbe', library_sha256=library_sha, splits={},
                  sampling='Frozen fixed-cell Langevin trajectories; labels reissued, dynamics not regenerated.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=f'.{RELEASE}.') as temp:
        stage = Path(temp) / RELEASE
        (stage / 'gates').mkdir(parents=True)
        for split, (count, pin) in FROZEN.items():
            path = base / f'{split}.xyz'
            raw = path.read_bytes()  # hash exactly the immutable snapshot we parse
            if sha256(raw).hexdigest() != pin:
                raise ValueError(f'{path}: frozen source sha256 mismatch')
            receipt = (reports / f'{split}.json').read_bytes()
            gate = _receipt(receipt, split, library_sha, count, pin)
            for key in ('library_version', 'bundle_sha256'):
                if record.setdefault(key, gate[key]) != gate[key]:
                    raise ValueError(f'{split}: gate reports disagree on {key}')
            source = StringIO(raw.decode('utf-8'))
            cells, sampler_builds = {}, set()
            dest = stage / path.name
            with dest.open('xb') as writer:
                for row in gate['rows']:
                    a, payload = _issue_frame(source, row, library_sha, RELEASE)
                    inventory.accept(a, split, None)  # includes cross-split ID/seed/geometry isolation
                    info = a.info
                    for role in ('teacher', 'sampler'):
                        if info[f'{role}_sha256'] != TEACHER_SHA256 or info[f'{role}_head'] != 'omat_pbe':
                            raise ValueError(f'{split}: unexpected {role} identity')
                    cell = a.cell.array.tobytes()
                    if info['integrator'] != 'langevin' or cells.setdefault(info['trajectory_id'], cell) != cell:
                        raise ValueError(f'{split}: not a fixed-cell Langevin trajectory; investigate sampling')
                    sampler_builds.add(info['engine_build'])
                    writer.write(payload)
                if source.read():
                    raise ValueError(f'{split}: extra source frames')
            output_sha = file_digest(dest)
            (stage / 'gates' / f'{split}.json').write_bytes(receipt)
            (stage / f'{split}.sha256').write_text(f'{output_sha}  {path.name}\n', encoding='utf-8')
            record['splits'][split] = dict(frames=count, trajectories=len(cells), sampler_builds=sorted(sampler_builds),
                                           source_sha256=pin, output_sha256=output_sha, gate_sha256=sha256(receipt).hexdigest())
        (stage / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        if os.path.lexists(out):
            raise FileExistsError(f'refusing to replace {out}')
        stage.rename(out)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='immutable Stage 0 base split directory')
    parser.add_argument('--reports', type=Path, required=True, help='trusted qualification reports, one SPLIT.json each')
    parser.add_argument('--library', type=Path, required=True, help='corrected, gate-tested libloupmd.so (hashed, never loaded)')
    parser.add_argument('--output', type=Path, required=True, help=f'new {RELEASE} directory')
    args = parser.parse_args(argv)
    print(json.dumps(_publish(args.base, args.reports, args.library, args.output), indent=2))


if __name__ == '__main__':
    main()
