"""Check every serialized water label against stock OFF24-M FP64 and FP32 on CPU.

CPU-only (e.g. leto): CUDA_VISIBLE_DEVICES='' python -m loupmd.distill.water_gate
SPLIT.xyz MACE-OFF24_medium.model OUT.json. Run separately per immutable shard.
No native library, sampling, student, or model fitting is involved.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from ase.io import iread
import numpy as np

from loupmd.validate import gate, references
from .train import file_digest
from .water import inspect_water


TEACHER = 'e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69'


def qualify(source, checkpoint):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('set CUDA_VISIBLE_DEVICES to the empty string on the CPU reference host')
    digest = file_digest(source)
    if file_digest(checkpoint) != TEACHER:
        raise ValueError('unexpected OFF24-medium checkpoint digest')
    configurations = []
    frame_ids = set()
    for a in iread(source, ':'):
        fid = a.info['frame_id']
        if fid in frame_ids or a.info['teacher_sha256'] != TEACHER or a.info['teacher_head'] != 'default':
            raise ValueError('invalid frame/teacher identity')
        inspect_water(a)
        frame_ids.add(fid)
        configurations.append((fid, a))
    if not configurations:
        raise ValueError('empty source')
    r64, r32, stress = references.compute(checkpoint, configurations, None, 'cpu', None, None)
    rows = []
    for (fid, a), r, n, s in zip(configurations, r64, r32, stress, strict=True):
        result = a.info['REF_energy'], a.arrays['REF_forces'], a.info['REF_stress']
        if not all(np.isfinite(x).all() for x in result):
            raise ValueError('invalid stored labels')
        rows.append(asdict(gate.compare(fid, {}, len(a), result, r, n, s)))
    if file_digest(source) != digest:
        raise ValueError('source changed during qualification')
    return dict(schema='loupmd.water-stored-label-gate.v1', source=source.name, source_sha256=digest,
                teacher_sha256=TEACHER, reference_device='cpu', reference_versions=references.versions(),
                rule='stock LoupMD gate.compare on every stored REF_energy/forces/stress at its serialized geometry',
                frames=len(rows), passed=all(row['passed'] for row in rows), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'checkpoint', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    record = qualify(args.source, args.checkpoint)
    with args.output.open('x') as stream:
        json.dump(record, stream, allow_nan=False)
    print(f"{args.source.name}: {sum(r['passed'] for r in record['rows'])}/{record['frames']} passed")
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
