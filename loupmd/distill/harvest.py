"""Select the worst independent harvest frames for a pre-phase-two restart. Never select test frames."""
from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import iread

from .fidelity import score_frames
from .frames import _write_set


def select(source, output, count):
    if count < 1:
        raise ValueError('count must be positive')
    def frames():
        return iread(source, index=':', format='extxyz')

    report = score_frames(frames())
    if report['provenance']['split'] != 'harvest' or count > len(report['frames']):
        raise ValueError('select at most the available frames of a separate harvest split')
    ranked = sorted(report['frames'], key=lambda f: (-f['force_rmse_eV_per_A'], f['frame_id']))[:count]
    wanted = {row['frame_id'] for row in ranked}
    selected = {}
    for a in frames():
        key = a.info['frame_id']
        if key in wanted:
            selected[key] = a
    def output_frames():
        for row in ranked:
            a = selected[row['frame_id']]
            a.info['harvest_force_rmse_eV_per_A'] = row['force_rmse_eV_per_A']
            for name in ('STUDENT_energy', 'STUDENT_stress'):
                del a.info[name]
            del a.arrays['STUDENT_forces']
            yield a
    return _write_set(Path(output), output_frames())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--count', type=int, required=True)
    args = parser.parse_args()
    print(f'{select(args.source, args.output, args.count)} harvest frames: {args.output}')


if __name__ == '__main__':
    main()
