"""Export the trained physical-energy model; publish only the precise bundle that passed its own 3x gate."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from loupmd import bundles, models
from loupmd.config import GateConfig
from loupmd.errors import GateFailed
from loupmd.validate import gate


def finish(output, cached_bundle, record):
    gate.write_record(record, output / 'gate.json')
    if not record['rows'] or not record['passed']:
        raise GateFailed('student bundle failed the 3x gate', gate_record=record)
    partial = output / 'student.loup.partial'
    try:
        shutil.copyfile(cached_bundle, partial)
        sha = models.sha256_of(partial)
        (output / 'student.loup.sha256').write_text(f'{sha}  student.loup\n')
        os.replace(partial, output / 'student.loup')
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path, help='trusted training output directory with manifest.json and student.model')
    parser.add_argument('structure', type=Path, help='periodic structure representative of the trained elements')
    parser.add_argument('output', type=Path, help='new directory; retains gate evidence even on failure')
    parser.add_argument('--ref-device', choices=('cuda', 'cpu', 'auto'), default='auto')
    args = parser.parse_args()
    manifest = json.loads((args.job / 'manifest.json').read_text())
    paths = set()
    def export(*args, **kwargs):
        path = bundles.ensure_bundle(*args, **kwargs)
        paths.add(path)
        return path
    args.output.mkdir(exist_ok=False, parents=True)
    record = gate.run_gate(args.job / 'student.model', sha256=manifest['student_sha256'], structures=[args.structure],
                           gate_config=GateConfig(reference_device=args.ref_device), ensure_bundle=export)
    print(gate.format_gate(record), flush=True)
    if len(paths) != 1:
        raise ValueError(f'expected a single gated element set, found {len(paths)}')
    finish(args.output, paths.pop(), record)


if __name__ == '__main__':
    main()
