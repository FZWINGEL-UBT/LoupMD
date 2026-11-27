#!/usr/bin/env bash
# Only the owner invokes colab_distill.py. Flat uploads: this payload plus
# curvature_train.py curvature.py train.py pairs.npz pairs.sha256 manifest.json
# pre_stage_two.pt train.xyz valid.xyz teacher-test.xyz.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1
printf '%s  pairs.npz\n' 12b121e62860989f2ff299ed8c62d2fe0fe3ebea951b499970477239dcb0d407 | sha256sum -c -
sha256sum -c pairs.sha256
python -m pip install -q mace-torch==0.3.16 e3nn==0.4.4
trap 'for f in plan.json manifest.json student.model pre_stage_two.pt; do if test -f "job/$f"; then cp "job/$f" out/; fi; done' EXIT
python curvature_train.py --train train.xyz --valid valid.xyz --teacher-test teacher-test.xyz --parent . \
    --pairs pairs.npz --seal pairs.sha256 --out job 2>&1 | tee out/train.log
