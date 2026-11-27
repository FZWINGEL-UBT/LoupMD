#!/usr/bin/env bash
# Payload for colab_distill.py. Inputs: standalone train.py, the three .xyz splits,
# and train.args (one argument per line; optional resume/harvest inputs alongside).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1
python -m pip install -q mace-torch==0.3.16 e3nn==0.4.4
python -c 'import torch, mace; print("torch", torch.__version__, torch.version.cuda, "mace", mace.__version__, "device", torch.cuda.get_device_name())'
mapfile -t recipe < train.args
trap 'for f in train.log plan.json manifest.json student.model pre_stage_two.pt; do if test -f "job/$f"; then cp "job/$f" out/; fi; done' EXIT
python train.py --train train.xyz --valid valid.xyz --teacher-test teacher-test.xyz --out job "${recipe[@]}"
