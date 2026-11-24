#!/usr/bin/env bash
# WSL: one independent arm of the fixed-composition liquid transport comparison.
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^(plan|run)$ || ! $2 =~ ^(trajectory|mix)$ ]]; then
    echo 'usage: salt_trial.sh plan|run trajectory|mix' >&2; exit 2
fi
mode=$1; arm=$2; cd "$(dirname "$0")/../.."; export PYTHONPATH=$PWD
root=$HOME/loup-s-stage0/molten-salt; run=$root/$arm-s1-v1
py=$HOME/environments/mace_cueq_latest/bin/python; trainer=$root/trainer-v1; train=$trainer/train.py
(cd "$trainer" && printf '%s  SHA256SUMS\n' a04f290501ea6c4e23d0f9c3a6eeb2bf696d3246e5d780293e3ac436845aa226 | sha256sum -c - >&2 && sha256sum -c SHA256SUMS >&2)
(cd "$root" && sha256sum -c inputs.sha256 >&2)
source=train; [[ $arm == trajectory ]] || source=train-mix
data=$(mktemp -d); trap 'rm -rf -- "$data"' EXIT
ln -s "$root/$source.extxyz" "$data/train.xyz"
ln -s "$root/production2.extxyz" "$data/valid.xyz"; ln -s "$root/production3.extxyz" "$data/teacher-test.xyz"
args=(--seed 1 --radial-head 8); files=("$data"/{train,valid,teacher-test}.xyz)
"$py" "$train" --train "${files[0]}" --valid "${files[1]}" --teacher-test "${files[2]}" --out "$run" "${args[@]}" --plan > "$data/plan.json"
python3 - "$root/$source-plan.json" "$data/plan.json" <<'PY'
import json, sys
from pathlib import Path
a, b = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
if (any(a[k] != b[k] for k in ('sources', 'pair_id', 'recipe', 'teacher_sha256', 'teacher_head')) or
        b['resume'] is not None or b['recipe']['dtype'] != 'float64'):
    raise SystemExit('current inputs, teacher or schedule differ from the pre-gated salt arm')
PY
if [[ $mode == plan ]]; then cat "$data/plan.json"; exit; fi
printf '%s\n' "${args[@]}" > "$data/train.args"
"$py" tools/dev/colab_distill.py --account 1 --gpu L4 --hours 6 --output "$run" "$trainer/colab_train.sh" "$train" "$data/train.args" "${files[@]}"
"$py" -m loupmd.distill.results "$run" "$data"
export LOUP_LIB=$PWD/build/sm_120/libloupmd.so LOUPMD_MODELS="$HOME/models" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
reserve=/mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py
python3 "$reserve" status
gpu(){ local why=$1; shift; python3 "$reserve" run --purpose "salt $arm $why" --wait-minutes 240 -- "$py" -m "$@"; }
gpu export loupmd.distill.publish "$run/checked" "${files[2]}" "$run/release" --ref-device cuda
model=$run/checked/student.model; sha=$(sha256sum "$model" | cut -d' ' -f1)
gpu valid loupmd.distill.frames --rescore "${files[1]}" --student "$model" --student-sha256 "$sha" --output "$run/valid-scored.xyz"
gpu teacher-test loupmd.distill.frames --rescore "${files[2]}" --student "$model" --student-sha256 "$sha" --output "$run/teacher-scored.xyz"
