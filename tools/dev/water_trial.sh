#!/usr/bin/env bash
# WSL. Use the water release's own pinned recipe and trainer, never a latest checkout upload.
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^(plan|run)$ || ! $2 =~ ^(control|net-torque)$ ]]; then
    echo 'usage: water_trial.sh plan|run control|net-torque' >&2; exit 2
fi
mode=$1; arm=$2; cd "$(dirname "$0")/../.."; export PYTHONPATH=$PWD
root=$HOME/loup-s-stage0/water; data=$root/dispatch; run=$root/$arm-s1-v1
py=$HOME/environments/mace_cueq_latest/bin/python
(cd "$data" && sha256sum -c inputs.sha256 >&2)
recipe=$(mktemp -d); trap 'rm -rf -- "$recipe"' EXIT
# Verify the executable release recipe equals what was signed off before spending CU.
PATH="$(dirname "$py"):$PATH" bash "$data/launch.sh" "$arm" "$run" --plan > "$recipe/plan.json"
"$py" - "$data/release.json" "$recipe/plan.json" "$arm" <<'PY'
import hashlib, json, sys
from pathlib import Path
release, plan = [json.loads(Path(p).read_text()) for p in sys.argv[1:3]]
arm = sys.argv[3]
wrapper = Path(sys.argv[1]).with_name('colab_train.sh')
if (release['schema'] != 'loupmd.water-release.v1' or plan['resume'] is not None or
        hashlib.sha256(wrapper.read_bytes()).hexdigest() != release['inputs_sha256'].get(wrapper.name) or
        any(plan[k] != release[arm][k] for k in ('recipe', 'molecular', 'pair_id')) or
        plan['teacher_sha256'] != release['teacher_sha256'] or plan['teacher_head'] != 'default' or
        plan['recipe']['molecular_loss'] != arm or
        [s['role'] for s in plan['sources']] != ['train', 'valid', 'teacher-test'] or
        any(release['inputs_sha256'][s['role']+'.xyz'] != s['sha256'] for s in plan['sources'])):
    raise SystemExit('water plan differs from the qualified release')
PY
if [[ $mode == plan ]]; then cat "$recipe/plan.json"; exit; fi
ln -s "$data/$arm.args" "$recipe/train.args"
"$py" tools/dev/colab_distill.py --account 1 --gpu L4 --hours 7 --output "$run" "$data/colab_train.sh" \
    "$data"/{train,molecular,molecular_loss}.py "$recipe/train.args" "$data"/{train,valid,teacher-test}.xyz
"$py" -m loupgym.distill.results "$run" "$data"
export LOUP_LIB=$PWD/build/sm_120/libloupmd.so LOUPMD_MODELS="$HOME/models" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py status
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "water $arm export gate" --wait-minutes 240 -- \
    "$py" -m loupgym.distill.publish "$run/checked" "$data/teacher-test.xyz" "$run/release" --ref-device cuda
