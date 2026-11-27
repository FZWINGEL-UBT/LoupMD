#!/usr/bin/env bash
# One cold-started Stage 0 experiment. Launch only after deciding dtype against
# the two FP64 controls, absolute fidelity and timing. No implicit GPU allocation.
set -euo pipefail
usage(){ echo 'usage: oxide_sweep.sh plan|run cutoff45|cutoff60|width32|width96|curve1000|curve3000|curve10000 float32|float64 CAMPAIGN_ROOT' >&2; exit 2; }
[[ $# == 4 ]] || usage
mode=$1; arm=$2; dtype=$3; root=$(realpath "$4"); [[ $mode == plan || $mode == run ]] || usage
[[ $dtype == float64 || $dtype == float32 ]] || usage
cd "$(dirname "$0")/../.."; export PYTHONPATH=$PWD
py="$HOME/environments/mace_cueq_latest/bin/python"; train=loupgym/distill/train.py
base=$root/base-labels-v2; data=$base; width=64; cutoff=5; epochs=150; boundary=111
case $arm in
    cutoff45) cutoff=4.5;; cutoff60) cutoff=6;; width32) width=32;; width96) width=96;;
    curve1000) epochs=45; boundary=33; data=$root/curve-v2-1000;;
    curve3000) epochs=15; boundary=11; data=$root/curve-v2-3000;;
    curve10000) epochs=5; boundary=4; data=$root/curve-v2-10000;;
    *) usage;;
esac
if [[ $data == "$base" ]]; then
    (cd "$base" && sha256sum -c train.sha256 valid.sha256 teacher-test.sha256 >&2)
else
    # Reject a data point made from old labels or an unmatched validation/test set.
    "$py" - "$base" "$data" "${arm#curve}" <<'PY'
import sys
from pathlib import Path
from loupgym.distill.curve import verify_training_release
verify_training_release(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
PY
fi
args=(--seed 1 --width "$width" --cutoff "$cutoff" --radial-head 64 --dtype "$dtype" --epochs "$epochs" --stage-two-start "$boundary")
run=$root/v2-l4-$arm-$dtype-seed1; input=("$data/train.xyz" "$data/valid.xyz" "$data/teacher-test.xyz")
if [[ $mode == plan ]]; then
    exec "$py" "$train" --train "${input[0]}" --valid "${input[1]}" --teacher-test "${input[2]}" --out "$run" "${args[@]}" --plan
fi
recipe=$(mktemp -d); trap 'rm -rf -- "$recipe"' EXIT; printf '%s\n' "${args[@]}" >"$recipe/train.args"
"$py" tools/dev/colab_distill.py --account 1 --gpu L4 --output "$run" --hours 6 tools/dev/colab_train.sh "$train" "$recipe/train.args" "${input[@]}"
bash tools/dev/check_oxide_student.sh "$run" "$data"
