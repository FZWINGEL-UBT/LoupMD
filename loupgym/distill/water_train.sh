#!/bin/bash
# Colab account 1 only; use after the dispatcher's existing runtime is prepared.
# Usage: bash launch.sh control|net-torque NEW_OUTPUT_DIRECTORY [--plan]
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
case ${1:-} in control|net-torque) mode=$1;; *) exit 2;; esac
out=${2:?new output directory required}
shift 2
(cd "$root" && sha256sum -c inputs.sha256 >&2)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mapfile -t recipe < "$root/$mode.args"
exec python "$root/train.py" --train "$root/train.xyz" --valid "$root/valid.xyz" --teacher-test "$root/teacher-test.xyz" --out "$out" "${recipe[@]}" "$@"
