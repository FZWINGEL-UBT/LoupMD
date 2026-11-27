#!/usr/bin/env bash
# WSL: check a collected Stage 0 oxide run on its training version of the splits.
# Usage: bash tools/dev/check_oxide_student.sh COLAB_RUN FROZEN_SPLITS
set -euo pipefail
if [[ $# != 2 ]]; then echo 'usage: check_oxide_student.sh COLAB_RUN FROZEN_SPLITS' >&2; exit 2; fi
run=$(realpath "$1"); splits=$(realpath "$2"); cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD LOUP_LIB=$PWD/build/sm_120/libloupmd.so LOUPMD_MODELS="$HOME/models"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py="$HOME/environments/mace_cueq_latest/bin/python"
reserve=/mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py
"$py" -m loupgym.distill.results "$run" "$splits"
model=$run/checked/student.model
sha=$(sha256sum "$model" | cut -d' ' -f1)
gpu(){ local why=$1; shift; python3 "$reserve" run --purpose "Loup-S $(basename "$run"): $why" --wait-minutes 240 -- "$py" -m "$@"; }
gpu export loupgym.distill.publish "$run/checked" "$splits/teacher-test.xyz" "$run/release" --ref-device cuda
gpu teacher-test loupgym.distill.frames --rescore "$splits/teacher-test.xyz" --student "$model" --student-sha256 "$sha" --output "$run/teacher-scored.xyz"
"$py" -m loupgym.distill.fidelity "$run/teacher-scored.xyz" --regime target --output "$run/teacher-score.json"
gpu fixed-student-test loupgym.distill.frames --split fixed-student-test --teacher mh1-omat --teacher-head omat_pbe --student "$model" --student-sha256 "$sha" --seed 3101 --target-K 300 --replicas 2 --target-frames 30 --burn-in 1000 --stride 50 --rattle-A 0.03 --output "$run/fixed-student-test.xyz"
"$py" -m loupgym.distill.fidelity "$run/fixed-student-test.xyz" --regime target --teacher-test "$run/teacher-scored.xyz" --round 0 --output "$run/fidelity.json"
