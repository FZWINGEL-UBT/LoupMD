#!/bin/bash
# Force-call timing, symmetrix main (b907f04) MACEFoundation vs LoupMD, OMAT-0 medium, SrTiO3 6^3 / 10^3 / 14^3:
# a 300 K snapshot, then every configuration in one order and again in the reverse order (one process each).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
P=/home/fzwin/environments/mace_cueq_latest/bin/python
D=/mnt/d/Margraf/LoupMD/results/symmetrix-main-20261116
S=/home/fzwin/src/symmetrix-eval/scratch
J=/home/fzwin/src/symmetrix-eval/json
OUT=$D/speed.jsonl
sha256sum /mnt/d/Margraf/LoupMD/libloupmd.so /home/fzwin/src/symmetrix-eval/b907f04/symmetrix/_mace_foundation*.so
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader || true
mkdir -p "$S"
rm -f "$OUT"
$P "$D/scripts/sx_speed.py" prep --scratch "$S" --sizes 6,10,14

one() {  # label pass
  case $1 in
    loup) args=(--impl loup) ;;
    loup-generic) args=(--impl loup-generic) ;;
    sx-f32-1024) args=(--impl sx --dtype float32 --json "$J/omat0m-8-22-38-1024.json") ;;
    sx-mix-1024) args=(--impl sx --dtype float32-mixed --json "$J/omat0m-8-22-38-1024.json") ;;
    sx-f32-256) args=(--impl sx --dtype float32 --json "$J/omat0m-8-22-38-256.json") ;;
    sx-mix-256) args=(--impl sx --dtype float32-mixed --json "$J/omat0m-8-22-38-256.json") ;;
  esac
  echo "== $1 (pass $2)"
  $P "$D/scripts/sx_speed.py" run --scratch "$S" --label "$1" "${args[@]}" --out "$OUT" --pass-tag "$2"
}

LABELS=(loup loup-generic sx-f32-1024 sx-mix-1024 sx-f32-256 sx-mix-256)
for l in "${LABELS[@]}"; do one "$l" 1; done
for ((i=${#LABELS[@]}-1; i>=0; i--)); do one "${LABELS[$i]}" 2; done
$P "$D/scripts/sx_speed.py" summary --scratch "$S" --out "$OUT" | tee "$D/speed_summary.md"
