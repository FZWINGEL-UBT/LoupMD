#!/bin/bash
# Round 2: force call vs force call at equal skin (0.5 and 1.0 A, both libraries), then an nsys kernel profile of
# 20 force calls per implementation (SrTiO3 5,000 atoms, skin 0.5 A). symmetrix at 1024 spline points.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
P=/home/fzwin/environments/mace_cueq_latest/bin/python
D=/mnt/d/Margraf/LoupMD/results/symmetrix-main-20261116
S=/home/fzwin/src/symmetrix-eval/scratch
J=/home/fzwin/src/symmetrix-eval/json/omat0m-8-22-38-1024.json
X=$D/scripts/sx_apples.py
OUT=$D/apples_time.jsonl
sha256sum /mnt/d/Margraf/LoupMD/libloupmd.so /home/fzwin/src/symmetrix-eval/b907f04/symmetrix/_mace_foundation*.so
rm -f "$OUT"

cfg() {
  case $1 in
    loup) echo "--impl loup" ;;
    loup-generic) echo "--impl loup-generic" ;;
    sx-f32) echo "--impl sx --dtype float32 --json $J" ;;
    sx-mix) echo "--impl sx --dtype float32-mixed --json $J" ;;
  esac
}

RUNS=()
for skin in 0.5 1.0; do
  for l in loup loup-generic sx-f32 sx-mix; do RUNS+=("$l:$skin"); done
done
for r in "${RUNS[@]}"; do
  $P "$X" time $(cfg "${r%%:*}") --skin "${r##*:}" --label "${r%%:*}" --out "$OUT" --pass-tag 1
done
for ((i=${#RUNS[@]}-1; i>=0; i--)); do
  r=${RUNS[$i]}
  $P "$X" time $(cfg "${r%%:*}") --skin "${r##*:}" --label "${r%%:*}" --out "$OUT" --pass-tag 2
done

for l in loup loup-generic sx-f32 sx-mix; do
  nsys profile -t cuda --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
    -f true -o "$S/prof_$l" $P "$X" prof $(cfg "$l") --skin 0.5 --size 10 --calls 20
  nsys stats --force-export true -r cuda_gpu_trace,cuda_gpu_kern_sum -f csv -o "$S/prof_$l" "$S/prof_$l.nsys-rep"
done
mkdir -p "$D/prof"
cp "$S"/prof_*_cuda_gpu_kern_sum.csv "$D/prof/"
$P "$X" report --scratch "$S" --out "$OUT" --labels loup,loup-generic,sx-f32,sx-mix | tee "$D/apples_report.md"
