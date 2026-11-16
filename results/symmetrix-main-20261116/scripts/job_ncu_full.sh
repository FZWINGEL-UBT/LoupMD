#!/bin/bash
# Round 3: ncu sections (speed of light, memory workload, warp states, scheduler, occupancy) for the layer-1
# convolution kernels of one force call: LoupMD default and generic, symmetrix float32 (SrTiO3 5,000 atoms, skin 0.5).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
P=/home/fzwin/environments/mace_cueq_latest/bin/python
D=/mnt/d/Margraf/LoupMD/results/symmetrix-main-20261116
S=/home/fzwin/src/symmetrix-eval/scratch
J=/home/fzwin/src/symmetrix-eval/json/omat0m-8-22-38-1024.json
X=$D/scripts/sx_apples.py
NCU=/usr/local/cuda/bin/ncu
SECTIONS="--section SpeedOfLight --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables"
SECTIONS="$SECTIONS --section WarpStateStats --section SchedulerStats --section Occupancy --section LaunchStats"
SECTIONS="$SECTIONS --section InstructionStats"
mkdir -p "$D/ncu_full"
run() {  # label regex impl-args...
  local label=$1 rx=$2
  shift 2
  $NCU --profile-from-start off --target-processes all -k "regex:$rx" $SECTIONS -f -o "$S/ncufull_$label" \
    $P "$X" prof "$@" --skin 0.5 --size 10 --calls 1
  $NCU --import "$S/ncufull_$label.ncu-rep" --page details --csv > "$D/ncu_full/$label.csv"
}
run loup "spline_forward_inter1|wprime_backward" --impl loup
run loup-generic "gconv_fwd_L1|gconv_bwd_L1" --impl loup-generic
run sx-f32 "k_Phi1" --impl sx --dtype float32 --json "$J"
ls -la "$D/ncu_full"
