#!/bin/bash
# 3x float32 gate (oxide set) on symmetrix main (b907f04) MACEFoundation, LoupMD rows alongside.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
P=/home/fzwin/environments/mace_cueq_latest/bin/python
D=/mnt/d/Margraf/LoupMD/results/symmetrix-main-20261116
J=/home/fzwin/src/symmetrix-eval/json
bash "$D/scripts/job_extract.sh" NaCl NbCl5
$P "$D/scripts/sx_gate.py" --json-dir "$J" | tee "$D/gate_summary.txt"
