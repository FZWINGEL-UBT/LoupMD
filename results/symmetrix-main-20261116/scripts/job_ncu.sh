#!/bin/bash
# Round 2b: ncu memory metrics of the convolution kernels of one force call per implementation (SrTiO3 5,000 atoms,
# skin 0.5 A), after the 20 warm-up calls of sx_apples.py prof (profiling opened by cudaProfilerStart).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
P=/home/fzwin/environments/mace_cueq_latest/bin/python
D=/mnt/d/Margraf/LoupMD/results/symmetrix-main-20261116
J=/home/fzwin/src/symmetrix-eval/json/omat0m-8-22-38-1024.json
X=$D/scripts/sx_apples.py
NCU=/usr/local/cuda/bin/ncu
M=gpu__time_duration.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_atom.sum
M=$M,lts__t_sectors_op_red.sum,dram__bytes_read.sum,dram__bytes_write.sum
M=$M,lts__t_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed
M=$M,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active
M=$M,launch__registers_per_thread,smsp__inst_executed.sum,l1tex__t_sector_hit_rate.pct
mkdir -p "$D/ncu"
run() {  # label regex impl-args...
  local label=$1 rx=$2
  shift 2
  $NCU --profile-from-start off --target-processes all -k "regex:$rx" --metrics "$M" --csv --page raw \
    --log-file "$D/ncu/$label.csv" $P "$X" prof "$@" --skin 0.5 --size 10 --calls 1
}
run loup "spline_forward_inter1|wprime_backward|tabled_outer_sum0" --impl loup
run loup-generic "gconv_fwd_L1|gconv_bwd_L1|first_fwd|first_bwd_edge" --impl loup-generic
run sx-f32 "k_Phi1|k_Phi0" --impl sx --dtype float32 --json "$J"
run sx-mix "k_Phi1|k_Phi0" --impl sx --dtype float32-mixed --json "$J"
ls -la "$D/ncu"
$P - <<'EOF' | tee "$D/ncu/edges.txt"
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

s = np.load("/home/fzwin/src/symmetrix-eval/scratch/srtio3_300K.npz")
at = Atoms(numbers=s["Z10"], positions=s["pos10"], cell=s["cell10"], pbc=True)
print("SrTiO3 10x10x10 snapshot: directed edges with r < 6 A:", len(neighbor_list("i", at, 6.0)))
EOF
