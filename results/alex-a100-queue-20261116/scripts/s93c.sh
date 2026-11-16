#!/bin/bash -l
# S93c on an A100: library ABBA on shared GEMM plans, s86 -> s93a (scratch: first_bwd_edge COLS = 2 capped at 6
# blocks / SM on sm_80, 80 registers, 8-16 B stack) and s86 -> s93b (s93a + gconv_fwd_L1 R1 at C = 128 capped at 5
# blocks / SM on sm_80: 104 -> 96 registers, no stack).
#SBATCH --job-name=loup-s93c
#SBATCH -p a100 --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/loup-s64/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
W=/home/atuin/v130cb/v130cb18/loup-s64
export SLURM_JOB_ID LOUP_LIB=$W/src/libloupmd_s86.so
bash $W/alex64/run.sh bash -c '
echo "=== $(date "+%F %T") $(hostname)"
sha256sum $W/src/libloupmd_s86.so $W/src/libloupmd_s93a.so $W/src/libloupmd_s93b.so | cut -c1-16
A_=$W/src/libloupmd_s86.so@$W/pynew
for v in s93b s93a; do
  B_=$W/src/libloupmd_$v.so@$W/pynew
  echo "--- $v"
  $PY $A/ab_libs.py $M/mace-omat-0-medium.model srtio3 2 $A_ $B_ --same-plans --generic 2>&1 | grep -v -i warn | grep -E "MEAN|rror|child|expected"
  $PY $A/ab_libs.py $M/2023-12-03-mace-128-L1_epoch-199.model srtio3 2 $A_ $B_ --same-plans 2>&1 | grep -v -i warn | grep -E "MEAN|rror|child|expected"
  $PY $A/ab_libs.py $M/MACE-OFF23_medium.model water216 2 $A_ $B_ --same-plans 2>&1 | grep -v -i warn | grep -E "MEAN|rror|child|expected"
done
echo "##### done $(date "+%F %T")"
'
