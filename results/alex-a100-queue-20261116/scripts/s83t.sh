#!/bin/bash -l
# S83t on an A100: in-process A/B of the generic conv-forward variant tuning (LOUP_CONV_TUNE=1, default on sm_120
# only) on library s81 (the S78 tree, cc 8.0 defaults, lane-group neighbour kernels).
#SBATCH --job-name=loup-s83t
#SBATCH -p a100 --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/loup-s64/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
W=/home/atuin/v130cb/v130cb18/loup-s64
export SLURM_JOB_ID LOUP_LIB=$W/src/libloupmd_s81.so
bash $W/alex64/run.sh bash -c '
echo "=== $(date "+%F %T") $(hostname) lib $LOUP_LIB"
export PYTHONPATH=$W/pynew:$PYTHONPATH LOUP_DEBUG_SPLINE=1
for c in "mace-omat-0-medium.model srtio3 --generic" "2023-12-03-mace-128-L1_epoch-199.model srtio3" "MACE-OFF24_medium.model water" "MACE-OFF23_medium.model water216"; do
  set -- $c
  echo "--- $c $(date "+%T")"
  $PY $A/ab_sides.py $M/$1 $2 3 - - LOUP_CONV_TUNE=1 ${@:3} 2>&1 | grep -v -i warn | grep -E "ms/step|rror| vs 0|conv|us:"
done
echo "##### done $(date "+%F %T")"
'
