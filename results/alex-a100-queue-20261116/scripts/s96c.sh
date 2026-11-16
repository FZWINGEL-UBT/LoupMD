#!/bin/bash -l
# S96c on an A100: the receiver-owned w (LOUP_WPRIME_RCV=1) against the default (edge-tiled below 512 atoms) on small
# SrTiO3 cells, OMAT-0 medium specialised path, library s86, separate processes (ab_libs.py +ENV side flag).
#SBATCH --job-name=loup-s96c
#SBATCH -p a100 --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/loup-s64/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
W=/home/atuin/v130cb/v130cb18/loup-s64
export SLURM_JOB_ID LOUP_LIB=$W/src/libloupmd_s86.so
bash $W/alex64/run.sh bash -c '
echo "=== $(date "+%F %T") $(hostname)"
A_=$W/src/libloupmd_s86.so@$W/pynew
for s in srtio3x3 srtio3x4 srtio3x2; do
  $PY $A/ab_libs.py $M/mace-omat-0-medium.model $s 3 $A_ $A_+LOUP_WPRIME_RCV=1 2>&1 | grep -v -i warn | grep -E "MEAN|rror|child"
done
echo "##### done $(date "+%F %T")"
'
