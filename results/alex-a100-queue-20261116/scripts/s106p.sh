#!/bin/bash -l
# S106p on an A100: OMAT-0 medium path comparison at library s106, specialised (side a) against generic (side b) in
# separate processes, 2 rounds, below 1 = generic faster; SrTiO3 5,000 / 1,080 / 320 atoms and NbCl5.
#SBATCH --job-name=loup-s106p
#SBATCH -p a100 --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/loup-s64/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
W=/home/atuin/v130cb/v130cb18/loup-s64
export SLURM_JOB_ID LOUP_LIB=$W/src/libloupmd_s106.so
bash $W/alex64/run.sh bash -c '
echo "=== $(date "+%F %T") $(hostname)"
sha256sum $W/src/libloupmd_s106.so | cut -c1-16
B_=$W/src/libloupmd_s106.so@$W/pynew
for s in srtio3 nbcl5 srtio3x6 srtio3x4; do
  $PY $A/ab_libs.py $M/mace-omat-0-medium.model $s 2 $B_ $B_+generic 2>&1 | grep -v -i warn | grep -E "MEAN|rror|child"
done
echo "##### done $(date "+%F %T")"
'
