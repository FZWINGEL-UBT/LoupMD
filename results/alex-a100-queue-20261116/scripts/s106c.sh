#!/bin/bash -l
# S106c on an A100: library ABBA s100 -> s106 on shared GEMM plans (S101 RED on sm_120 only, no sm_80 change; S105
# first_fwd_kernel without its density-only warp; S106 Langevin first half with one thread per normal draw, noise
# bitwise unchanged), 2 rounds, below 1 = s106 faster. Library s106 = a0d04cd6 + bsrc/src/s106_worktree.patch.
#SBATCH --job-name=loup-s106c
#SBATCH -p a100 --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/loup-s64/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
W=/home/atuin/v130cb/v130cb18/loup-s64
export SLURM_JOB_ID LOUP_LIB=$W/src/libloupmd_s106.so
bash $W/alex64/run.sh bash -c '
echo "=== $(date "+%F %T") $(hostname)"
sha256sum $W/src/libloupmd_s100.so $W/src/libloupmd_s106.so | cut -c1-16
A_=$W/src/libloupmd_s100.so@$W/pynew
B_=$W/src/libloupmd_s106.so@$W/pynew
$PY $A/ab_libs.py $M/mace-omat-0-medium.model srtio3 2 $A_ $B_ --same-plans --generic 2>&1 | grep -v -i warn | grep -E "MEAN|rror|child|expected"
for c in "2023-12-03-mace-128-L1_epoch-199.model srtio3" "MACE-OFF23_medium.model water216" "MACE-OFF23_small.model water216" "mace-mh-1.model nacl216" "MACE-POLAR-1-S.model nacl216"; do
  set -- $c
  $PY $A/ab_libs.py $M/$1 $2 2 $A_ $B_ --same-plans 2>&1 | grep -v -i warn | grep -E "MEAN|rror|child|expected"
done
echo "##### done $(date "+%F %T")"
'
