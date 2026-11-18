#!/bin/bash -l
# A100 library ABBA s106 -> s121 on shared plans, 2 rounds, below 1 = s121 faster: POLAR (S115, S118). s121 = round6-hardening 990b261 (S115 POLAR tpo fold, S116 single radial table, S118 POLAR-1-S live k-space channels, S119 / S121 degree-two Cartesian paths (sm_120 only), S120 neighbour bin skip), sm_80
#SBATCH --job-name=loup-q121c1
#SBATCH -p a100 --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/loup-s64/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
W=/home/atuin/v130cb/v130cb18/loup-s64
export SLURM_JOB_ID LOUP_LIB=$W/src/libloupmd_s121.so
bash $W/alex64/run.sh bash -c '
echo "=== $(date "+%F %T") $(hostname) lib $LOUP_LIB $(sha256sum $LOUP_LIB | cut -c1-16)"
A_=$W/src/libloupmd_s106.so@$W/pynew
B_=$W/src/libloupmd_s121.so@$W/py121
for c in MACE-POLAR-1-S:nacl216: MACE-POLAR-1-M:nacl216: MACE-POLAR-1-L:nacl216: MACE-POLAR-1-S:nbcl5:; do
  IFS=: read -r m s x <<< "$c"
  $PY $A/ab_libs.py $M/$m.model $s 2 $A_ $B_ --same-plans $x 2>&1 | grep -v -i warn | grep -E "MEAN|rror|child|expected"
done
echo "##### done $(date "+%F %T")"
'
