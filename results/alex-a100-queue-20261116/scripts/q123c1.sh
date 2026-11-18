#!/bin/bash -l
# A100 library ABBA s121 -> s123 on shared plans, 2 rounds, below 1 = s123 faster: S123 / S124 (POLAR-1-S / L), POLAR-1-M as the neutral control. s123 = round6-hardening 990b261 + S122 (x-degree-1 Cartesian forward paths, sm_120 only), S123 (POLAR zero weight blocks skipped), S124 (POLAR-S scalar-only readout), sm_80; s123x80 = the same tree with the sm_120-only Cartesian forms (S119, S121, S122) also compiled for cc 8.0 (test build)
#SBATCH --job-name=loup-q123c1
#SBATCH -p a100 --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/loup-s64/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
W=/home/atuin/v130cb/v130cb18/loup-s64
export SLURM_JOB_ID LOUP_LIB=$W/src/libloupmd_s123.so
bash $W/alex64/run.sh bash -c '
echo "=== $(date "+%F %T") $(hostname) lib $LOUP_LIB $(sha256sum $LOUP_LIB | cut -c1-16)"
A_=$W/src/libloupmd_s121.so@$W/py121
B_=$W/src/libloupmd_s123.so@$W/pys123
for c in MACE-POLAR-1-S:nacl216: MACE-POLAR-1-L:nacl216: MACE-POLAR-1-S:nbcl5: MACE-POLAR-1-M:nacl216:; do
  IFS=: read -r m s x <<< "$c"
  $PY $A/ab_libs.py $M/$m.model $s 2 $A_ $B_ --same-plans $x 2>&1 | grep -v -i warn | grep -E "MEAN|rror|child|expected"
done
echo "##### done $(date "+%F %T")"
'
