#!/bin/bash -l
# A100 library ABBA s123 -> s123x80 on shared plans, 2 rounds, below 1 = the Cartesian forms (S119 / S121 / S122) faster on cc 8.0. s123 = round6-hardening 990b261 + S122 (x-degree-1 Cartesian forward paths, sm_120 only), S123 (POLAR zero weight blocks skipped), S124 (POLAR-S scalar-only readout), sm_80; s123x80 = the same tree with the sm_120-only Cartesian forms (S119, S121, S122) also compiled for cc 8.0 (test build)
#SBATCH --job-name=loup-q123c2
#SBATCH -p a100 --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/loup-s64/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
W=/home/atuin/v130cb/v130cb18/loup-s64
export SLURM_JOB_ID LOUP_LIB=$W/src/libloupmd_s123x80.so
bash $W/alex64/run.sh bash -c '
echo "=== $(date "+%F %T") $(hostname) lib $LOUP_LIB $(sha256sum $LOUP_LIB | cut -c1-16)"
A_=$W/src/libloupmd_s123.so@$W/pys123
B_=$W/src/libloupmd_s123x80.so@$W/pys123x80
for c in MACE-OFF23_large:water216: MACE-OFF23_medium:water216: MACE-OFF24_medium:water: 2023-12-03-mace-128-L1_epoch-199:srtio3: mace-omat-0-medium:srtio3:--generic MACE-POLAR-1-L:nacl216: MACE-omol-0-extra-large-1024:water216:; do
  IFS=: read -r m s x <<< "$c"
  $PY $A/ab_libs.py $M/$m.model $s 2 $A_ $B_ --same-plans $x 2>&1 | grep -v -i warn | grep -E "MEAN|rror|child|expected"
done
echo "##### done $(date "+%F %T")"
'
