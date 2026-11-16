#!/bin/bash -l
# S86c on an A100: library ABBA s84 -> s86 (S85 / S86 POLAR launch trims: bias + LayerNorm-SiLU forward fused, SCF
# copy + add fused) on shared GEMM plans, POLAR-1-S/M/L NaCl216.
#SBATCH --job-name=loup-s86c
#SBATCH -p a100 --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/loup-s64/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
W=/home/atuin/v130cb/v130cb18/loup-s64
export SLURM_JOB_ID LOUP_LIB=$W/src/libloupmd_s86.so
bash $W/alex64/run.sh bash -c '
echo "=== $(date "+%F %T") $(hostname)"
sha256sum $W/src/libloupmd_s84.so $W/src/libloupmd_s86.so | cut -c1-16
A_=$W/src/libloupmd_s84.so@$W/pynew
B_=$W/src/libloupmd_s86.so@$W/pynew
for c in "MACE-POLAR-1-M.model nacl216" "MACE-POLAR-1-S.model nacl216" "MACE-POLAR-1-L.model nacl216"; do
  set -- $c
  $PY $A/ab_libs.py $M/$1 $2 3 $A_ $B_ --same-plans 2>&1 | grep -v -i warn | grep -E "MEAN|rror|child|expected"
done
echo "##### done $(date "+%F %T")"
'
