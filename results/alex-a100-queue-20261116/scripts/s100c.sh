#!/bin/bash -l
# S100c on an A100: library ABBA s86 -> s100 (S96 sm_120-only; S97-S100 generic pass glue: h0 gather, readout seed, top conv
# backward assigns edge_dvec, dx zeroed in the forward conv, readout in the product kernels) on shared GEMM plans; then the OMAT-0 medium path comparison (specialised / generic, separate processes) and 3x gates at s100 (OMAT-0 generic oxide, OFF23-S organic).
#SBATCH --job-name=loup-s100c
#SBATCH -p a100 --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/loup-s64/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
W=/home/atuin/v130cb/v130cb18/loup-s64
export SLURM_JOB_ID LOUP_LIB=$W/src/libloupmd_s100.so
bash $W/alex64/run.sh bash -c '
echo "=== $(date "+%F %T") $(hostname)"
sha256sum $W/src/libloupmd_s86.so $W/src/libloupmd_s100.so | cut -c1-16
A_=$W/src/libloupmd_s86.so@$W/pynew
B_=$W/src/libloupmd_s100.so@$W/pynew
$PY $A/ab_libs.py $M/mace-omat-0-medium.model srtio3 2 $A_ $B_ --same-plans --generic 2>&1 | grep -v -i warn | grep -E "MEAN|rror|child|expected"
$PY $A/ab_libs.py $M/2023-12-03-mace-128-L1_epoch-199.model srtio3 2 $A_ $B_ --same-plans 2>&1 | grep -v -i warn | grep -E "MEAN|rror|child|expected"
$PY $A/ab_libs.py $M/MACE-OFF23_medium.model water216 2 $A_ $B_ --same-plans 2>&1 | grep -v -i warn | grep -E "MEAN|rror|child|expected"
for s in srtio3 nbcl5; do
  $PY $A/ab_libs.py $M/mace-omat-0-medium.model $s 2 $B_ $B_+generic 2>&1 | grep -v -i warn | grep -E "MEAN|rror|child"
done
export LOUP_GATE_CACHE=$W/gate_refs
g() { echo "--- $* $(date "+%T")"; PYTHONPATH=$W/pynew:$PYTHONPATH $PY $W/pynew/test_gate.py "$@" > /tmp/g_$SLURM_JOB_ID.log 2>&1; rc=$?
      grep -E "PASS|FAIL|Error|Traceback" /tmp/g_$SLURM_JOB_ID.log | grep -v -i warn | tail -20; echo "    rc=$rc"; }
g $M/mace-omat-0-medium.model --generic
g $M/MACE-OFF23_small.model --set organic
echo "##### done $(date "+%F %T")"
'
