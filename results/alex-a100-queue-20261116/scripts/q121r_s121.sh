#!/bin/bash -l
# A100 spread of the S106 OMAT-0 stress failure: specialised oxide gate three times at library s121 (and the generic unfolded one below, three times).
#SBATCH --job-name=loup-q121r_s121
#SBATCH -p a100 --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/loup-s64/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
W=/home/atuin/v130cb/v130cb18/loup-s64
export SLURM_JOB_ID LOUP_LIB=$W/src/libloupmd_s121.so
bash $W/alex64/run.sh bash -c '
echo "=== $(date "+%F %T") $(hostname) lib $LOUP_LIB $(sha256sum $LOUP_LIB | cut -c1-16)"
export LOUP_GATE_CACHE=$W/gate_refs 
fails=0
g() { echo "--- $* $(date "+%T")"; PYTHONPATH=$W/py121:$PYTHONPATH $PY $W/py121/test_gate.py "$@" > /tmp/g_$SLURM_JOB_ID.log 2>&1; rc=$?
      grep -E "set:|PASS|FAIL|ref32|Error|Traceback" /tmp/g_$SLURM_JOB_ID.log | grep -v -i warn
      [ $rc = 0 ] || { fails=$((fails + 1)); echo "    rc=$rc"; tail -5 /tmp/g_$SLURM_JOB_ID.log; }; }
g $M/mace-omat-0-medium.model --set oxide
g $M/mace-omat-0-medium.model --set oxide
g $M/mace-omat-0-medium.model --set oxide
echo "##### gates done $(date "+%F %T"): $fails failing gate runs"
export LOUP_GATE_CACHE=$W/gate_refs LOUP_CHAIN_FOLD=0
fails=0
g() { echo "--- $* $(date "+%T")"; PYTHONPATH=$W/py121:$PYTHONPATH $PY $W/py121/test_gate.py "$@" > /tmp/g_$SLURM_JOB_ID.log 2>&1; rc=$?
      grep -E "set:|PASS|FAIL|ref32|Error|Traceback" /tmp/g_$SLURM_JOB_ID.log | grep -v -i warn
      [ $rc = 0 ] || { fails=$((fails + 1)); echo "    rc=$rc"; tail -5 /tmp/g_$SLURM_JOB_ID.log; }; }
g $M/mace-omat-0-medium.model --generic --set oxide
g $M/mace-omat-0-medium.model --generic --set oxide
g $M/mace-omat-0-medium.model --generic --set oxide
echo "##### gates done $(date "+%F %T"): $fails failing gate runs"
echo "##### done $(date "+%F %T")"
'
