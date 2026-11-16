#!/bin/bash -l
# S86g on an A100: POLAR-1-S/M/L crystal gates at library s86 (= round6-hardening 9f8cac77 sources) with the defaults.
#SBATCH --job-name=loup-s86g
#SBATCH -p a100 --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/loup-s64/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
W=/home/atuin/v130cb/v130cb18/loup-s64
export SLURM_JOB_ID LOUP_LIB=$W/src/libloupmd_s86.so
bash $W/alex64/run.sh bash -c '
echo "=== $(date "+%F %T") $(hostname) lib $LOUP_LIB"
export LOUP_GATE_CACHE=$W/gate_refs
fails=0
g() { echo "--- $* $(date "+%T")"; PYTHONPATH=$W/pynew:$PYTHONPATH $PY $W/pynew/test_gate.py "$@" > /tmp/g_$SLURM_JOB_ID.log 2>&1; rc=$?
      grep -E "set:|PASS|FAIL|ref32|Error|Traceback" /tmp/g_$SLURM_JOB_ID.log | grep -v -i warn
      [ $rc = 0 ] || { fails=$((fails + 1)); echo "    rc=$rc"; tail -5 /tmp/g_$SLURM_JOB_ID.log; }; }
g $M/MACE-POLAR-1-M.model --set crystals
g $M/MACE-POLAR-1-L.model --set crystals
g $M/MACE-POLAR-1-S.model --set crystals
echo "##### gates done $(date "+%F %T"): $fails failing gate runs"
'
