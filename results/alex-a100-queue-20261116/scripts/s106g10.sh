#!/bin/bash -l
# S106G10 on an A100: 3x force / stress gates at library s106 (a0d04cd6 + bsrc/src/s106_worktree.patch), cc 8.0
# defaults, with LOUP_CHAIN_FOLD=0: OMAT-0 medium generic and MH-1 without the skip-chain fold. Part of the S106 sweep (s106g1-s106g10) over every family gated locally.
#SBATCH --job-name=loup-s106g10
#SBATCH -p a100 --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/loup-s64/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
W=/home/atuin/v130cb/v130cb18/loup-s64
export SLURM_JOB_ID LOUP_LIB=$W/src/libloupmd_s106.so
bash $W/alex64/run.sh bash -c '
echo "=== $(date "+%F %T") $(hostname) lib $LOUP_LIB $(sha256sum $LOUP_LIB | cut -c1-16)"
export LOUP_GATE_CACHE=$W/gate_refs LOUP_CHAIN_FOLD=0
fails=0
g() { echo "--- $* $(date "+%T")"; PYTHONPATH=$W/pynew:$PYTHONPATH $PY $W/pynew/test_gate.py "$@" > /tmp/g_$SLURM_JOB_ID.log 2>&1; rc=$?
      grep -E "set:|PASS|FAIL|ref32|Error|Traceback" /tmp/g_$SLURM_JOB_ID.log | grep -v -i warn
      [ $rc = 0 ] || { fails=$((fails + 1)); echo "    rc=$rc"; tail -5 /tmp/g_$SLURM_JOB_ID.log; }; }
g $M/mace-omat-0-medium.model --generic --set oxide
g $M/mace-mh-1.model --set crystals
echo "##### gates done $(date "+%F %T"): $fails failing gate runs"
'
