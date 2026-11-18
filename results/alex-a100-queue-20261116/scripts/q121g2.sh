#!/bin/bash -l
# A100 3x force / stress gates at library s121, cc 8.0 defaults. s121 = round6-hardening 990b261 (S115 POLAR tpo fold, S116 single radial table, S118 POLAR-1-S live k-space channels, S119 / S121 degree-two Cartesian paths (sm_120 only), S120 neighbour bin skip), sm_80
#SBATCH --job-name=loup-q121g2
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
g $M/MACE_MPtrj_2022.9.model --set oxide
g $M/mace-omat-0-small.model --set oxide
g $M/mace-mpa-0-medium.model --set oxide
echo "##### gates done $(date "+%F %T"): $fails failing gate runs"
echo "##### done $(date "+%F %T")"
'
