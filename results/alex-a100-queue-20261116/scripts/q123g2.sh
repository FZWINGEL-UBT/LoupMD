#!/bin/bash -l
# A100 3x gates at s123x80: x-degree-2 (S119 / S121) families. s123 = round6-hardening 990b261 + S122 (x-degree-1 Cartesian forward paths, sm_120 only), S123 (POLAR zero weight blocks skipped), S124 (POLAR-S scalar-only readout), sm_80; s123x80 = the same tree with the sm_120-only Cartesian forms (S119, S121, S122) also compiled for cc 8.0 (test build)
#SBATCH --job-name=loup-q123g2
#SBATCH -p a100 --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/atuin/v130cb/v130cb18/loup-s64/runs/slurm-%x-%j.out
unset SLURM_EXPORT_ENV
W=/home/atuin/v130cb/v130cb18/loup-s64
export SLURM_JOB_ID LOUP_LIB=$W/src/libloupmd_s123x80.so
bash $W/alex64/run.sh bash -c '
echo "=== $(date "+%F %T") $(hostname) lib $LOUP_LIB $(sha256sum $LOUP_LIB | cut -c1-16)"
export LOUP_GATE_CACHE=$W/gate_refs 
fails=0
g() { echo "--- $* $(date "+%T")"; PYTHONPATH=$W/pys123x80:$PYTHONPATH $PY $W/pys123x80/test_gate.py "$@" > /tmp/g_$SLURM_JOB_ID.log 2>&1; rc=$?
      grep -E "set:|PASS|FAIL|ref32|Error|Traceback" /tmp/g_$SLURM_JOB_ID.log | grep -v -i warn
      [ $rc = 0 ] || { fails=$((fails + 1)); echo "    rc=$rc"; tail -5 /tmp/g_$SLURM_JOB_ID.log; }; }
g $M/MACE-OFF23_large.model --set organic
g $M/MACE-omol-0-extra-large-1024.model --set organic
g $M/MACE-POLAR-1-L.model --set crystals
echo "##### gates done $(date "+%F %T"): $fails failing gate runs"
echo "##### done $(date "+%F %T")"
'
