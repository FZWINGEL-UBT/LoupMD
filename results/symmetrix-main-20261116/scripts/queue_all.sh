#!/bin/bash
# Queue the timing job, then the gate job, each under its own local-GPU reservation (logs in ../logs).
D=/mnt/d/Margraf/LoupMD/results/symmetrix-main-20261116
Q=/mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py
python3 "$Q" run --quiet --purpose 'loupmd dev: symmetrix main vs LoupMD force-call timing (SrTiO3, OMAT-0)' \
  --wait-minutes 240 -- bash "$D/scripts/job_speed.sh" > "$D/logs/speed.log" 2>&1
echo "speed job exit $?"
python3 "$Q" run --purpose 'loupmd dev: symmetrix main 3x gate (oxide set)' \
  --wait-minutes 240 -- bash "$D/scripts/job_gate.sh" > "$D/logs/gate.log" 2>&1
echo "gate job exit $?"
