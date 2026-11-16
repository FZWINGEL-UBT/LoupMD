#!/bin/bash
# Queue round 2b (job_ncu.sh) under one local-GPU reservation; log in ../logs/ncu.log.
D=/mnt/d/Margraf/LoupMD/results/symmetrix-main-20261116
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run \
  --purpose 'loupmd dev: ncu conv kernels, symmetrix vs LoupMD (SrTiO3 5,000, OMAT-0)' \
  --wait-minutes 240 -- bash "$D/scripts/job_ncu.sh" > "$D/logs/ncu.log" 2>&1
echo "ncu job exit $?"
