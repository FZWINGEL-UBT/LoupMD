#!/bin/bash
# Queue round 2 (job_apples.sh) under one local-GPU reservation; log in ../logs/apples.log.
D=/mnt/d/Margraf/LoupMD/results/symmetrix-main-20261116
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --quiet \
  --purpose 'loupmd dev: symmetrix vs LoupMD equal-skin force calls + nsys profiles (SrTiO3, OMAT-0)' \
  --wait-minutes 240 -- bash "$D/scripts/job_apples.sh" > "$D/logs/apples.log" 2>&1
echo "apples job exit $?"
