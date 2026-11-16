#!/bin/bash
# Queue round 3 (job_ncu_full.sh) under one local-GPU reservation; log in ../logs/ncu_full.log.
D=/mnt/d/Margraf/LoupMD/results/symmetrix-main-20261116
python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run \
  --purpose 'loupmd dev: ncu sections, layer-1 conv kernels, LoupMD vs symmetrix (SrTiO3 5,000)' \
  --wait-minutes 240 -- bash "$D/scripts/job_ncu_full.sh" > "$D/logs/ncu_full.log" 2>&1
echo "ncu_full job exit $?"
