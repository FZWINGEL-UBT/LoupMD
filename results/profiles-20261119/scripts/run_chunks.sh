#!/usr/bin/env bash
# Runs results/profiles-20261119/scripts/job_profiles.sh in short chunks, each through its own local-GPU reservation
# (one after another, so other users' queued jobs interleave).
L=/mnt/d/Margraf/LoupMD
J=$L/results/profiles-20261119/scripts/job_profiles.sh
LIB=${LIB:?LIB}
while read -r part only; do
  [ -z "$part" ] && continue
  echo "##### chunk $part [$only] $(date '+%F %T')"
  python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "loupmd dev: deep profile $part ${only:0:40}" \
    --wait-minutes 900 --quiet -- env LIB=$LIB ONLY="$only" bash $J $part
done << 'EOF'
nsys omat0m_spec|omat0m_gen|mp0m_srtio3|mp0m_nbcl5|mptrj|omat0s
nsys mh1_|off23s|off23m
nsys off24m|off23l|omol0xl
nsys polar
gemm mh1|mp0m|omat0m|mptrj
gemm off23l|polar1m|omol0xl
ncu mp0m_srtio3|omat0m_spec
ncu omat0m_gen|off23l|mptrj_srtio3
ncu mh1_srtio3
ncu polar1m
EOF
echo "##### all chunks done $(date '+%F %T')"
