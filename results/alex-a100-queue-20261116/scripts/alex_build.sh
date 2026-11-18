#!/bin/bash -l
# On the Alex login node: bash alex_build.sh TAG TGZ -> $W/src/libloupmd_TAG.so (sm_80) and $W/pyTAG (python/).
# Builds in $W/bsrcTAG; log in $W/bsrcTAG/build.log; writes $W/bsrcTAG/DONE (or FAILED) at the end.
set -uo pipefail
TAG=$1
TGZ=$2
W=/home/atuin/v130cb/v130cb18/loup-s64
module purge >/dev/null 2>&1
module load cuda/13.2.0
B=$W/bsrc$TAG
rm -rf "$B" && mkdir -p "$B" && tar xzf "$TGZ" -C "$B"
rm -rf "$W/py$TAG" && cp -r "$B/python" "$W/py$TAG"
cd "$B"
{
  echo "$(date '+%F %T') build $TAG from $TGZ ($(sha256sum "$TGZ" | cut -c1-16))"
  NV=$(command -v nvcc || true)
  [ -n "$NV" ] || { echo "no nvcc on PATH (run with bash -l)"; touch FAILED; exit 1; }
  echo "nvcc: $NV"
  if make NVCC="$NV" CUDA_ARCH=sm_80 -j8 build/sm_80/libloupmd.so && [ -f build/sm_80/libloupmd.so ]; then
    cp build/sm_80/libloupmd.so "$W/src/libloupmd_$TAG.so"
    echo "$(date '+%F %T') ok $(sha256sum "$W/src/libloupmd_$TAG.so" | cut -c1-16)"
    touch DONE
  else
    echo "$(date '+%F %T') make failed"
    touch FAILED
  fi
} > build.log 2>&1
