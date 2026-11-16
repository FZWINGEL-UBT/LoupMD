#!/bin/bash
# Extract MACE-OMAT-0 medium for symmetrix MACEFoundation: the given families (SrTiO3 NaCl NbCl5) at 1024 spline
# points (the CLI's and extract_mace_data's default) and 256 (the pair_symmetrix README's stated default). CPU only,
# run under the GPU reservation because the MACE / torch imports probe the driver.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
P=/home/fzwin/environments/mace_cueq_latest/bin/python
D=/mnt/d/Margraf/LoupMD/results/symmetrix-main-20261116
J=/home/fzwin/src/symmetrix-eval/json
mkdir -p "$J"
for fam in "$@"; do
  case $fam in
    SrTiO3) zs=8,22,38 ;;
    NaCl) zs=11,17 ;;
    NbCl5) zs=17,41 ;;
    *) echo "unknown family $fam"; exit 2 ;;
  esac
  for pts in 1024 256; do
    $P "$D/scripts/sx_extract.py" --pkg /home/fzwin/src/symmetrix-eval/b907f04 \
      --model /home/fzwin/models/mace-omat-0-medium.model --species "$zs" --points "$pts" \
      --output "$J/omat0m-${zs//,/-}-$pts.json"
  done
done
ls -la "$J"
