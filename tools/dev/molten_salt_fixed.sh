#!/usr/bin/env bash
# Run once for the chosen reference student; comparison students RESCORE these
# frozen bytes, not new trajectories. Arguments: trusted checkpoint and digest.
set -euo pipefail
root=$(pwd -P); d=$HOME/loup-s-stage0/molten-salt; py=$HOME/environments/mace_cueq_latest/bin/python
model=$(realpath "${1:?student.model}"); digest=${2:?trusted sha256}; shift 2
export LOUP_LIB=$root/build/sm_120/libloupmd.so LOUPMD_MODELS=$HOME/models PYTHONPATH=$root TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$d"; sha256sum -c inputs.sha256; printf '%s  %s\n' "$digest" "$model" | sha256sum -c -
run() { python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "salt-fixed-$1" --wait-minutes 240 -- "$py" -m loupgym.distill.frames --teacher mh1-omat --teacher-head omat_pbe "${@:2}"; }
for i in 4 5; do
  # Reserved disjoint randomized cation roots. Melt at the teacher NPT mean
  # volume measured for the first two replicas; no train geometry is reused.
  "$py" - "$i" <<'PY'
from ase.io import read,write
from pathlib import Path
import numpy as np, sys
i=sys.argv[1]; x=read(f'seed{i}.extxyz'); v=np.mean([read(f'liquid{j}.extxyz').get_volume() for j in (0,1)]); x.set_cell(x.cell*(v/x.get_volume())**(1/3),scale_atoms=True)
p=Path(f'fixed-seed{i}.extxyz'); assert not p.exists(); write(p,x)
PY
  run melt$i --split teacher-test --input fixed-seed$i.extxyz --output fixed-melt$i.extxyz --seed $((96000+i)) --target-K 1800 --target-frames 1 --stride 1 --burn-in 6000
  run cool$i --split teacher-test --input fixed-melt$i.extxyz --output fixed-cool$i.extxyz --seed $((97000+i)) --target-K 1200 --target-frames 100 --stride 50 --burn-in 2000
  "$py" -m loupgym.distill.salt transport fixed-cool$i.extxyz > fixed-cool$i.json
  "$py" -c 'import json,sys; assert all(v["D_A2_ps"]>0 and v["msd_A2"][-1]>3 for v in json.load(open(sys.argv[1]))["species"].values())' fixed-cool$i.json
  extra=(); if [[ $i == 5 ]]; then extra=(--hot-K 1800 --hot-frames 24 --strain .02 --shear .01); fi
  run student$i --split fixed-student-test --student "$model" --student-sha256 "$digest" --input fixed-cool$i.extxyz --input-index -1 --output fixed-student$i.extxyz --seed $((98000+i)) --target-K 1200 --target-frames 24 --stride 125 --burn-in 1500 "${extra[@]}"
  python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "salt-fixed-gate$i" --wait-minutes 240 -- "$py" "$root/tools/dev/qualify_frozen_cells.py" fixed-student$i.extxyz fixed-gate$i.json
  "$py" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["passed"] and all(x["historical"]["passed"] for x in r["rows"])' fixed-gate$i.json
done
cat fixed-student{4,5}.extxyz > fixed-student-test.extxyz
sha256sum fixed-*.extxyz fixed-*.json "$model" "$root/tools/dev/molten_salt_fixed.sh" > fixed.sha256
