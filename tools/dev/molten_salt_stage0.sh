#!/usr/bin/env bash
# Run from this checkout in WSL, one FIFO reservation at a time. Intermediate
# trajectories are NOT training data. Student-visited inputs require a trained model.
set -euo pipefail
root=$(pwd -P); d=$HOME/loup-s-stage0/molten-salt; py=$HOME/environments/mace_cueq_latest/bin/python
export LOUP_LIB=$root/build/sm_120/libloupmd.so LOUPMD_MODELS=$HOME/models PYTHONPATH=$root TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$d"; sha256sum -c seeds.sha256
run() { python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "salt-$1" --wait-minutes 240 -- "${@:2}"; }
frames() { run "$1" "$py" -m loupmd.distill.frames --teacher mh1-omat --teacher-head omat_pbe --rattle-A 0 --strain 0 --shear 0 "${@:2}"; }
for i in 0 1 2 3; do
  case $i in 0|1) split=train; n=50; h=26;; 2) split=valid; n=30; h=0;; 3) split=teacher-test; n=60; h=0;; esac
  if [[ ! -f premelt$i.extxyz ]]; then
    frames melt$i --split "$split" --input seed$i.extxyz --output premelt$i.extxyz --seed $((91000+i)) --target-K 1800 --target-frames 40 --burn-in 2000 --stride 100
  fi
  # Sample NPT from the hot end, not the ordered seed. Use second-half mean
  # volume to fix the production cell and discard the pressure relaxation.
  if [[ ! -f hot-end$i.extxyz ]]; then "$py" -c 'from ase.io import read,write; import sys; write(sys.argv[2], read(sys.argv[1],index=-1))' premelt$i.extxyz hot-end$i.extxyz; fi
  if [[ ! -f npt$i/segment-8000.json ]]; then
    run npt$i "$py" -m loupmd.distill.observables md --input hot-end$i.extxyz --directory npt$i --model mh1-omat --head omat_pbe --role teacher --ensemble npt --seed $((92000+i)) --temperature-K 1200 --timestep-fs 1 --steps 8000 --stride 50 --pressure-GPa 0.0001
  fi
  while [[ ! -f liquid$i.extxyz ]]; do
    endpoint=$(ls -v npt"$i"/segment-*.json | tail -1)
    if "$py" - "$i" "$endpoint" <<'PY'
from loupmd.distill.observable_md import Snapshot
from ase.io import write
from pathlib import Path
import numpy as np, sys,json
i=sys.argv[1]; s=Snapshot(Path(sys.argv[2])); a=list(s.data(thermo=False)); v=np.array([x.get_volume() for x in a[len(a)//2:]]); x=a[-1]; ratio=v[:len(v)//2].mean()/v[len(v)//2:].mean(); rho=x.get_masses().sum()*1.66053906660/v.mean()
print(json.dumps(dict(replica=i, mean_volume_A3=v.mean(),density_g_cm3=rho,half_ratio=ratio)),flush=True)
if abs(ratio-1)>.05 or not 1<rho<2.5: sys.exit(42)
x.set_cell(x.cell*(v.mean()/x.get_volume())**(1/3),scale_atoms=True); write(f'liquid{i}.extxyz',x,write_results=False)
PY
    then break; else rc=$?; fi
    if [[ $rc != 42 || $endpoint == */segment-32000.json ]]; then exit "$rc"; fi
    run npt-extend$i "$py" -m loupmd.distill.observables extend --previous "$endpoint" --steps 8000
  done
  if [[ ! -f transport$i.extxyz ]]; then frames transport$i --split "$split" --input liquid$i.extxyz --output transport$i.extxyz --seed $((93000+i)) --target-K 1200 --burn-in 2000 --stride 50 --target-frames 200; fi
  "$py" -m loupmd.distill.salt transport transport$i.extxyz > transport$i.json
  "$py" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert all(v["D_A2_ps"]>0 and v["msd_A2"][-1]>6 for v in r["species"].values()),"no four-species liquid diffusion evidence"' transport$i.json
  if [[ ! -f production$i.extxyz ]]; then frames production$i --split "$split" --input transport$i.extxyz --input-index -1 --output production$i.extxyz --first-replica "$i" --seed $((94000+i)) --target-K 1200 --hot-K 1800 --target-frames "$n" --hot-frames "$h" --stride 250 --burn-in 2000; fi
  if [[ ! -f gate$i.json ]]; then run gate$i "$py" "$root/tools/dev/qualify_frozen_cells.py" production$i.extxyz gate$i.json; fi
done
cat production{0,1}.extxyz > train.extxyz
if [[ ! -f train-mix.extxyz ]]; then run mix "$py" -m loupmd.distill.salt mix train-mix.extxyz --seed 95000 --sigma 0.04 train.extxyz; fi
# Every atom order is qualified independently; the reference tool rejects mixed orders.
for i in 0 1; do
  "$py" -c 'from ase.io import iread,write; import sys; i=int(sys.argv[1]); write(f"mix{i}.extxyz",[f for f in iread("train-mix.extxyz",index=":") if f.info["replica"]==i])' "$i"
  if [[ ! -f gate-mix$i.json ]]; then run gate-mix$i "$py" "$root/tools/dev/qualify_frozen_cells.py" mix$i.extxyz gate-mix$i.json; fi
done
"$py" - <<'PY'
from pathlib import Path
import json
for p in Path().glob('gate*.json'):
 r=json.loads(p.read_text()); assert r['passed'] and all(x['historical']['passed'] for x in r['rows']),p
PY
for arm in train train-mix; do "$py" -m loupmd.distill.train --train "$arm.extxyz" --valid production2.extxyz --teacher-test production3.extxyz --out "$arm-job" --seed 1 --radial-head 8 --plan > "$arm-plan.json"; done
sha256sum *.extxyz *.json "$LOUP_LIB" "$root/loupmd/distill/salt.py" "$root/tools/dev/molten_salt_stage0.sh" > inputs.sha256
