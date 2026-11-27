#!/usr/bin/env bash
# Invoke from checkout after BOTH export gates and teacher rescores are published.
set -euo pipefail
root=$(pwd -P); d=$HOME/loup-s-stage0/molten-salt; py=$HOME/environments/mace_cueq_latest/bin/python
export LOUP_LIB=$root/build/sm_120/libloupmd.so LOUPMD_MODELS=$HOME/models PYTHONPATH=$root TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
run() { python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "salt-$1" --wait-minutes 240 -- "$py" -m loupgym.distill.frames "${@:2}"; }
cd "$d"; sha256sum -c inputs.sha256
declare -A admitted
for arm in trajectory mix; do
  job=$arm-s1-v1
  admitted[$arm]=$("$py" -m loupgym.distill.salt_admission "$arm")
  test -s "$job/teacher-scored.xyz"
done
traj=trajectory-s1-v1/checked/student.model; ts=${admitted[trajectory]}
if [[ ! -f fixed.sha256 ]]; then (cd "$root" && bash tools/dev/molten_salt_fixed.sh "$d/$traj" "$ts"); fi
sha256sum -c fixed.sha256
for arm in trajectory mix; do
 job=$arm-s1-v1; model=$job/checked/student.model; sha=${admitted[$arm]}
 if [[ $arm == mix ]]; then
   if [[ ! -f $job/anchor-scored.xyz ]]; then run mix-anchor --rescore fixed-student-test.extxyz --student "$model" --student-sha256 "$sha" --output "$job/anchor-scored.xyz"; fi
   for i in 4 5; do
     extra=(); if [[ $i == 5 ]]; then extra=(--hot-K 1800 --hot-frames 24 --strain .02 --shear .01); fi
     if [[ ! -f $job/own$i.xyz ]]; then run mix-own$i --teacher mh1-omat --teacher-head omat_pbe --split fixed-student-test --student "$model" --student-sha256 "$sha" --input fixed-cool$i.extxyz --input-index -1 --seed $((98000+i)) --target-K 1200 --target-frames 24 --stride 125 --burn-in 1500 "${extra[@]}" --output "$job/own$i.xyz"; fi
   done
   cat "$job"/own{4,5}.xyz > "$job/own.xyz"
   "$py" -m loupgym.distill.fidelity "$job/anchor-scored.xyz" --output "$job/anchor.json"
 else
   cp fixed-student-test.extxyz "$job/own.xyz"
 fi
 "$py" -m loupgym.distill.fidelity "$job/own.xyz" --regime target --teacher-test "$job/teacher-scored.xyz" --round 0 --output "$job/own-target.json"
 "$py" -m loupgym.distill.fidelity "$job/own.xyz" --regime hot --output "$job/own-hard.json"
 for i in 0 1 2 3; do
   if [[ ! -f $job/transport$i.extxyz ]]; then run "$arm-transport$i" --teacher mh1-omat --teacher-head omat_pbe --split fixed-student-test --student "$model" --student-sha256 "$sha" --input liquid$i.extxyz --output "$job/transport$i.extxyz" --seed $((93000+i)) --target-K 1200 --burn-in 2000 --stride 50 --target-frames 200 --rattle-A 0 --strain 0 --shear 0; fi
 done
 "$py" "$root/tools/dev/molten_salt_transport.py" "$job" > "$job/transport-reference.json"
done
"$py" "$root/tools/dev/molten_salt_transport.py" . > teacher-transport-reference.json
"$py" - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('teacher-transport-reference.json').read_text())['diffusion_A2_ps']; out={}
for arm in ('trajectory','mix'):
 j=Path(f'{arm}-s1-v1'); s=json.loads((j/'transport-reference.json').read_text())['diffusion_A2_ps']; out[arm]={k:dict(student_mean=v['mean'], teacher_mean=r[k]['mean'], difference=v['mean']-r[k]['mean'], within_teacher_seed_range=r[k]['minimum']<=v['mean']<=r[k]['maximum'],all_seeds_diffusive=min(v['per_seed'])>0) for k,v in s.items()}
Path('transport-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
PY
sha256sum -c thermo/teacher/measured.sha256
for arm in trajectory mix; do
  while [[ ! -f thermo/$arm/measured.sha256 ]]; do sleep 30; done
  sha256sum -c "thermo/$arm/measured.sha256"
done
sha256sum fixed.sha256 teacher-transport-reference.json transport-comparison.json {trajectory,mix}-s1-v1/{own.xyz,own-target.json,own-hard.json,transport-reference.json,checked/student.model,release/gate.json} mix-s1-v1/{anchor-scored.xyz,anchor.json} > student-evaluation.sha256
sha256sum thermo/{teacher,trajectory,mix}/measured.sha256 >> student-evaluation.sha256
