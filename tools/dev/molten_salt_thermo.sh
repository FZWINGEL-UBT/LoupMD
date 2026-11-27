#!/usr/bin/env bash
# Run separately for teacher, trajectory, or mix. These are new independent MD
# seeds, not the four transport replicas. Both roles start at the same melt.
set -euo pipefail
root=$(pwd -P); d=$HOME/loup-s-stage0/molten-salt; py=$HOME/environments/mace_cueq_latest/bin/python
export LOUP_LIB=$root/build/sm_120/libloupmd.so LOUPMD_MODELS=$HOME/models PYTHONPATH=$root TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
arm=${1:?teacher, trajectory, or mix}; cd "$d"; out=thermo/$arm; mkdir -p "$out"
case $arm in teacher) role=teacher; model=(--model mh1-omat --head omat_pbe);; trajectory|mix) role=student; j=$arm-s1-v1; sha=$("$py" -m loupgym.distill.salt_admission "$arm"); model=(--model "$j/checked/student.model" --sha256 "$sha");; *) exit 2;; esac
obs() { "$py" -m loupgym.distill.observables "$@"; }
md() { local name=$1 steps=$2; shift 2; if [[ ! -f $out/$name/segment-$steps.json ]]; then python3 /mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py run --purpose "salt-$arm-$name" --wait-minutes 240 -- "$py" -m loupgym.distill.observables md --input liquid0.extxyz --temperature-K 1200 --role "$role" "${model[@]}" --directory "$out/$name" --steps "$steps" "$@"; fi; }
for seed in 99001 99002; do md npt$seed 10000 --seed "$seed" --ensemble npt --timestep-fs 1 --stride 20 --pressure-GPa .0001; done
md nve1 4000 --seed 99003 --ensemble nve --timestep-fs 1 --stride 10
md nve05 8000 --seed 99003 --ensemble nve --timestep-fs .5 --stride 20
t=thermo/teacher; student=(); prod=(); cons=(); if [[ $role == student ]]; then student=(--student "$out"/npt{99001,99002}/segment-10000.json); prod=("$out/nve1/segment-4000.json"); cons=("$out/nve05/segment-8000.json"); fi
for b in 50 100; do if [[ ! -f $out/statistics-$b.json ]]; then obs statistics --teacher "$t"/npt{99001,99002}/segment-10000.json "${student[@]}" --discard-frames 101 --block-frames "$b" --lags 5 10 20 30 --rmax-A 7.5 --bins 60 --output "$out/statistics-$b.json"; fi; done
if [[ ! -f $out/nve.json ]]; then obs nve --production "$t/nve1/segment-4000.json" "${prod[@]}" --conserving "$t/nve05/segment-8000.json" "${cons[@]}" --blocks 4 --output "$out/nve.json"; fi
sha256sum liquid0.extxyz "$out"/{statistics-50,statistics-100,nve}.json > "$out/measured.sha256"
