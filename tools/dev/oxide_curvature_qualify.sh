#!/usr/bin/env bash
# Post-collection, distinct new child only. Check ordinary label bars before independently
# relaxing BOTH models from the original unrelaxed protocol input (not the training probes).
set -euo pipefail
run=$(realpath "${1:?collected child directory}"); root=$HOME/loup-s-stage0/sto320
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD LOUP_LIB=$PWD/build/sm_120/libloupmd.so LOUPMD_MODELS=$HOME/models
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
py=$HOME/environments/mace_cueq_latest/bin/python
lock=/mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py
python3 "$lock" status
bash tools/dev/check_oxide_student.sh "$run" "$root/base-labels-v2"
"$py" - "$run" "$root/oxide-curvature-v1/pairs.sha256" <<'PY'
import json, sys
from pathlib import Path
from loupmd.distill.curvature import PARENT
m = json.loads((Path(sys.argv[1])/'checked/manifest.json').read_text())
if (m['experiment'] != 'oxide-curvature-phase-two-child' or m['parent_manifest_sha256'] != PARENT or
        m['curvature']['pairs_sha256'] != Path(sys.argv[2]).read_text().split()[0]):
    raise SystemExit('not the specified sealed child')
PY
out=$run/cold-static; mkdir "$out"
obs(){ "$py" -m loupmd.distill.observables "$@"; }
gpu(){ python3 "$lock" run --purpose "Loup-S curvature $role $1" --wait-minutes 240 -- "$py" -m loupmd.distill.observables "$@"; }
for role in teacher student; do
    opts=(--role "$role")
    if [[ $role == teacher ]]; then opts+=(--model mh1 --head omat_pbe)
    else opts+=(--model "$run/checked/student.model" --sha256 "$(sha256sum "$run/checked/student.model" | cut -d' ' -f1)"); fi
    relax=$out/$role-relax.json
    gpu relax "${opts[@]}" --input "$root/equilibrated.extxyz" --pressure-GPa 0 --fmax .0005 --steps 1800 --output "$relax"
    for d in .005 .01; do
        shards=()
        for ((i=0;i<960;i+=240)); do
            shard=$out/$role-$d-$i.json; shards+=("$shard")
            gpu phonons "${opts[@]}" --relaxation "$relax" --delta-A "$d" --start "$i" --stop "$((i+240))" --output "$shard"
        done
        obs spectrum --relaxation "$relax" --shards "${shards[@]}" --output "$out/$role-$d.json"
    done
done
for d in .005 .01; do
    obs static-compare --teacher "$out/teacher-$d.json" --student "$out/student-$d.json" --output "$out/compare-$d.json"
done
"$py" - "$run" <<'PY'
import json, sys
from pathlib import Path
from loupmd.distill.observable_io import record
r = Path(sys.argv[1]); load = lambda name: json.loads((r/name).read_text())
ok = load('fidelity.json')['student_visited_label_bar'] == 'pass' and load('teacher-score.json')['label_metrics_passed']
for d in ('.005', '.01'):
    a = record(r/'cold-static'/f'compare-{d}.json', 'static-compare')
    modes = a['phonon_pass']; cells = a['lattice_pass']; print(d, sum(modes), '/', len(modes), 'cell', cells)
    ok = ok and len(modes) == 960 and all(modes) and len(cells) == 3 and all(cells)
raise SystemExit(0 if ok else 'child fails ordinary and/or deterministic screens')
PY
