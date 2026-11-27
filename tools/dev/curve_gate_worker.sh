#!/usr/bin/env bash
# Short, preemptible stock FP64/FP32 + corrected fresh/sequential native checks.
# Run one instance per admitted host. Kernel flock releases even on interruption;
# immutable inputs are shared, an interrupted or failed gate is never published.
set -euo pipefail
root=${1:?corrected-curve-v2 release root}; r=$(dirname "$root"); host=$(hostname -s)
case "$host" in
  leeloo) sm=89; libsha=c147f32501f36ba57bb10a016b6611ecb3701f0fff5dd89c68ab76e61095d494;;
  coraline) sm=86; libsha=a536f453dc147235d40eeacbe0aca49f9115b637fd0dbbcd255d4a8c1e9511ce;;
  *) echo "no qualified library for $host" >&2; exit 2;;
esac
PY=/mnt/homes/zwingel/environments/mace_cueq_latest/bin/python
guard=/mnt/homes/zwingel/loupmd-s140-20261119/gpu_guard.py
export PATH="$(dirname "$PY"):$PATH" PYTHONPATH="$root/source" PYTHONDONTWRITEBYTECODE=1
default_lib="$r/cell-fix-048489b9/tree/build/sm_$sm/libloupmd.so"
export LOUPMD_MODELS="$r/curve1/inputs" LOUPMD_CACHE="$r/curve1/cache-$host"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
printf '%s  %s\n' "$libsha" "$default_lib" | sha256sum -c -
registry=$root/exact-$host.json
run_gate() {
  local mode=$1 i=$2 final=$3 rejected=$4 attempt rc=0
  export LOUP_LIB="$default_lib"
  if [[ $mode == exact ]]; then
    # Only the explicitly certified strict build can replace a rejected GROUP.
    local strict digest
    strict=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["library"])' "$registry")
    digest=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["library_sha256"])' "$registry")
    printf '%s  %s\n' "$digest" "$strict" | sha256sum -c -
    export LOUP_LIB="$strict"
  fi
  attempt=$(mktemp -d "$root/work/gate-$i-$host-$mode.XXXXXX")
  timeout 12m /usr/bin/python3 "$guard" "$PY" -u "$root/source/tools/dev/qualify_frozen_cells.py" "$root/work/group-$i.xyz" "$attempt/gate.json" --fresh-workers 4 >"$attempt/log" 2>&1 || rc=$?
  if ((rc==1)) && [[ $mode == default && -s $attempt/gate.json ]] && "$PY" -c 'import json,sys; a=json.load(open(sys.argv[1])); sys.exit(not (a["passed"] is False and len(a["rows"]) == a["frames"] == 100))' "$attempt/gate.json"; then
    ln "$attempt/gate.json" "$rejected"
    echo "rejected group $i $host $mode $(date -u +%FT%TZ); not issued"
    return 0
  fi
  if ((rc)); then echo "$host: group $i $mode gate exit $rc; $attempt" >&2; exit "$rc"; fi
  mv -T "$attempt/gate.json" "$final"
  echo "qualified group $i $host $mode $(date -u +%FT%TZ)"
}
exec 9>"$r/curve1/worker-$host.lock"; flock -n 9 || exit 75
for ((i=${2:-0}; i<388; i+=4)); do
  final=$root/gates/group-$i.json
  rejected=$root/work/rejected-$i.json
  [[ ! -e $final && ( ! -e $rejected || -e $registry ) ]] || continue
  exec 8>"$root/work/group-$i.lock"; flock -n 8 || continue
  [[ ! -e $final ]] || continue
  # Pin exactly the bytes we will send to the gate. The gate rechecks after execution.
  "$PY" -c 'import json,sys,hashlib; from pathlib import Path; root=Path(sys.argv[1]); i=sys.argv[2]; assert hashlib.sha256((root/f"work/group-{i}.xyz").read_bytes()).hexdigest()==json.loads((root/"catalog.json").read_bytes())["groups"][i]' "$root" "$i"
  if [[ ! -e $rejected ]]; then run_gate default "$i" "$final" "$rejected"; fi
  if [[ -e $rejected && -e $registry ]]; then run_gate exact "$i" "$final" "$rejected"; fi
  flock -u 8
done
