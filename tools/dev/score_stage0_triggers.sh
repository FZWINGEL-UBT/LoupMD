#!/usr/bin/env bash
# Git Bash driver. All DATA paths are absolute WSL paths. Separate one-time
# preparation, foreground FIFO reservations, and CPU-only analysis:
#   bash tools/dev/score_stage0_triggers.sh prepare OUT SYSTEM JOB GATE TRAIN VALID0 TEACHER0 VALID TEACHER TARGET HARD
#   for part in train valid fixed; do bash tools/dev/score_stage0_triggers.sh capture OUT "$part"; done
#   bash tools/dev/score_stage0_triggers.sh score OUT [--error-eV-A 0.03 --error-eV-A 0.05]
# JOB contains the independently gated physical-energy student.model and manifest.json.
# VALID0/TEACHER0 are the frozen raw validation and teacher-test inputs from JOB.
# TARGET/HARD must already be independent, teacher-qualified, and student-visited.
# Pass the same fixed file twice to partition its target/hot regimes byte-for-byte.
set -euo pipefail
if [[ $1 == _prepare ]]; then
    shift; out=$1; system=$2; job=$3; gate=$4; shift 4
    [[ $# == 7 && $out == /* && $system =~ ^[[:alnum:]_-]+$ && ! -e $out ]]
    for p in "$job/student.model" "$job/manifest.json" "$gate" "$@"; do [[ $p == /* && -s $p ]] || { echo "Missing input: $p" >&2; exit 1; }; done
    tmp=$(mktemp -d "${out}.XXXXXX"); trap 'rm -rf -- "$tmp"' EXIT
    names=(train original-valid original-teacher valid teacher target hard)
    for name in "${names[@]}"; do ln -s "$1" "$tmp/$name.xyz"; shift; done
    ln -s "$job/manifest.json" "$tmp/training-manifest.json"; ln -s "$gate" "$tmp/student-gate.json"; ln -s "$job/student.model" "$tmp/student.model"
    cd "$tmp"; python3 - <<'PY'
import hashlib,json
from pathlib import Path
m=json.loads(Path('training-manifest.json').read_text());gate=json.loads(Path('student-gate.json').read_text());h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if not gate['passed'] or not gate['rows'] or gate['model']['sha256']!=m['student_sha256'] or h('student.model')!=m['student_sha256']:raise ValueError('checkpoint, trusted manifest and passing gate must agree')
pins={s['role']:s['sha256'] for s in m['sources']}
for role,path in [('train','train.xyz'),('valid','original-valid.xyz'),('teacher-test','original-teacher.xyz')]:
    if h(path)!=pins[role]:raise ValueError(f'{role}: wrong frozen input')
if Path('target.xyz').samefile('hard.xyz'):
    import shlex
    source=Path('target.xyz').resolve()
    Path('target.xyz').unlink(); Path('hard.xyz').unlink()
    Path('fixed-source.xyz').symlink_to(source)
    with source.open('rb') as src, open('target.xyz','xb') as target, open('hard.xyz','xb') as hard:
        while count:=src.readline():
            n=int(count); header=src.readline()
            regimes=[s for s in shlex.split(header.decode()) if s.startswith('regime=')]
            if n<1 or regimes not in (['regime=target'],['regime=hot']):raise ValueError('invalid fixed regime/count')
            out=target if regimes==['regime=target'] else hard
            out.write(count);out.write(header)
            for _ in range(n):
                row=src.readline()
                if not row:raise ValueError('truncated fixed input')
                out.write(row)
        if not target.tell() or not hard.tell():raise ValueError('fixed source must contain both regimes')
PY
    cat target.xyz hard.xyz > fixed.xyz; printf '%s\n' "$system" > system; sha256sum system *.xyz *.json student.model > inputs.sha256
    mv -T "$tmp" "$out"; trap - EXIT; exit
fi
if [[ $1 == _capture ]]; then
    out=$2; part=$3; repo=$(cd "$(dirname "$0")/../.." && pwd -P)
    cd "$out"; sha256sum -c inputs.sha256 >/dev/null
    case $part in train|valid) split=$part;; fixed) split=fixed-student-test;; *) exit 2;; esac
    digest=$(python3 -c 'import json;print(json.load(open("training-manifest.json"))["student_sha256"])'); guard=/mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py
    export PYTHONPATH=$repo LOUP_LIB=$repo/build/sm_120/libloupmd.so TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
    python3 "$guard" status
    exec python3 "$guard" run --purpose "Stage 0 $(cat system) trigger $part" --wait-minutes 20 -- "$HOME/environments/mace_cueq_latest/bin/python" -m loupmd.distill.trigger_features --model student.model --sha256 "$digest" --device cuda --split "$split" --frames "$part.xyz" --out "$part-features"
fi
export MSYS2_ARG_CONV_EXCL='*'
repo=$(cd "$(dirname "$0")/../.." && pwd -P); cd "$repo"; self=$(wsl.exe -e wslpath -u "$(cygpath -w "$repo/tools/dev/score_stage0_triggers.sh")" | tr -d '\r\n')
action=$1; shift
case $action in
 prepare|capture) exec wsl.exe -e bash "$self" "_$action" "$@";;
 score)
    out=$1; shift; system=$(wsl.exe -e cat "$out/system" | tr -d '\r\n'); [[ $system =~ ^[[:alnum:]_-]+$ ]]
    ssh=/c/Windows/System32/OpenSSH/ssh.exe; remote=$($ssh leto "mktemp -d /mnt/homes/zwingel/loup-s-stage0/trigger-$system-XXXXXXXX" | tr -d '\r\n'); echo "CPU artifacts: $remote" >&2
    "$ssh" leto "mkdir $remote/tree"; git archive --format=tar.gz HEAD loupmd | "$ssh" leto "tar -xzf - -C $remote/tree"
    wsl.exe -e bash -c 'cd "$1" && sha256sum -c inputs.sha256 >/dev/null && test ! -e reports' _ "$out"
    wsl.exe -e tar chzf - -C "$out" train.xyz original-valid.xyz original-teacher.xyz valid.xyz teacher.xyz target.xyz hard.xyz train-features valid-features fixed-features training-manifest.json student-gate.json | "$ssh" leto "tar xzf - -C $remote"
    cat tools/dev/score_stage0_triggers.py | "$ssh" leto "cat > $remote/score_stage0_triggers.py"
    printf -v extra ' %q' "$@"; if [[ $# == 0 ]]; then extra=''; fi
    "$ssh" leto "cd $remote && export CUDA_VISIBLE_DEVICES='' PYTHONPATH=$remote/tree PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; /mnt/homes/zwingel/environments/mace_cueq_latest/bin/python score_stage0_triggers.py --system $system --train train.xyz --valid valid.xyz --teacher teacher.xyz --target target.xyz --hard hard.xyz --training-manifest training-manifest.json --student-gate student-gate.json --original-valid original-valid.xyz --original-teacher original-teacher.xyz --train-features train-features --valid-features valid-features --fixed-features fixed-features --output reports $extra"
    "$ssh" leto "cd $remote && tar czf - reports" | wsl.exe -e tar xzf - -C "$out";;
 *) echo 'Expected prepare, capture, or score; see file header.' >&2; exit 2;;
esac
