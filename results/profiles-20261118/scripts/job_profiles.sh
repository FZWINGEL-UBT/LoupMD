#!/usr/bin/env bash
# In-depth profiles of the round's library on the local RTX 5060 Ti (sm_120), for the next idea round:
#   1. nsys: per-kernel time per MD step (300 K Langevin, 1 fs, skin 0.5 A; 100 warm-up steps, then STEPS captured)
#      for every benchmark family, full kernel tables kept (nsys/<label>_cuda_gpu_kern_sum.csv);
#   2. GEMM shapes with their tuned times (LOUP_DEBUG_GEMM, fresh plans in a scratch dir) for the GEMM-heavy families;
#   3. ncu sections (speed of light, memory workload, warp states, scheduler, occupancy, launch, instructions) for the
#      top kernels of one force call (after 20 warm-up calls) of five families.
# usage: LIB=<name of libloupmd_<name>.so> [ONLY=<label regex>] bash job_profiles.sh [nsys|gemm|ncu]...
set -uo pipefail
L=/mnt/d/Margraf/LoupMD
R=$L/results/profiles-20261118
S=$R/scripts                                   # md_prof.py, prof_eval.py
T=${T:-$L/.tmp_o/profiles} && mkdir -p $T      # scratch: nsys / ncu reports, plan directories
P=/home/fzwin/environments/mace_cueq_latest/bin/python
M=/home/fzwin/models
O=/home/fzwin/loup_benchmark_env/inputs/mace-omat-0-medium
NCU=/usr/local/cuda/bin/ncu
LIB=${LIB:?LIB}
export PYTHONPATH=$L/python TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 LOUP_LIB=$L/libloupmd_$LIB.so
mkdir -p $R/nsys $R/gemm $R/ncu
echo "=== $(date '+%F %T') lib $LOUP_LIB $(sha256sum $LOUP_LIB | cut -c1-12)" | tee -a $R/libs.txt
# label:model:system:steps[:generic]
CASES="omat0m_spec_srtio3:$O:srtio3:64 omat0m_gen_srtio3:$O:srtio3:64:g mp0m_srtio3:$M/2023-12-03-mace-128-L1_epoch-199:srtio3:64
mp0m_nbcl5:$M/2023-12-03-mace-128-L1_epoch-199:nbcl5:128 mptrj_srtio3:$M/MACE_MPtrj_2022.9:srtio3:64
omat0s_srtio3:$M/mace-omat-0-small:srtio3:64 mh1_srtio3:$M/mace-mh-1:srtio3:16 mh1_nacl216:$M/mace-mh-1:nacl216:128
off23s_water216:$M/MACE-OFF23_small:water216:256 off23m_water216:$M/MACE-OFF23_medium:water216:256
off24m_water:$M/MACE-OFF24_medium:water:32 off23l_water216:$M/MACE-OFF23_large:water216:64
omol0xl_water216:$M/MACE-omol-0-extra-large-1024:water216:32 polar1s_nacl216:$M/MACE-POLAR-1-S:nacl216:128
polar1s_nbcl5:$M/MACE-POLAR-1-S:nbcl5:32 polar1m_nacl216:$M/MACE-POLAR-1-M:nacl216:128
polar1m_nbcl5:$M/MACE-POLAR-1-M:nbcl5:16 polar1l_nacl216:$M/MACE-POLAR-1-L:nacl216:64"
for part in "$@"; do
case $part in
nsys)
  for c in $CASES; do
    IFS=: read -r label m s n g <<< "$c"
    [ -n "${ONLY:-}" ] && ! [[ $label =~ $ONLY ]] && continue
    if [ "$g" = g ]; then export LOUP_GENERIC=1; else unset LOUP_GENERIC; fi
    o=$T/dp_$label
    nsys profile -t cuda --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node -f true \
      -o $o $P $S/md_prof.py $m.model $s $n 2>&1 | grep -o "PROF.*" | tee -a $R/nsys/steps.txt
    nsys stats --force-export true -r cuda_gpu_kern_sum -f csv -o $R/nsys/$label $o.nsys-rep > /dev/null 2>&1
    echo "steps=$n" > $R/nsys/$label.steps
    rm -f $o.nsys-rep $o.sqlite
  done
  unset LOUP_GENERIC ;;
gemm)
  for c in mh1_srtio3:$M/mace-mh-1:srtio3 mp0m_srtio3:$M/2023-12-03-mace-128-L1_epoch-199:srtio3 \
           omat0m_gen_srtio3:$O:srtio3:g off23l_water216:$M/MACE-OFF23_large:water216 \
           polar1m_nbcl5:$M/MACE-POLAR-1-M:nbcl5 omol0xl_water216:$M/MACE-omol-0-extra-large-1024:water216; do
    IFS=: read -r label m s g <<< "$c"
    [ -n "${ONLY:-}" ] && ! [[ $label =~ $ONLY ]] && continue
    if [ "$g" = g ]; then export LOUP_GENERIC=1; else unset LOUP_GENERIC; fi
    D=$T/plans_dp && rm -rf $D && mkdir -p $D
    LOUP_DEBUG_GEMM=1 LOUP_CACHE_DIR=$D $P $S/md_prof.py $m.model $s 4 2>&1 | grep -E "^gemm " \
      | sort | uniq -c | sort -k2 > $R/gemm/$label.txt
    echo "$label: $(wc -l < $R/gemm/$label.txt) GEMM shape lines"
    rm -rf $D
  done
  unset LOUP_GENERIC ;;
ncu)
  SECTIONS="--section SpeedOfLight --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables"
  SECTIONS="$SECTIONS --section WarpStateStats --section SchedulerStats --section Occupancy --section LaunchStats"
  SECTIONS="$SECTIONS --section InstructionStats"
  # label:model:system:regex[:generic]
  for c in "mp0m_srtio3:$M/2023-12-03-mace-128-L1_epoch-199:srtio3:gconv|first_|gprod|sgemm|fm_tf32|cutlass|assemble" \
           "omat0m_spec_srtio3:$O:srtio3:spline_forward|wprime|tabled_outer|sc0_|sc1_|sgemm|fm_tf32|cutlass|assemble" \
           "omat0m_gen_srtio3:$O:srtio3:gconv|first_|gprod|sgemm|fm_tf32|cutlass:g" \
           "mh1_srtio3:$M/mace-mh-1:srtio3:gconv|first_|gprod|gate_|sgemm|cutlass|magma|fm_tf32" \
           "off23l_water216:$M/MACE-OFF23_large:water216:gconv|first_|gprod|sgemm|fm_tf32|cutlass" \
           "polar1m_nbcl5:$M/MACE-POLAR-1-M:nbcl5:sf_kernel|grad_pass|feat_pass|inv_|ro_|sgemm|cutlass|gconv|gprod"; do
    IFS=: read -r label m s rx g <<< "$c"
    [ -n "${ONLY:-}" ] && ! [[ $label =~ $ONLY ]] && continue
    x=""; [ "$g" = g ] && x=--generic
    $NCU --profile-from-start off --target-processes all -k "regex:$rx" $SECTIONS -f -o $T/ncu_$label \
      $P $S/prof_eval.py $m.model $s $x 2>&1 | grep -E "PROF-EVAL|ERR|rror" | head -3
    $NCU --import $T/ncu_$label.ncu-rep --page details --csv 2>/dev/null | gzip -9 > $R/ncu/$label.csv.gz
    $NCU --import $T/ncu_$label.ncu-rep --page raw --csv 2>/dev/null | gzip -9 > $R/ncu/${label}_raw.csv.gz
    rm -f $T/ncu_$label.ncu-rep
  done ;;
esac
done
echo "##### done $(date '+%F %T')"
