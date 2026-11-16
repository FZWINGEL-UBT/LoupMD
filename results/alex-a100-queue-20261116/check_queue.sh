#!/bin/bash
# State and key result lines of the A100 validation queue listed in queue.txt (see README.md next to this file).
# Runs on Alex: ssh alex 'bash /home/atuin/v130cb/v130cb18/loup-s64/alex64/check_queue.sh'
# Per job: its Slurm state, then the A/B summaries (MEAN lines of ab_libs.py: b/a below 1 = side b faster, with both
# orders), ab_sides.py ratios, and per gate run the row count, failing rows (printed) and worst force / stress ratio
# (the 3x rule fails a row at 3.0), plus errors, time-outs and the "#####" end markers.
W=/home/atuin/v130cb/v130cb18/loup-s64
L=${1:-$(dirname "$0")/queue.txt}
ids=$(awk '{print $1}' "$L" | paste -sd, -)
sacct -X -j "$ids" --format=JobID,JobName%14,State%12,Elapsed,Start,End 2>/dev/null
while read -r id name; do
  f=$W/runs/slurm-loup-$name-$id.out
  echo "=== $name ($id): $f"
  [ -f "$f" ] || { echo "    (no log yet)"; continue; }
  awk -F'|' '
    function flush() { if (n > 0) printf "        rows %d, failing %d, worst force %.2f, worst stress %.2f\n", n, nf, wf, ws; n = nf = wf = ws = 0 }
    /^=== [0-9]/ { print "    " $0; next }
    /^--- / { flush(); print "    " $0; next }
    /\| *(PASS|FAIL) *$/ { split($3, a, " "); split($4, b, " "); n++
                          if ($5 ~ /FAIL/) { nf++; print "        " $0 }
                          if (a[2] + 0 > wf) wf = a[2] + 0; if (b[2] + 0 > ws) ws = b[2] + 0; next }
    /MEAN|ms\/step  ratio|conv forward us:|^##### |rror|Traceback|CANCELLED|rc=[1-9]/ { flush(); print "    " $0 }
    END { flush() }' "$f"
done < "$L"
