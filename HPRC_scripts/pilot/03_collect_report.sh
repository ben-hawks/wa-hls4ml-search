#!/bin/bash
# Phase 0 pilot, final step: bundle every artifact from 00/01/02 into one file to send
# back. Run this from the same directory you ran the other three scripts in.
#
# Usage (run with `bash`, NOT `source`): bash 03_collect_report.sh

set -uo pipefail

OUT="pilot_full_report_$(date +%Y%m%d_%H%M%S).txt"

{
echo "############################################################"
echo "# 00_gather_env"
echo "############################################################"
if ls pilot_00_env_*.txt >/dev/null 2>&1; then
    cat pilot_00_env_*.txt
else
    echo "(not found -- did you run 00_gather_env.sh?)"
fi

echo
echo "############################################################"
echo "# 01_array_concurrency_test"
echo "############################################################"
if ls -d pilot_01_array_concurrency_*/ >/dev/null 2>&1; then
    for d in pilot_01_array_concurrency_*/; do
        echo "--- ${d}report.txt ---"
        cat "${d}report.txt" 2>/dev/null
        echo
        echo "--- ${d}poll_log.csv ---"
        cat "${d}poll_log.csv" 2>/dev/null
    done
else
    echo "(not found -- did you run 01_array_concurrency_test.sh?)"
fi

echo
echo "############################################################"
echo "# 02_synthesis_timing_pilot"
echo "############################################################"
# Tagged per parallelism level (pilot_02_p<P>_*) so multiple comparison runs (e.g.
# P=4 vs P=2) each show up separately instead of one overwriting another. Also picks
# up the untagged pilot_02_{prepare,submit,...}.log names from before this tagging
# convention existed, so an earlier P=4 run collected before this script was updated
# doesn't silently disappear from future reports.
found_any=0

if [ -f pilot_02_prepare.log ]; then
    found_any=1
    echo "=== pilot_02 (untagged -- from before per-P tagging existed) ==="
    for suffix in prepare.log submit.log status.log sacct.csv seff.txt; do
        f="pilot_02_${suffix}"
        echo "--- $f ---"
        if [ -f "$f" ]; then
            cat "$f"
        else
            echo "(missing)"
        fi
        echo
    done
fi

if ls pilot_02_p*_prepare.log >/dev/null 2>&1; then
    found_any=1
    for prepare_log in pilot_02_p*_prepare.log; do
        tag=$(basename "$prepare_log" "_prepare.log")  # e.g. pilot_02_p4
        echo "=== $tag ==="
        for suffix in prepare.log submit.log status.log sacct.csv seff.txt; do
            f="${tag}_${suffix}"
            echo "--- $f ---"
            if [ -f "$f" ]; then
                cat "$f"
            else
                echo "(missing)"
            fi
            echo
        done
    done
fi

if [ "$found_any" -eq 0 ]; then
    echo "(not found in $(pwd) -- did you run 02_synthesis_timing_pilot.sh or recover_run.sh"
    echo "without --dry-run, and are you running this from the same directory as those?)"
fi
} > "$OUT"

echo "Wrote $OUT ($(wc -l < "$OUT") lines)."
echo "Paste its full contents back into chat, or send the file itself."
