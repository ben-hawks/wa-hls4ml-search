#!/bin/bash
# Recover from a killed terminal/tmux/SSH session that was running
# 02_synthesis_timing_pilot.sh (or any run_synthesis_array.py --submit call): reattach
# to the SLURM array that kept running independently, wait for it to finish, then
# collect the same sacct/seff/status artifacts the original script's tail would have
# written -- with the same tagged filenames so 03_collect_report.sh still finds them.
#
# Usage (run with `bash`, NOT `source`):
#   bash recover_run.sh <RUN_DIR> [TAG]
#
# RUN_DIR: the run_<arch>_<timestamp> directory from the --prepare step (find it under
#   <output>/_pilot_runs/ or <output>/_runs/ if you don't have it handy -- see below).
# TAG: optional; defaults to whatever's after "pilot_synthtest_" in the run dir's arch
#   name (e.g. "p2" for a run created with --arch pilot_synthtest_p2). Only matters for
#   naming the output files consistently with 02_synthesis_timing_pilot.sh's convention.
#
# If you don't know the RUN_DIR, find the most recent one:
#   ls -dt "${PROJECT_DIR:-/scratch/group/p.cis250242.000/wa-hls4ml}"/output/*/_pilot_runs/run_*/ | head -1

set -uo pipefail

RUN_DIR="${1:?Usage: bash recover_run.sh <RUN_DIR> [TAG]}"
if [ ! -d "$RUN_DIR" ]; then
    echo "No such directory: $RUN_DIR" >&2
    exit 1
fi

TAG="${2:-}"
if [ -z "$TAG" ]; then
    # Best-effort: pull "pN" out of the manifest's arch field (e.g. "pilot_synthtest_p2" -> "p2").
    TAG=$(python3 -c "
import json
try:
    m = json.load(open('$RUN_DIR/manifest.json'))
    arch = m.get('arch', '')
    print(arch.rsplit('_', 1)[-1] if '_p' in arch else 'recovered')
except Exception:
    print('recovered')
" 2>/dev/null)
    TAG="${TAG:-recovered}"
fi

PILOT_DIR="$(pwd)"
echo "Recovering $RUN_DIR (tag: $TAG) into ${PILOT_DIR}/pilot_02_${TAG}_*"

REPO_DIR="${REPO_DIR:-${SCRATCH}/wa-hls4ml-search}"
cd "$REPO_DIR"

JOB_ID_PATH="$RUN_DIR/slurm_job_id.txt"
if [ ! -f "$JOB_ID_PATH" ]; then
    echo "No slurm_job_id.txt in $RUN_DIR -- this run was never actually submitted "
    echo "(the session likely died before sbatch ran, or during --prepare). Nothing is"
    echo "in flight -- it's safe to just re-run 02_synthesis_timing_pilot.sh fresh."
    exit 1
fi
JOB_ID=$(cat "$JOB_ID_PATH")
echo "Found job $JOB_ID -- reattaching and waiting for it to finish (this blocks until done)..."

python3 run_synthesis_array.py --wait "$RUN_DIR" 2>&1 | tee "${PILOT_DIR}/pilot_02_${TAG}_status.log"

echo "=== sacct timing/memory detail ==="
sacct -j "$JOB_ID" --format=JobID,JobName%20,Elapsed,TotalCPU,MaxRSS,ReqMem,State,ExitCode -P \
    | tee "${PILOT_DIR}/pilot_02_${TAG}_sacct.csv"

echo "=== seff per completed array task (up to 10 sampled) ==="
{
for tid in $(sacct -j "$JOB_ID" --format=JobID -n -P | grep -E "^${JOB_ID}_[0-9]+$" | head -10); do
    echo "--- seff $tid ---"
    seff "$tid" 2>&1
done
} | tee "${PILOT_DIR}/pilot_02_${TAG}_seff.txt"

echo
echo "Recovered. Send back: ${PILOT_DIR}/pilot_02_${TAG}_{status.log,sacct.csv,seff.txt}"
echo "(prepare.log/submit.log from the original run may still be sitting in whatever"
echo "directory that session was in, if it got that far before dying -- check there too.)"
