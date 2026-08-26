#!/bin/bash
# Phase 0 pilot, step 3 of 3: real synthesis timing/memory pilot, using
# run_synthesis_array.py against dense_latency_fast_small -- confirmed zero overlap
# with the already-published dataset and the lowest-risk corpus to burn real SUs on
# (planning/dataset_gen_plan.md #2b).
#
# COST: this runs real Vivado synthesis (--vsynth). Capped to 20 units via --limit
# regardless of how big the underlying batch file turns out to be (the first
# dense_latency_fast_small batch file has been observed to contain 200 models, not the
# 50 repo_notes.md's table describes -- --limit keeps this pilot's cost small and
# predictable either way). Review this script (and the rendered array script it prints
# via --dry-run) before letting it submit for real.
#
# Usage (run with `bash`, NOT `source` -- sourcing this changes your interactive
# shell's directory and shell options, and a mid-script `exit` would close your shell):
#   bash 02_synthesis_timing_pilot.sh --dry-run   # prepare + render only, no sbatch, no cost
#   bash 02_synthesis_timing_pilot.sh              # the real thing -- submits and blocks
#                                                   # until done (historically tens of
#                                                   # minutes per model; could be a while)

set -uo pipefail

DRY_RUN=""
if [ "${1:-}" == "--dry-run" ]; then
    DRY_RUN="--dry-run"
fi

# Anchor log file locations to wherever this script was invoked from, resolved once up
# front -- everything below is written via an absolute path under $PILOT_DIR rather than
# a bare relative filename, so log output can't end up somewhere unexpected regardless
# of any `cd` later in this script (or any cwd quirk from how it was invoked).
PILOT_DIR="$(pwd)"

REPO_DIR="${REPO_DIR:-${SCRATCH}/wa-hls4ml-search}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/group/p.cis250242.000/wa-hls4ml}"
HLS_PROJ_OUT="${HLS_PROJ_OUT:-${PROJECT_DIR}/hlsproj/output}"
OUT_DIR="${PROJECT_DIR}/output/pilot_synthtest_run_vsynth_2024-2"

source "${REPO_DIR}/HPRC_scripts/modules.sh"
source activate wa-hls4ml
VIVADO_SETUP_PATH="/sw/hprc/sw/amd/Vivado/2024.2/settings64.sh"
if [ -f "$VIVADO_SETUP_PATH" ]; then
    source "$VIVADO_SETUP_PATH"
else
    echo "Warning: Vivado settings script not found at $VIVADO_SETUP_PATH" >&2
fi

BATCH_FILE=$(ls "${REPO_DIR}"/dense_latency_fast_small/dense_latency_fast_batch_*.json 2>/dev/null | head -1)
if [ -z "$BATCH_FILE" ]; then
    echo "Couldn't find a dense_latency_fast_small batch file under ${REPO_DIR}/dense_latency_fast_small/" >&2
    exit 1
fi
echo "Using batch file: $BATCH_FILE (--limit 20 below caps the actual pilot size)"

cd "$REPO_DIR"

echo "=== Preparing pilot joblist (RF=1 only, matching dense_latency_fast_slurm.sh; --limit 20) ==="
# Captured into a variable (not re-read from the log file afterward) specifically so
# extracting RUN_DIR doesn't depend on a second, separate read of a just-written file --
# that dependency is exactly what broke when this script was run with `source` instead
# of `bash` (the write and the re-read ended up seeing different working directories).
PREPARE_OUTPUT=$(python3 run_synthesis_array.py --prepare \
    --arch pilot_synthtest \
    --batch-glob "$BATCH_FILE" \
    --output "$OUT_DIR" \
    --hlsproj "$HLS_PROJ_OUT" \
    --strat latency --rf-lower 1 --rf-upper 2 --rf-step 1 \
    --run-dir-root "${PROJECT_DIR}/output/_pilot_runs" \
    --limit 20 2>&1)
echo "$PREPARE_OUTPUT"
echo "$PREPARE_OUTPUT" > "${PILOT_DIR}/pilot_02_prepare.log"

RUN_DIR=$(echo "$PREPARE_OUTPUT" | grep "Run directory:" | awk '{print $NF}')
if [ -z "$RUN_DIR" ]; then
    echo "Couldn't find 'Run directory:' in --prepare's output -- see ${PILOT_DIR}/pilot_02_prepare.log for the full log and any error." >&2
    exit 1
fi
echo "Run dir: $RUN_DIR"

echo "=== Rendering array script (K=5 units/chunk, P=4 parallel/chunk, %5 concurrency) ==="
python3 run_synthesis_array.py --submit "$RUN_DIR" \
    --units-per-chunk 5 --units-parallel 4 --array-concurrency 5 \
    $DRY_RUN 2>&1 | tee "${PILOT_DIR}/pilot_02_submit.log"

if [ -n "$DRY_RUN" ]; then
    echo
    echo "--dry-run: nothing was submitted. Review the rendered script above, then re-run"
    echo "without --dry-run to actually submit (real SUs will be spent)."
    exit 0
fi

JOB_ID=$(cat "$RUN_DIR/slurm_job_id.txt" 2>/dev/null || echo "")
echo "Slurm job id: $JOB_ID"

echo "=== Final status ==="
python3 run_synthesis_array.py --status "$RUN_DIR" 2>&1 | tee "${PILOT_DIR}/pilot_02_status.log"

if [ -n "$JOB_ID" ]; then
    echo "=== sacct timing/memory detail ==="
    sacct -j "$JOB_ID" --format=JobID,JobName%20,Elapsed,TotalCPU,MaxRSS,ReqMem,State,ExitCode -P \
        | tee "${PILOT_DIR}/pilot_02_sacct.csv"

    echo "=== seff per completed array task (up to 10 sampled) ==="
    {
    for tid in $(sacct -j "$JOB_ID" --format=JobID -n -P | grep -E "^${JOB_ID}_[0-9]+$" | head -10); do
        echo "--- seff $tid ---"
        seff "$tid" 2>&1
    done
    } | tee "${PILOT_DIR}/pilot_02_seff.txt"
fi

echo
echo "Send back: ${PILOT_DIR}/pilot_02_{prepare.log,submit.log,status.log,sacct.csv,seff.txt}"
