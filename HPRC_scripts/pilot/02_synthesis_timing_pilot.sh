#!/bin/bash
# Phase 0 pilot, step 3 of 3: real synthesis timing/memory pilot, using
# run_synthesis_array.py against ONE batch file (50 models) from dense_latency_fast_small
# -- confirmed zero overlap with the already-published dataset and the lowest-risk
# corpus to burn real SUs on (planning/dataset_gen_plan.md #2b).
#
# COST: this runs real Vivado synthesis (--vsynth) on up to 50 models x 1 RF value.
# Not free. Review this script (and the rendered array script it prints first) before
# letting it submit.
#
# Usage:
#   bash 02_synthesis_timing_pilot.sh --dry-run   # prepare + render only, no sbatch, no cost
#   bash 02_synthesis_timing_pilot.sh              # the real thing -- submits and blocks
#                                                   # until done (historically tens of
#                                                   # minutes per model; could be a while)

set -uo pipefail

DRY_RUN=""
if [ "${1:-}" == "--dry-run" ]; then
    DRY_RUN="--dry-run"
fi

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
echo "Using batch file: $BATCH_FILE (up to 50 models)"

cd "$REPO_DIR"

echo "=== Preparing pilot joblist (RF=1 only, matching dense_latency_fast_slurm.sh) ==="
python3 run_synthesis_array.py --prepare \
    --arch pilot_synthtest \
    --batch-glob "$BATCH_FILE" \
    --output "$OUT_DIR" \
    --hlsproj "$HLS_PROJ_OUT" \
    --strat latency --rf-lower 1 --rf-upper 2 --rf-step 1 \
    --run-dir-root "${PROJECT_DIR}/output/_pilot_runs" \
    | tee pilot_02_prepare.log

RUN_DIR=$(grep "Run directory:" pilot_02_prepare.log | awk '{print $NF}')
if [ -z "$RUN_DIR" ]; then
    echo "Couldn't parse the run directory out of pilot_02_prepare.log -- check it for errors." >&2
    exit 1
fi
echo "Run dir: $RUN_DIR"

echo "=== Rendering array script (K=10 units/chunk, P=4 parallel/chunk, %5 concurrency) ==="
python3 run_synthesis_array.py --submit "$RUN_DIR" \
    --units-per-chunk 10 --units-parallel 4 --array-concurrency 5 \
    $DRY_RUN | tee pilot_02_submit.log

if [ -n "$DRY_RUN" ]; then
    echo
    echo "--dry-run: nothing was submitted. Review the rendered script above, then re-run"
    echo "without --dry-run to actually submit (real SUs will be spent)."
    exit 0
fi

JOB_ID=$(cat "$RUN_DIR/slurm_job_id.txt" 2>/dev/null || echo "")
echo "Slurm job id: $JOB_ID"

echo "=== Final status ==="
python3 run_synthesis_array.py --status "$RUN_DIR" | tee pilot_02_status.log

if [ -n "$JOB_ID" ]; then
    echo "=== sacct timing/memory detail ==="
    sacct -j "$JOB_ID" --format=JobID,JobName%20,Elapsed,TotalCPU,MaxRSS,ReqMem,State,ExitCode -P \
        | tee pilot_02_sacct.csv

    echo "=== seff per completed array task (up to 10 sampled) ==="
    {
    for tid in $(sacct -j "$JOB_ID" --format=JobID -n -P | grep -E "^${JOB_ID}_[0-9]+$" | head -10); do
        echo "--- seff $tid ---"
        seff "$tid" 2>&1
    done
    } | tee pilot_02_seff.txt
fi

echo
echo "Send back: pilot_02_prepare.log, pilot_02_submit.log, pilot_02_status.log, pilot_02_sacct.csv, pilot_02_seff.txt"
