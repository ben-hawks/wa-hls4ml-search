#!/bin/bash
# Phase 0 pilot, step 3 of 3: real synthesis timing/memory pilot, using
# run_synthesis_array.py against dense_latency_fast_small -- confirmed zero overlap
# with the already-published dataset and the lowest-risk corpus to burn real SUs on
# (planning/dataset_gen_plan.md #2b).
#
# Parameterized over -P/--units-parallel, -K/--units-per-chunk, and --cpus-per-unit so
# the SAME script can run comparable pilots at different concurrency/core settings.
#
# 2026-08-26 findings: the P=4 pilot showed a 2/20 failure rate, root-caused to
# Vivado's synth_design internally multithreading up to 7 processes PER unit (visible
# in slurm_logs/task_N.out as "Multithreading enabled for synth_design using a maximum
# of 7 processes"). A P=2 follow-up (same --cpus-per-unit=2) showed the IDENTICAL 2/20
# failure rate, on a completely different, non-overlapping set of models, with the same
# exact "Vivado synthesis report not found" / missing-VivadoSynthReport signature both
# times -- ruling out a model-specific cause. The reason P alone didn't help: the
# oversubscription ratio is (P*7 threads) / (P*cpus_per_unit cores) = 7/cpus_per_unit,
# which cancels P out entirely. --cpus-per-unit is the knob that actually changes that
# ratio -- e.g. --cpus-per-unit 4 halves it (7/4 instead of 7/2).
#
# COST: this runs real Vivado synthesis (--vsynth). Capped via --limit (default 20)
# regardless of how big the underlying batch file turns out to be (the first
# dense_latency_fast_small batch file has been observed to contain 200 models, not the
# 50 repo_notes.md's table describes). Review this script (and the rendered array
# script it prints via --dry-run) before letting it submit for real.
#
# Usage (run with `bash`, NOT `source` -- sourcing this changes your interactive
# shell's directory and shell options, and a mid-script `exit` would close your shell):
#   bash 02_synthesis_timing_pilot.sh --dry-run                     # render only, no cost
#   bash 02_synthesis_timing_pilot.sh                                # P=4, K=5, cpus=2 (baseline)
#   bash 02_synthesis_timing_pilot.sh --units-parallel 2 --units-per-chunk 4  # P=2 comparison (already run)
#   bash 02_synthesis_timing_pilot.sh --cpus-per-unit 4              # P=4, K=5, cpus=4 -> ratio 7/4
#                                                                     # instead of 7/2 -- the next test
#
# Log files are tagged with parallelism + cpus-per-unit (pilot_02_p<P>c<CPUS>_*) so
# different configurations don't overwrite each other's artifacts.

set -uo pipefail

DRY_RUN=""
UNITS_PARALLEL=4
UNITS_PER_CHUNK=5
LIMIT=20
ARRAY_CONCURRENCY=5
CPUS_PER_UNIT=2

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --units-parallel|-P) UNITS_PARALLEL="$2"; shift 2 ;;
        --units-per-chunk|-K) UNITS_PER_CHUNK="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --array-concurrency) ARRAY_CONCURRENCY="$2"; shift 2 ;;
        --cpus-per-unit) CPUS_PER_UNIT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

TAG="p${UNITS_PARALLEL}c${CPUS_PER_UNIT}"

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
echo "Using batch file: $BATCH_FILE (--limit $LIMIT below caps the actual pilot size)"
WAVES=$(( (UNITS_PER_CHUNK + UNITS_PARALLEL - 1) / UNITS_PARALLEL ))
RATIO=$(python3 -c "print(f'{7/$CPUS_PER_UNIT:.2f}')" 2>/dev/null || echo "?")
echo "Config: P=$UNITS_PARALLEL K=$UNITS_PER_CHUNK (-> $WAVES wave(s)/chunk), cpus-per-unit=$CPUS_PER_UNIT (oversubscription ratio 7/$CPUS_PER_UNIT=${RATIO}x), %$ARRAY_CONCURRENCY concurrency, tag=$TAG"

cd "$REPO_DIR"

echo "=== Preparing pilot joblist (RF=1 only, matching dense_latency_fast_slurm.sh; --limit $LIMIT) ==="
# A fresh --prepare here draws a new random --limit subset (auto-excluding whatever's
# already complete from a prior pilot run against the same output dir), so a P=2 run
# after a P=4 run mostly hits different units -- comparable, not identical, samples.
#
# Captured into a variable (not re-read from the log file afterward) specifically so
# extracting RUN_DIR doesn't depend on a second, separate read of a just-written file --
# that dependency is exactly what broke when this script was run with `source` instead
# of `bash` (the write and the re-read ended up seeing different working directories).
PREPARE_OUTPUT=$(python3 run_synthesis_array.py --prepare \
    --arch "pilot_synthtest_${TAG}" \
    --batch-glob "$BATCH_FILE" \
    --output "$OUT_DIR" \
    --hlsproj "$HLS_PROJ_OUT" \
    --strat latency --rf-lower 1 --rf-upper 2 --rf-step 1 \
    --run-dir-root "${PROJECT_DIR}/output/_pilot_runs" \
    --limit "$LIMIT" 2>&1)
echo "$PREPARE_OUTPUT"
echo "$PREPARE_OUTPUT" > "${PILOT_DIR}/pilot_02_${TAG}_prepare.log"

RUN_DIR=$(echo "$PREPARE_OUTPUT" | grep "Run directory:" | awk '{print $NF}')
if [ -z "$RUN_DIR" ]; then
    echo "Couldn't find 'Run directory:' in --prepare's output -- see ${PILOT_DIR}/pilot_02_${TAG}_prepare.log for the full log and any error." >&2
    exit 1
fi
echo "Run dir: $RUN_DIR"

echo "=== Rendering array script (K=$UNITS_PER_CHUNK units/chunk, P=$UNITS_PARALLEL parallel/chunk, cpus-per-unit=$CPUS_PER_UNIT, %$ARRAY_CONCURRENCY concurrency) ==="
python3 run_synthesis_array.py --submit "$RUN_DIR" \
    --units-per-chunk "$UNITS_PER_CHUNK" --units-parallel "$UNITS_PARALLEL" \
    --array-concurrency "$ARRAY_CONCURRENCY" --cpus-per-unit "$CPUS_PER_UNIT" \
    $DRY_RUN 2>&1 | tee "${PILOT_DIR}/pilot_02_${TAG}_submit.log"

if [ -n "$DRY_RUN" ]; then
    echo
    echo "--dry-run: nothing was submitted. Review the rendered script above, then re-run"
    echo "without --dry-run to actually submit (real SUs will be spent)."
    exit 0
fi

JOB_ID=$(cat "$RUN_DIR/slurm_job_id.txt" 2>/dev/null || echo "")
echo "Slurm job id: $JOB_ID"

echo "=== Final status ==="
python3 run_synthesis_array.py --status "$RUN_DIR" 2>&1 | tee "${PILOT_DIR}/pilot_02_${TAG}_status.log"

if [ -n "$JOB_ID" ]; then
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
fi

echo
echo "Send back: ${PILOT_DIR}/pilot_02_${TAG}_{prepare.log,submit.log,status.log,sacct.csv,seff.txt}"
echo "Run directory (for a closer look at failures, e.g. slurm_logs/task_N.{out,err}): $RUN_DIR"
