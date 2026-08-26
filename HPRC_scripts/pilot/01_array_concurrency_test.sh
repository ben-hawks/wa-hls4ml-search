#!/bin/bash
# Phase 0 pilot, step 2 of 3: does a SLURM job array count each task individually
# against ACES's 40-concurrent-running-job cap (planning/golden_rules.md #3), or does
# the whole array count as one job against that cap? This determines the real ceiling
# for run_synthesis_array.py's --array-concurrency.
#
# Cost: near-zero -- this submits pure `sleep` tasks, no Vivado/hls4ml/real synthesis
# involved. Requests a 60-task array at %60 (i.e. no self-imposed throttle below the
# 40-job cap), specifically so the test can observe whether SLURM caps concurrently
# RUNNING tasks at ~40 on its own.
#
# Usage (run with `bash`, NOT `source` -- sourcing this changes your interactive
# shell's directory and shell options, and a mid-script `exit` would close your shell):
#   bash 01_array_concurrency_test.sh
# Blocks for a few minutes (until the test array finishes), then writes a report.

set -uo pipefail

ACCOUNT="157537460776"
N_TASKS=60
CONCURRENCY=60
SLEEP_SECS=90

# Anchored to an absolute path resolved once up front, same reasoning as
# 02_synthesis_timing_pilot.sh's PILOT_DIR -- keeps output location independent of any
# cwd surprise later in the script.
OUT="$(pwd)/pilot_01_array_concurrency_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/poll_log.csv"
SCRIPT="$OUT/test_array.sh"

cat > "$SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=pilot_array_test
#SBATCH --account=$ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:05:00
#SBATCH --array=0-$((N_TASKS - 1))%$CONCURRENCY
#SBATCH --output=$OUT/task_%a.out
sleep $SLEEP_SECS
echo "task \$SLURM_ARRAY_TASK_ID done"
EOF

echo "Submitting a $N_TASKS-task array (%$CONCURRENCY requested concurrency, ${SLEEP_SECS}s/task)..."
SUBMIT_OUT=$(sbatch "$SCRIPT")
echo "$SUBMIT_OUT"
JOB_ID=$(echo "$SUBMIT_OUT" | awk '{print $NF}')
if [ -z "$JOB_ID" ]; then
    echo "sbatch failed -- nothing to poll. Check the sbatch output above." >&2
    exit 1
fi

echo "job_id,timestamp,running_this_job,pending_this_job,running_total_user" > "$LOG"

# Tracked inline as the loop runs, not by re-reading $LOG afterward -- avoids depending
# on a second read of a file this same script just wrote (the class of bug that broke
# 02_synthesis_timing_pilot.sh under `source`). $LOG is still written for the full
# timeseries record, just not load-bearing for this headline number.
MAX_CONCURRENT=0

echo "Polling every 5s until job $JOB_ID finishes..."
while true; do
    STATE_LINES=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null)
    RUNNING_THIS=$(printf '%s\n' "$STATE_LINES" | grep -c RUNNING || true)
    PENDING_THIS=$(printf '%s\n' "$STATE_LINES" | grep -c PENDING || true)
    RUNNING_TOTAL_USER=$(squeue -u "$USER" -t RUNNING -h 2>/dev/null | wc -l)
    TS=$(date +%s)
    echo "$JOB_ID,$TS,$RUNNING_THIS,$PENDING_THIS,$RUNNING_TOTAL_USER" >> "$LOG"
    echo "  t=$TS  running(this job)=$RUNNING_THIS  pending(this job)=$PENDING_THIS  running(all \$USER jobs)=$RUNNING_TOTAL_USER"
    if [ "$RUNNING_THIS" -gt "$MAX_CONCURRENT" ]; then
        MAX_CONCURRENT=$RUNNING_THIS
    fi
    if [ -z "$STATE_LINES" ]; then
        echo "Job $JOB_ID has no more tasks in the queue -- done."
        break
    fi
    sleep 5
done

{
echo "=== Array concurrency pilot report ==="
echo "job_id: $JOB_ID"
echo "requested array size: $N_TASKS"
echo "requested concurrency (%): $CONCURRENCY"
echo "observed max concurrently-RUNNING tasks for this job: $MAX_CONCURRENT"
echo
echo "Interpretation:"
echo "  If max_concurrent got close to $CONCURRENCY -> array tasks do NOT count"
echo "  individually against the 40-job cap (or nothing else was competing for it)."
echo "  If max_concurrent plateaued near 40 -> array tasks DO count individually"
echo "  against the cap, and --array-concurrency should stay well under that,"
echo "  accounting for whatever else is running under this account at the same time."
echo
echo "Full poll log: $LOG"
} | tee "$OUT/report.txt"

echo
echo "Send back: $OUT/report.txt and $OUT/poll_log.csv"
