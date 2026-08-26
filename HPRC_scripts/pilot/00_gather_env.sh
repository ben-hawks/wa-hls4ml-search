#!/bin/bash
# Phase 0 pilot, step 1 of 3: static ACES environment facts.
#
# Read-only, safe to run directly on a login node -- nothing here submits a job or does
# real computation, so it's fine outside sbatch (see planning/golden_rules.md #5's
# interactive-session limits, which this stays well under).
#
# Usage (run with `bash`, NOT `source` -- sourcing this changes your interactive
# shell's directory and shell options for no benefit here):
#   bash 00_gather_env.sh
# Output: pilot_00_env_<timestamp>.txt in the current directory.

set -uo pipefail

OUT="pilot_00_env_$(date +%Y%m%d_%H%M%S).txt"

{
echo "=== $(date) ==="
echo "=== whoami / groups ==="
id

echo
echo "=== SLURM cluster config (MaxArraySize is the key number this plan needs) ==="
scontrol show config 2>&1 | grep -iE 'MaxArraySize|MaxJobCount|SchedulerParameters' \
    || echo "(scontrol show config failed, or found none of those keys -- paste the full command's error if so)"

echo
echo "=== Partition / queue info ==="
sinfo -o "%P %l %D %c %m %a" 2>&1

echo
echo "=== Current queue occupancy for \$USER (baseline, before any pilot jobs) ==="
echo "running-job count: $(squeue -u "$USER" -h 2>/dev/null | wc -l)"
squeue -u "$USER" -o "%.10i %.9P %.20j %.8T %.10M %.6D %R" 2>&1

echo
echo "=== Project account / SU balance ==="
myproject -l 2>&1 || echo "(myproject not found or failed)"

echo
echo "=== Storage quota ==="
showquota 2>&1 || echo "(showquota not found or failed)"

echo
echo "=== Modules referenced by HPRC_scripts/modules.sh ==="
for m in GCC/11.2.0 Python/3.9.6 ncurses/6.2 Miniconda3; do
    echo "--- module spider $m ---"
    module spider "$m" 2>&1
done

echo
echo "=== Vivado 2024.2 path (hardcoded in the *_slurm.sh scripts) ==="
ls -la /sw/hprc/sw/amd/Vivado/2024.2/settings64.sh 2>&1

echo
echo "=== wa-hls4ml conda env ==="
conda env list 2>&1 | grep -i wa-hls4ml || echo "(wa-hls4ml conda env not found -- see HPRC_scripts/README.md setup)"

echo
echo "=== Repo location sanity check ==="
echo "Expecting: \${SCRATCH}/wa-hls4ml-search"
ls -la "${SCRATCH}/wa-hls4ml-search" 2>&1 | head -5

echo
echo "=== dense_latency_fast_small corpus (used by 02_synthesis_timing_pilot.sh) ==="
ls "${SCRATCH}/wa-hls4ml-search/dense_latency_fast_small/" 2>&1 | head -5

} | tee "$OUT"

echo
echo "Done -- wrote $OUT"
