#!/bin/bash
# Track wa-hls4ml FPGA synthesis progress across the five active architectures, plus any
# active run_synthesis_array.py runs. No dedicated progress-monitoring tool existed for
# this pipeline before this script -- HPRC_scripts/README.md's own workflow only had
# per-CONFIG log files under logs/ to check by hand.
#
# Usage:
#   bash check_progress.sh            -- print once and exit
#   bash check_progress.sh --watch N  -- clear and reprint every N seconds until Ctrl+C
#
# "Succeeded" reuses util/completion.py's real-success definition (non-empty
# resource_report -- the actual Vivado place-and-route report, not just the
# pre-synthesis HLS estimate that survives a failed --vsynth run), not just
# "*_processed.json exists", since a failed synthesis still writes a valid-looking
# processed JSON with empty report dicts (see run_search_iteration.py / Phase A fixes).
#
# Override PROJECT_DIR if your output lives somewhere other than the default ACES
# project allocation path the *_slurm.sh scripts use.

PROJECT_DIR="${PROJECT_DIR:-/scratch/group/p.cis250242.000/wa-hls4ml}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# ---- Colors (disabled when not a terminal) ---------------------------------------
if [[ -t 1 ]]; then
    BOLD=$'\e[1m'; DIM=$'\e[2m'; RED=$'\e[31m'; GREEN=$'\e[32m'
    YELLOW=$'\e[33m'; CYAN=$'\e[36m'; RESET=$'\e[0m'
else
    BOLD='' DIM='' RED='' GREEN='' YELLOW='' CYAN='' RESET=''
fi

hdr()  { echo "${BOLD}${CYAN}$*${RESET}"; }
good() { printf '%s' "${GREEN}$*${RESET}"; }
warn() { printf '%s' "${YELLOW}$*${RESET}"; }
bad()  { printf '%s' "${RED}$*${RESET}"; }
dim()  { printf '%s' "${DIM}$*${RESET}"; }

# cpad STR WIDTH -- left-pad STR to WIDTH visible columns, ignoring ANSI escapes.
cpad() {
    local str="$1" w="$2"
    local visible; visible=$(printf '%s' "$str" | sed 's/\x1b\[[0-9;]*m//g')
    local pad=$(( w - ${#visible} ))
    printf '%s' "$str"
    (( pad > 0 )) && printf '%*s' "$pad" ''
}

color_pct() {
    local pct="$1"
    if   (( pct >= 100 )); then printf '%s' "${BOLD}${GREEN}${pct}%${RESET}"
    elif (( pct >= 75  )); then printf '%s' "${GREEN}${pct}%${RESET}"
    elif (( pct >= 25  )); then printf '%s' "${YELLOW}${pct}%${RESET}"
    else                        printf '%s' "${RED}${pct}%${RESET}"
    fi
}

LABEL_W=20

# check_arch LABEL OUTPUT_DIR EXPECTED_UNITS
check_arch() {
    local label="$1" dir="$2" expected="$3"
    local label_col; label_col=$(cpad "$label" "$LABEL_W")

    if [ ! -d "$dir" ]; then
        printf "  %s  %s\n" "$label_col" "$(dim "not started -- $dir doesn't exist yet")"
        return
    fi

    local attempted=0 succeeded=0
    read -r attempted succeeded < <(python3 "${REPO_DIR}/util/completion.py" "$dir" 2>/dev/null)
    # Strip a stray trailing \r defensively (e.g. if python's stdout is line-ending
    # translated in some environment) -- an untrimmed \r here corrupts every
    # arithmetic expansion below with a cryptic "invalid arithmetic operator" error.
    attempted="${attempted%$'\r'}"
    succeeded="${succeeded%$'\r'}"
    attempted=${attempted:-0}
    succeeded=${succeeded:-0}
    local failed=$(( attempted - succeeded ))
    local remaining=$(( expected - attempted ))
    (( remaining < 0 )) && remaining=0

    local pct=0
    (( expected > 0 )) && pct=$(( succeeded * 100 / expected ))

    local detail="${CYAN}${succeeded}${RESET}/${expected} succeeded"
    (( failed > 0 ))    && detail="${detail}, $(bad "${failed} failed")"
    (( remaining > 0 )) && detail="${detail}, $(dim "${remaining} not attempted")"

    printf "  %s  %s  (%s)\n" "$label_col" "$detail" "$(color_pct "$pct")"
}

print_status() {
    local active_jobs; active_jobs=$(squeue -u "$USER" -h 2>/dev/null | wc -l)

    hdr "=========================================================="
    hdr " wa-hls4ml FPGA Synthesis Progress"
    echo " $(date '+%Y-%m-%d %H:%M:%S')  |  active Slurm jobs for \$USER: ${active_jobs}"
    hdr "=========================================================="
    echo ""

    # Expected totals per planning/dataset_gen_plan.md #2a (models x RF-sweep-count).
    check_arch "dense_latency"      "${PROJECT_DIR}/output/dense_latency_run_vsynth_2024-2"      66800
    check_arch "dense_resource"     "${PROJECT_DIR}/output/dense_resource_run_vsynth_2024-2"     66800
    check_arch "dense_latency_fast" "${PROJECT_DIR}/output/dense_latency_fast_run_vsynth_2024-2" 10000
    check_arch "conv1d"             "${PROJECT_DIR}/output/conv1d_run_vsynth_2024-2"             133600
    check_arch "conv2d"             "${PROJECT_DIR}/output/conv2d_run_vsynth_2024-2"             133600
    echo ""

    # Any run_synthesis_array.py run directories (see HPRC_scripts/README.md), wherever
    # they landed under this PROJECT_DIR's output tree.
    local run_dirs=() rd
    while IFS= read -r rd; do
        run_dirs+=("$rd")
    done < <(find "${PROJECT_DIR}/output" -maxdepth 3 -type d -path "*/_runs/run_*" 2>/dev/null | sort)

    if [ "${#run_dirs[@]}" -gt 0 ]; then
        hdr "[ run_synthesis_array.py runs ]"
        for rd in "${run_dirs[@]}"; do
            python3 "${REPO_DIR}/run_synthesis_array.py" --status "$rd" 2>/dev/null \
                | sed -n 's/^.*INFO - /  /p'
        done
        echo ""
    fi
}

if [ "$1" == "--watch" ]; then
    interval="${2:-15}"
    while true; do
        clear
        print_status
        echo " (refreshing every ${interval}s, Ctrl+C to stop)"
        sleep "$interval"
    done
else
    print_status
fi
