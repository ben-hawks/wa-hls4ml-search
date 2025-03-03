#!/bin/bash
# run_wa_hls4ml.sh
# This script mimics the Kubernetes Job for running wa-hls4ml-search.
# Usage: ./run_wa_hls4ml.sh START END
#   Runs all configs from START to END in parallel

set -euo pipefail

# Check that a configuration argument was provided.
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 START_CONFIG END_CONFIG"
    exit 1
fi

START_CONFIG="$1"
END_CONFIG="$2"

# -------------------------------
# 1. Set up the environment
# -------------------------------

# Export any required environment variables.
export REPO_DIR="${SCRATCH}/wa-hls4ml-search/"
export MODULE_PATH="${REPO_DIR}/HPRC_scripts/modules.sh"
export LD_PRELOAD="/lib/x86_64-linux-gnu/libudev.so.1"
export VENV_PATH="${REPO_DIR}/HPRC_scripts/venv/bin/activate"
export VIVADO_SETUP_PATH="/sw/hprc/sw/amd/Vivado/2024.2/settings64.sh"
export HLS_PROJ_OUT="${REPO_DIR}/hlsproj/output"

#load modules
source $MODULE_PATH

# Activate the virtual environment.
if [ -f $VENV_PATH ]; then
    source $VENV_PATH
    echo "Activated virtual environment."
else
    echo "Warning: /venv/bin/activate not found."
fi

# Source Vivado settings.
if [ -f $VIVADO_SETUP_PATH ]; then
    source $VIVADO_SETUP_PATH
else
    echo "Warning: Vivado settings script not found at /tools/Xilinx/Vivado/2020.1/settings64.sh."
fi

# -------------------------------
# 2. Initialize the repository
# -------------------------------

#if [ -d "${REPO_DIR}" ]; then
#    echo "Repository found at ${REPO_DIR}. Pulling latest changes..."
#    cd "${REPO_DIR}"
#    git pull
#    cd - > /dev/null
#else
#    echo "Repository not found. Cloning repository into ${REPO_DIR}..."
#  git clone https://github.com/ben-hawks/wa-hls4ml-search -b main "${REPO_DIR}"
#fi

# -------------------------------
# 3. Run the main Python job
# -------------------------------

echo "Starting the iter_manager.py job..."

for (( CONFIG=START_CONFIG; CONFIG<END_CONFIG; CONFIG++));
do
python "${REPO_DIR}/iter_manager.py" \
    -f "${REPO_DIR}/pregen_2layer_models/filelist_${CONFIG}.csv" \
    -o "${REPO_DIR}/output/2layer_run_vsynth" \
    --hls4ml_strat resource \
    --rf_upper 4097 \
    --rf_lower 1024 \
    --rf_step 1024 \
    --prefix $REPO_DIR \
    --hlsproj $HLS_PROJ_OUT \
    --vsynth > "logs/${CONFIG}.log" &
done
wait

echo "Job completed."

