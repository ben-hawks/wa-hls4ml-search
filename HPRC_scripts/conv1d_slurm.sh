#!/bin/bash
#SBATCH --job-name=fermi_hls4ml_conv1d
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=16384
#SBATCH --cpus-per-task=2       
#SBATCH --output=out
#SBATCH --error=error


set -euo pipefail

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 START_CONFIG END_CONFIG"
    exit 1
fi

START_CONFIG="$1"
END_CONFIG="$2"

# -------------------------------
# 1. Set up the environment
# -------------------------------
export REPO_DIR="${SCRATCH}/wa-hls4ml-search/"
export MODULE_PATH="${REPO_DIR}/HPRC_scripts/modules.sh"
#export LD_PRELOAD="/lib/x86_64-linux-gnu/libudev.so.1"
export VENV_PATH="${REPO_DIR}/HPRC_scripts/venv/bin/activate"
export VIVADO_SETUP_PATH="/sw/hprc/sw/amd/Vivado/2024.2/settings64.sh"
export HLS_PROJ_OUT="${REPO_DIR}/hlsproj/output"

source $MODULE_PATH

# Activate virtual environment
#if [ -f $VENV_PATH ]; then
#    source $VENV_PATH
#    echo "Activated virtual environment."
#else
#    echo "Warning: Virtual environment not found."
#fi

source activate wa-hls4ml
which python


# Source Vivado settings
if [ -f $VIVADO_SETUP_PATH ]; then
    source $VIVADO_SETUP_PATH
else
    echo "Warning: Vivado settings script not found."
fi
# -------------------------------
# 2. Run the main Python job
# -------------------------------
echo "Starting the iter_manager.py job..."
for (( CONFIG=START_CONFIG; CONFIG<END_CONFIG; CONFIG++ )); do
    srun -N 1 -n 1 -c 2 --overlap conda run --no-capture-output --name wa-hls4ml python3 "${REPO_DIR}/iter_manager_v2.py" \
        -f "${REPO_DIR}/conv1d_models/conv1d_batch_$CONFIG.json" \
        -o "${REPO_DIR}/output/conv1d_run_vsynth_2023-2" \
        --hls4ml_strat resource \
        --rf_upper 32769 \
        --rf_lower 8192 \
        --rf_step 8192 \
        --prefix $REPO_DIR \
        --hlsproj $HLS_PROJ_OUT \
        --vsynth > "logs/conv1d_$CONFIG.log" &
done
wait
echo "Job completed."
