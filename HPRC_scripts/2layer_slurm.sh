#!/bin/bash
#SBATCH --job-name=fermi_hls4ml
#SBATCH --time=05:00:00
#SBATCH --mem=480G
#SBATCH --ntasks=24                # Total tasks (each task runs 1 Python job)
#SBATCH --cpus-per-task=4          # Each Python process gets 4 CPUs
#SBATCH --nodes=1
#SBATCH --output=out
#SBATCH --error=error
#SBATCH --partition=staff 

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
export LD_PRELOAD="/lib/x86_64-linux-gnu/libudev.so.1"
export VENV_PATH="${REPO_DIR}/HPRC_scripts/venv/bin/activate"
export VIVADO_SETUP_PATH="/sw/hprc/sw/amd/Vivado/2024.2/settings64.sh"
export HLS_PROJ_OUT="${REPO_DIR}/hlsproj/output"

source $MODULE_PATH

# Activate virtual environment
if [ -f $VENV_PATH ]; then
    source $VENV_PATH
    echo "Activated virtual environment."
else
    echo "Warning: Virtual environment not found."
fi

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
    srun -n4 python "${REPO_DIR}/iter_manager.py" \
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

