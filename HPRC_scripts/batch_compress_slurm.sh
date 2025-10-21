#!/bin/bash
#SBATCH --job-name=fermi_hls4ml_batch_compress
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --cpus-per-task=16       
#SBATCH --output=batch_compress_out
#SBATCH --error=batch_compress_error
#SBATCH --account=157537460776
set -euo pipefail
# Check arguments
BC_SOURCE="$1"
BC_TARGET="$2"
# -------------------------------
# 1. Set up the environment
# -------------------------------
export REPO_DIR="${SCRATCH}/wa-hls4ml-search/"
export MODULE_PATH="${REPO_DIR}/HPRC_scripts/modules.sh"
#export LD_PRELOAD="/lib/x86_64-linux-gnu/libudev.so.1"
export VENV_PATH="${REPO_DIR}/HPRC_scripts/venv/bin/activate"
export CURL_CA_BUNDLE=''
source $MODULE_PATH

module load WebProxy

export http_proxy=http://10.71.8.1:8080
export https_proxy=http://10.71.8.1:8080

# -------------------------------
# 2. Run the main Python job
# -------------------------------
echo "Starting the batch compress job... compressing ${BC_SOURCE} into ${BC_TARGET} - index csv is ${BC_TARGET}.csv"
srun -N 1 -n 1 -c 2 --overlap conda run --no-capture-output --name wa-hls4ml python3 ${REPO_DIR}/util/batch_compress_files.py ${BC_SOURCE} ${BC_TARGET} 100 ${BC_TARGET}.csv --use-pigz --pigz-cores 16
wait
echo "Job completed."
