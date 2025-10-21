#!/bin/bash
#SBATCH --job-name=fermi_hls4ml_hf_upload_project
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --cpus-per-task=16       
#SBATCH --output=hf_upload_out
#SBATCH --error=hf_upload_error
#SBATCH --account=157537460776
set -euo pipefail
# Check arguments
UPLOAD_FOLDER="$1"
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
echo "Starting the HF Dataset upload job... Uploading ${UPLOAD_FOLDER}"
srun -N 1 -n 1 -c 16 --overlap conda run --no-capture-output --name wa-hls4ml huggingface-cli upload-large-folder fastmachinelearning/wa-hls4ml-projects --repo-type=dataset $UPLOAD_FOLDER --num-workers=16
wait
echo "Job completed."
