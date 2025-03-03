#!/bin/bash
# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 START_CONFIG END_CONFIG"
    exit 1
fi

export NUM_TASKS=$(($2 - $1))

sbatch --ntasks=$NUM_TASKS 2layer_slurm.sh $1 $2

