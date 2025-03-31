#!/usr/bin/env bash

source /venv/bin/activate
source /opt/Xilinx/Vivado/2020.1/settings64.sh
echo "Activated Venv & Setup Vivado..."
python /opt/repo/wa-hls4ml-search/iter_manager.py \
-f /opt/repo/wa-hls4ml-search/pregen_2layer_models/filelist_2.csv \
-o /output/2layer_run_vsynth_test \
--hls4ml_strat resource \
--rf_upper 4097 \
--rf_lower 1024 \
--rf_step 1024 \
--vsynth