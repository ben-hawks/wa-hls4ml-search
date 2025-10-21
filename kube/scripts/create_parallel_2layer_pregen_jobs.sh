#! /bin/bash

mkdir -p ../kube/pregen_2layer_jobs_2023
NUM_FILES=57
for i in $(seq 1 $NUM_FILES)
do
  cat ../kube/wa-hls4ml-search-job-template-2layer-2023.yml | sed "s/\$CONFIG/$i/" > ../kube/pregen_2layer_jobs_2023/job-$i.yaml
done
