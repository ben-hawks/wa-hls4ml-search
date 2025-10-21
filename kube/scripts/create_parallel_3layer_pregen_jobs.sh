#! /bin/bash

mkdir -p ../kube/pregen_3layer_jobs
NUM_FILES=132
for i in $(seq 1 $NUM_FILES)
do
  cat ../kube/wa-hls4ml-search-job-template-3layer.yml | sed "s/\$CONFIG/$i/" > ../kube/pregen_3layer_jobs/job-$i.yaml
done
