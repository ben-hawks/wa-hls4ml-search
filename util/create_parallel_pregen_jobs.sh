#! /bin/bash

mkdir -p ../kube/pregen_latency_jobs
NUM_FILES=82
for i in $(seq 1 $NUM_FILES)
do
  cat ../kube/wa-hls4ml-search-job-template-latency.yml | sed "s/\$CONFIG/$i/" > ../kube/pregen_latency_jobs/job-$i.yaml
done
