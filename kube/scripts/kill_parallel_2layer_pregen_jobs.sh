#! /bin/bash

START_JOB=$1
STOP_JOB=$2

for i in $(seq $START_JOB $STOP_JOB)
do
  kubectl delete -f ../kube/pregen_2layer_jobs_2023/job-$i.yaml
done