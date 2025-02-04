#! /bin/bash

START_JOB=$1
STOP_JOB=$2

for i in $(seq $START_JOB $STOP_JOB)
do
  kubectl delete -f ../kube/pregen_3layer_jobs/job-$i.yaml
done