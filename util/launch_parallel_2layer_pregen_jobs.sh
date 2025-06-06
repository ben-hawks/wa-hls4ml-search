#! /bin/bash

START_JOB=$1
STOP_JOB=$2

for i in $(seq $START_JOB $STOP_JOB)
do
  sleep 1 # sleep for 1 to try and reduce the chance of the job init failing
  kubectl apply -f ../kube/pregen_2layer_jobs_2023/job-$i.yaml
done