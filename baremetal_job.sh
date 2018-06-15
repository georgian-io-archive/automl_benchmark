#!/usr/bin/env bash

cd /root/
cd automl_benchmark
source automl_benchmark/bin/activate

python baremetal_wrapper.py

aws s3 cp logs.out s3://${S3_BUCKET}/${S3_FOLDER}h2o_logs/${TASK}/$(date).log
rm logs.out
