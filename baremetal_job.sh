#!/usr/bin/env bash

cd /root/
cd automl_benchmark
source automl_benchmark/bin/activate


timeout 12600 python baremetal_wrapper.py

aws s3 cp /home/ec2-user/logs.out s3://${S3_BUCKET}/${S3_FOLDER}h2o_logs/${TASK}/$(date +%Y%m%d%H%M%S).log

cp /home/ec2-user/logs.out /home/ec2-user/${TASK}.log
rm /home/ec2-user/logs.out
