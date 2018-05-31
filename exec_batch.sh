#!/usr/bin/env bash

#Upload SSH files
aws s3 cp --recursive ssh s3://$(python config.py s3_bucket_root)/$(python config.py s3_folder)ssh

#Ensure correct file permissions
chmod 755 batch_job.sh

#Build Docker Container
docker build -t auto-ml-exploration .

#Login to aws
eval "$(aws ecr get-login --no-include-email)"

#Push to container repo
docker tag auto-ml-exploration:latest $(python config.py ecr_repo)
docker push $(python config.py ecr_repo)

#Execute main python script
python batch.py
