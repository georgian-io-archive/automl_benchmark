#!/usr/bin/env bash

#Upload swig dependency
curl -o swig-3.0.12.tar.gz -L https://downloads.sourceforge.net/project/swig/swig/swig-3.0.12/swig-3.0.12.tar.gz
aws s3 cp swig-3.0.12.tar.gz s3://$(python -m benchmark.config s3_bucket_root)/$(python -m benchmark.config s3_folder)swig.tar.gz

#Upload SSH files

aws s3 cp --recursive benchmark/config/ssh s3://$(python -m benchmark.config s3_bucket_root)/$(python -m benchmark.config s3_folder)ssh

aws s3 cp $(python -m benchmark.config repo_ssh_key) s3://$(python -m benchmark.config s3_bucket_root)/$(python -m benchmark.config s3_folder)ssh/

aws s3 cp $(python -m benchmark.config ec2_ssh_key) s3://$(python -m benchmark.config s3_bucket_root)/$(python -m benchmark.config s3_folder)ssh/

#Ensure correct file permissions
chmod 755 batch_job.sh

#Build Docker Container
docker build -t auto-ml-exploration .

#Login to aws
eval "$(aws ecr get-login --no-include-email)"

#Push to container repo
docker tag auto-ml-exploration:latest $(python -m benchmark.config ecr_repo)
docker push $(python -m benchmark.config ecr_repo)

