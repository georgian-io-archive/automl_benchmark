#!/usr/bin/env bash

#Upload SSH files
aws s3 cp --recursive ssh s3://georgianpartners-auto-ml-data/ssh

#Ensure correct file permissions
chmod 755 dispatch.sh

#Build Docker Container
docker build -t auto-ml-exploration .

#Login to aws
eval "$(aws ecr get-login --region us-east-1 --no-include-email)"

#Push to container repo
docker tag auto-ml-exploration:latest 823217009914.dkr.ecr.us-east-1.amazonaws.com/auto-ml-exploration:latest
docker push 823217009914.dkr.ecr.us-east-1.amazonaws.com/auto-ml-exploration:latest
