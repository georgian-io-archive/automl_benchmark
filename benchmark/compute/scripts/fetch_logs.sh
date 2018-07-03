#!/usr/bin/env bash

#Download logs

mkdir logs

aws s3 cp --recursive s3://$(python -m benchmark.config s3_bucket_root)/$(python -m benchmark.config s3_folder)logs/ ./logs




