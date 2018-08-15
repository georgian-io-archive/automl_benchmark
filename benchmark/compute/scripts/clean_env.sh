#!/usr/bin/env bash

#Cleanup S3
aws s3 rm --recursive s3://$(python -m benchmark.config s3_bucket_root)/$(python -m benchmark.config s3_folder)

