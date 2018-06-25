#!/usr/bin/env bash

#Download logs

mkdir logs

aws s3 cp --recursive s3://$(python -m benchmark.config.config s3_root)/$(python -m benchmark.config.config s3_folder)err/ ./logs




