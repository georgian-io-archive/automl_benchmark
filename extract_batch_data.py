#!/usr/bin/env python

import boto3

from config import load_config


def extract():
    s3 = boto3.resource('s3')
    cfg = load_config()
    s3_bucket = cfg["s3_bucket_root"]
    s3_folder = cfg["s3_folder"]
  
    count = 0

    with open("compiled_results.csv", "ab") as f:
        while True:
            try:
                s3.Bucket(s3_bucket).download_fileobj(s3_folder + "out/results" + str(count) + ".csv",  f)
                s3.Bucket(s3_bucket).delete_objects(Delete={'Objects':[{'Key':s3_folder+ "out/results" + str(count) + ".csv"}]})
                count += 1
            except Exception as e:
                break
     

if __name__ == '__main__':
    extract()
