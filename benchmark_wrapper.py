#!/usr/bin/env python

import os
import pickle
import boto3
import time

from benchmark import process
from get_datasets import single_dataset

def check_file(s3, s3_bucket,s3_folder,key):
    """Checks if a file exists in a s3 folder"""

    try:
        s3.Object(s3_bucket, s3_folder + key).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise e

    return True

def execute():
    """Executes benchmarking from AWS Batch"""

    #Load environment variables
    batch_id = int(os.environ["AWS_BATCH_JOB_ARRAY_INDEX"])
    s3_bucket = os.environ["S3_BUCKET"]
    s3_folder = os.getenv("S3_FOLDER","")

    #Load metadata
    with open("tests.dat", "rb") as f:
        data = pickle.load(f)

    test_info = data[batch_id]
    model = test_info[0]
    dataset = test_info[1]
    dtype = test_info[2]
    seed = test_info[3]

    #Download dataset
    single_dataset(dataset)

    #Execute benchmark
    results = process(model, dataset, dtype, seed)

    #Upload results to s3
    s3 = boto3.resource('s3')

    #Wait until lock is deleted
    while(file_exists(s3, s3_bucket, s3_folder, "LOCK")):
       time.sleep(1)

    #Write lock to s3
    with open("LOCK", 'a') as f: 
        s3.Bucket(s3_bucket).put_object(Key=s3_folder+"LOCK", Body = f)

    #Download append and upload
    s3.Bucket(s3_bucket).download_file(s3_folder+"results.csv", "results.csv")
    with open("results.csv", "a") as f:
        csv = ','.join(map(str,results))
        f.write(csv + '\n')
    with open("results.csv", "r") as f:
        s3.Bucket(s3_bucket).put_object(Key=s3_folder+"results.csv", Body = f)


    #Delete lock
    s3.Object(s3_bucket, s3_folder + "LOCK").delete()


if __name__ == '__main__':
    execute()
