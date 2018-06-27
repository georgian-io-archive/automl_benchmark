#!/usr/bin/env python

import os
import pickle
import boto3
import time

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
    runid = test_info[0]
    model = test_info[1]
    dataset = test_info[2]
    dtype = test_info[3]
    seed = test_info[4]

    if model == 'tpot':
        import multiprocessing as mp
        # this needs to be here because other libs import mp
        try:
            mp.set_start_method('forkserver')
        except RuntimeError:
            print('Failed to set forkserver')

    #Download dataset
    from ...analysis import single_dataset
    single_dataset(dataset, use_cache=True)

    #Execute benchmark
    from ...analysis import process
    print(model, dataset, dtype, seed)
    results = process(model, dataset, dtype, seed)

    #Upload results to s3
    s3 = boto3.resource('s3')

    csv = (','.join(map(str,results))+'\n').encode("utf-8")
    key = (s3_folder+"out/"+"results" + str(runid) +".csv")
    s3.Bucket(s3_bucket).put_object(Key=key, Body = csv)


if __name__ == '__main__':
    try:
        execute()
    except Exception as e:
        
        s3 = boto3.resource('s3')
        batch_id = int(os.environ["AWS_BATCH_JOB_ARRAY_INDEX"])
        s3_bucket = os.environ["S3_BUCKET"]
        s3_folder = os.getenv("S3_FOLDER","")

        with open("tests.dat", "rb") as f:
            data = pickle.load(f)

        test_info = data[batch_id]
        runid = test_info[0]
        model = test_info[1]
        dataset = test_info[2]
        dtype = test_info[3]
        seed = test_info[4]

        err = str(e).encode("utf-8")
        key = '{}err/{}/{}-{}-{}-{}'.format(s3_folder, model, runid, dataset, dtype, seed)
        s3.Bucket(s3_bucket).put_object(Key=key, Body=err)

        for subdir, dirs, files in os.walk("/tmp"):
            for file in files:
                try:
                    full_path = os.path.join(subdir, file)
                    print(full_path)
                    with open(full_path, 'rb') as data:
                        data.seek(0)
                        key = '{}err/tmp/{}-{}-{}-{}-{}/{}'.format(s3_folder, 
                                                                  model, 
                                                                  runid, 
                                                                  dataset, 
                                                                  dtype, 
                                                                  seed, 
                                                                  full_path[len('/tmp')+1:])
                        print(key)
                        s3.Bucket(s3_bucket).put_object(Key=key, Body=data)
                except Exception as e:
                    print('Error Saving File: ,', str(e))

