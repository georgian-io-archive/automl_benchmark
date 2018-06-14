#!/usr/bin/env python

import os
import shutil

import boto3

from config import load_config
from benchmark import process
from get_datasets import single_dataset

def execute():

    config = load_config()
    s3_bucket = config["s3_bucket"]
    s3_folder = config["s3_folder"]
    task = os.getenv("TASK")

    single_dataset(dataset)
    s3 = boto3.resource('s3')

    test = task[1:-1].split(',')
    results = process(test[1], test[2], test[3], test[4])

    csv = (','.join(map(str,results))+'\n').encode("utf-8")
    key = (s3_folder+"out/"+"results" + str(runid) +".csv")
    s3.Bucket(s3_bucket).put_object(Key=key, Body = csv)

    shutil.rmtree("/tmp") 

if __name__ == '__main__':
    try:
        try:
            mp.set_start_method('forkserver')
        except RuntimeError:
            pass
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
        key = s3_folder + "err/" + model + "/" + str(runid) + "-"  + dataset + "-" + dtype + "-" + str(seed)
        s3.Bucket(s3_bucket).put_object(Key=key, Body=err)

        for subdir, dirs, files in os.walk("/tmp"):
            for file in files:
                full_path = os.path.join(subdir, file)
                with open(full_path, 'rb') as data:
                    s3.Bucket(s3_bucket).put_object(Key=s3_folder + "err/tmp/" + model + str(runid) + "-" + dataset + "-" + dtype + "-" + str(seed) + "/" + full_path[len("/tmp")+1:], Body=data)
          
        shutil.rmtree("/tmp") 
