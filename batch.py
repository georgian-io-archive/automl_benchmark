#!/usr/bin/env python

import boto3
import pickle
import numpy as np

from config import load_config
from benchmark import generate_tests

from dispatcher import execute_methods
from baremetal import BareDispatch
from aws_batch import AWSBatchDispatch

def benchmark(get_tests):
    """Splits data between benchmarking implementations"""
    execute_methods(get_tests)

def partial(nums, models):
    
    def get_partial():
        fail = get_failures()
        runs = []
        for j, v in enumerate(nums):
            runs += [x for x in fail if x[1] == models[j]][:v]
        return runs

    benchmark(get_partial)

def resume():
    """Resumes process and restarts failed tasks"""
    benchmark(get_failures)
    

def get_failures():
    """Retrieve Failures from S3"""

    config = load_config()
    s3_bucket = config["s3_bucket_root"]
    s3_folder = config["s3_folder"]

    s3 = boto3.resource('s3')
    client = boto3.client('s3')
    batch = boto3.client('batch')

    data = generate_tests()

    """
    job_index = []
    jobs = batch.list_jobs(arrayJobId=job_id)
    print(jobs)
    job_index += jobs['jobSummaryList']
    while 'nextToken' in jobs.keys():
         jobs = client.list_jobs(arrayJobId=job_id, nextToken=jobs['nextToken'])
         job_index += jobs['jobSummaryList']

    idx = { str(job["arrayProperties"]["index"]):job["jobId"] for job in job_index }
    data = [d + [ idx[str(i)] ] for i, d in enumerate(data)]
    """

    paginator = client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_folder + "out/", 
                                 PaginationConfig={"MaxItems":len(data), "PageSize":len(data)})

    prefix_length = len(s3_folder + "out/")

    keys = []

    for p in pages: 
        #Extracts indicies from filenames
        keys += [ int(k["Key"][7 + prefix_length:-4]) for k in p["Contents"] ]
    
    failures =  [ run for run in data if run[0] not in keys ]

    return failures


if __name__ == '__main__':
    benchmark(generate_tests)
