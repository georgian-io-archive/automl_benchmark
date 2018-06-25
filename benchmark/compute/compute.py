#!/usr/bin/env python

import boto3
import pickle
import numpy as np
import pandas as pd

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


def generate_smart_reruns(missing_df, missing_leq=5):
    counts = missing_df.groupby('MODEL')['DATASET_ID'].value_counts()
    counts = counts[counts <= missing_leq]
    indicies = counts.index.values.tolist()
    missing_df = missing_df.set_index(['MODEL', 'DATASET_ID'])
    filtered_df = missing_df.loc[indicies].reset_index(level=['MODEL', 'DATASET_ID'])
    ret_list = filtered_df[['ID', 'MODEL', 'DATASET_ID', 'TYPE', 'SEED']].values.tolist() # puts it in order

    return ret_list

def reruns_wrapper():
    runs_df = pd.read_csv('./compiled_results.csv', header=0)
    missing_df = pd.DataFrame(get_failures(), columns=['ID', 'MODEL', 'DATASET_ID', 'TYPE', 'SEED'])
    return generate_smart_reruns(missing_df)


def cleanup():

    with open("running.dat", "rb") as f:
        dat = pickle.load(f)

    for i in dat:
        i.reload()
        i.terminate()

if __name__ == '__main__':
    benchmark(generate_tests)
