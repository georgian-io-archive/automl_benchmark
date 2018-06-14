#!/usr/bin/env python

import os

import boto3

from benchmark import process
from get_datasets import single_dataset

def execute():

    s3_bucket = os.getenv("S3_BUCKET")
    s3_folder = os.getenv("S3_FOLDER","")
    task = os.getenv("TASK")

    s3 = boto3.resource('s3')

    print(task)
    print(task[1:-1])
    test = task[1:-1].split(",")
    print(test)
    single_dataset(test[2])
    results = process(test[1], test[2], test[3], int(test[4]))

    csv = (','.join(map(str,results))+'\n').encode("utf-8")
    key = (s3_folder+"out/"+"results" + str(runid) +".csv")
    s3.Bucket(s3_bucket).put_object(Key=key, Body = csv)


if __name__ == '__main__':
    execute()
