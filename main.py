#!/usr/bin/env python

from benchmark import analysis
from benchmark import compute

def local_benchmark():
    """Use command line args to process single run
    """
    analysis.process()


def download_results():
    """Download run data from S3
    """
    analysis.download_data()


def download_datasets():
    """Download datasets
    """
    analysis.get_datasets()

def execute_job():
    """Execute full run
    """
    compute.update_environment()
    compute.run_full()

def update_compute_env():
    """Update compute environment to S3
    """
    compute.update_environment()

def resume_job():
    """Resume running job
    """
    compute.resume()

def sample_run():
    """Use commmand line args to dispatch specific job
    """
    compute.sample_run()

def download_logs():
    """Download logs from S3
    """
    compute.get_logs()


def clean_environment():
   """Clear logs and output from S3 environment
   """
   compute.clean_s3()


if __name__ == '__main__':
    #Run code
