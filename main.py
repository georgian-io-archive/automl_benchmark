#!/usr/bin/env python

import argparse
import pickle
import sys

from benchmark import analysis
from benchmark import compute

def local_benchmark():
    """Use command line args to process single run
    """
    analysis.process(model, d_id, d_type, seed)


def download_results():
    """Download run data from S3
    """
    analysis.download_data()


def download_datasets():
    """Download datasets
    """
    analysis.get_studies()

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
    compute.sample_run(num, models)

def download_logs():
    """Download logs from S3
    """
    compute.get_logs()


def clean_environment():
   """Clear logs and output from S3 environment
   """
   compute.clean_s3()

def run_file(fname):
    """Execute runs from pickle file
    """
    tasks = pickle.load(open(fname, 'rb'))
    compute.update_environment()
    compute.execute_list(tasks)

def delete_output(fname):
    """Delete specific output from S3 to re-run
    """
    compute.delete_runs(fname)

def export_failures(fname):
    """Export failures to a pickle file
    """
    compute.export_failures(fname)

def do_analysis():
    """Execute analysis on local data
    """
    analysis.analysis_suite()


if __name__ == '__main__':


    commands = {
                 'local': [local_benchmark,
                           [('model',str),('openml_id',int),('type',str),('seed',int)],
                           'Run benchmarking locally'],
                 'fetch-results': [download_results,[],'Download results from S3 to local CSV'],
                 'download-data': [download_datasets,[],'Download datasets used'],
                 'execute-compute': [execute_job,[],'Run full benchmark suite on AWS'],
                 'init-compute': [update_compute_env,[],'Init S3 env for AWS'],
                 'resume-compute': [resume_job,[],'Resume partially run job on AWS'],
                 'get-logs': [download_logs,[],'Download all logs locally'],
                 'clean-s3': [clean_environment,[],'Clean logs and output from S3'],
                 'file-compute': [run_file,[('filename',str)],'Run job defined by task file on AWS'],
                 'run-analysis': [do_analysis,[],'Execute analysis on locally downloaded results'],
                 'delete-runs': [delete_output,[('filename',str)],'Delete run output using file'],
                 'export-failures': [export_failures,[('filename',str)],'Export failures to file']
               }

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    for com, data in commands.items():
        sub = subparsers.add_parser(com, help=data[2])
        for arg in data[1]:
            sub.add_argument(arg[0], type=arg[1])
   
    args = parser.parse_args()
    params = list(vars(args).values())
    commands[sys.argv[1]][0](*params) 
