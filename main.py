#!/usr/bin/env python

import argparse
import pickle
import sys

import comet_ml

def local_benchmark(exp_id, model, openml_id, problem_type, seed, **kwargs):
    """Use command line args to process single run
    """
    analysis.process(exp_id, model, openml_id, problem_type, seed, **kwargs)

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

def init_env():
    """Init compute environment
    """
    analysis.upload_datasets()
    compute.update_environment()

def refresh_env():
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


def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='Subcommands', dest='command')

    local_parser = subparsers.add_parser('local-benchmark', help='Run benchmarking locally')
    local_parser.add_argument('exp_id', type=str, default='local',
                              help='Experiment ID that is unique to each '
                                   'framework-dataset-seed combination.')
    local_parser.add_argument('model',
                              choices=('auto_sklearn', 'tpot', 'h2o', 'auto_ml', 'foreshadow'),
                              help='AutoML framework to use')
    local_parser.add_argument('openml_id', type=int, help='OpenML dataset ID to use')
    local_parser.add_argument('problem_type', choices=('regression', 'classification'),
                              help='Problem type')
    local_parser.add_argument('seed', type=int, help='Seed for reproducibility')
    local_parser.add_argument('--debug', action='store_true',
                              help='Disables CometML experiment logging and enable logging')
    local_parser.add_argument('-t', '--training-time-mins', type=int, default=5,
                              help='Time allocated for training and optimizing '
                                   'models (mins)')
    local_parser.add_argument('--min-mem', type=str, default='7g',
                              help='Minimum memory allocation for H2O framework')
    local_parser.add_argument('--max-mem', type=str, default='100g',
                              help='Maximum memory allocation for H2O framework')
    local_parser.add_argument('-c', '--n-cores', type=int, default=2,
                              help='Number of cores to allocate for training')
    local_parser.add_argument('--grace-period-secs', type=int, default=300,
                              help='Grace period (secs) before terminating TPOT')
    local_parser.add_argument('-n', '--notes', type=str, default='',
                              help='Optional notes for experiment')

    subparsers.add_parser('fetch-results', help='Download results from S3 to local CSV')
    subparsers.add_parser('download-data', help='Download datasets used')
    subparsers.add_parser('execute-compute', help='Run full benchmark suite on AWS')
    subparsers.add_parser('init-compute', help='Init S3 env for AWS')
    subparsers.add_parser('refresh-compute', help='Lightweight update of S3 environment')
    subparsers.add_parser('resume-compute', help='Resume partially run job on AWS')
    subparsers.add_parser('get-logs', help='Download all logs locally')
    subparsers.add_parser('clean-s3', help='Clean logs and output from S3')
    subparsers.add_parser('run-analysis', help='Execute analysis on locally downloaded results')
    (
        subparsers
        .add_parser('file-compute', help='Run job defined by task file on AWS')
        .add_argument('filename', type=str)
    )
    (
        subparsers
        .add_parser('delete-runs', help='Delete run output using file')
        .add_argument('filename', type=str)
    )
    (
        subparsers
        .add_parser('export-failures', help='Export failures to file')
        .add_argument('filename', type=str)
    )


    commands = {
        'local-benchmark': local_benchmark,
        'fetch-results': download_results,
        'download-data': download_datasets,
        'execute-compute': execute_job,
        'init-compute': init_env,
        'refresh-compute': refresh_env,
        'resume-compute': resume_job,
        'get-logs': download_logs,
        'clean-s3': clean_environment,
        'file-compute': run_file,
        'run-analysis': do_analysis,
        'delete-runs': delete_output,
        'export-failures': export_failures,
    }

    args = parser.parse_args()
    command = commands[args.command]
    del args.command

    return args, command

if __name__ == '__main__':
    args, command = parse_args()

    from benchmark import analysis
    from benchmark import compute

    command(**vars(args))
