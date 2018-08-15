# AutoML Framework Benchmarking System

## About

This library was created to perform rigorous benchmarking of machine learning libraries 
across a wide variety of datasets from OpenML. This framework can easily be extended to test
other automl packages, to test on different datasets, to test on other hardware platforms, and to
expand the metrics and rigor of testing.

## Installation

Simply installs in a Python 3 environment. (We recommend virtualenv)

```bash
$ virtualenv -p python3 automl_benchmark
$ source automl_benchmark/bin/activate
(automl_benchmark) $ pip install -r pre-requirements.txt # Required for auto-sklearn and openml to install correctly
(automl_benchmark) $ pip install -r requirements.txt
```

**Note**: For MacOS you must set [TkAgg](https://github.com/pypa/virtualenv/issues/54#issuecomment-223204279) 
as the matplotlib backend in order to use the plotting functions


## Usage

A `main.py` file has been provided to serve as an interface to the benchmarking module which contains
all the code neccesary to execute. The library can also be used by doing `from benchmark import analysis` and
`from benchmark import compute` in REPL or a custom script.

### Arguments

The main file is executed in the following manner. `python main.py [command] [arguments]`

| Command | Arguments | Description |
| ------- | --------- | ----------- |
| local | `model` `openml_id` `type` `seed` | Executes a single benchmarking task on the local machine
| | | **model** is the name of the model defined in benchmark.py
| | | **openml_id** is the dataset id of an OpenML dataset to benchmark on
| | | **type** is either *classification* or *regression* depending on the problem type
| | | **seed** is the numpy / python random seed to use for consistency
| download-data | none | Downloads the datasets for the studies defined in the configuration file
| init-compute| none | Initializes AWS tooling (uploads datasets, dependencies, and RSA keys defined in config and also pushes docker container to ECR)
| refresh-compute | none | Performs same actions as init-compute except does not upload datasets
| execute-compute | none | Executes benchmarking from scratch on AWS
| resume-compute | none | Executes benchmarking for any tasks which do not have results on S3
| file-compute | `file_path` | Executes benchmarking for tasks from pickle file of task list (See `benchmarking.py`)
| | | **file_path** Relative or absolute path to read task pickle file
| export-failures | `file_path` | Exports benchmarking tasks that do not have results in S3 to pickle file locally
| | | **file_path** Relative or absolute path to write task pickle file
| fetch-results | none | Downloads run results from S3 to file `compiled_results.csv`
| delete-runs | `file_path` | Deletes run results from S3 that exist in pickled task file
| | | **file_path** Relative or absolute path to read task pickle file
| get-logs | none | Downloads log directory from S3 to local `./log/` folder 
| clean-s3 | none | **CAUTION** Deletes ALL files from S3 including logs and results
| run-analysis | none | Executes analysis of local `compiled_results.csv` file, exports charts to `./figures/` and reports statistics

## Configuration

Before executing the benchmarking on AWS you must ensure that your configuration file is up to date and correct.
The configuration is a json file named `batch.config` in the root directory of the repo. Before modifying the
config be sure to copy `batch.config.default` to `batch.config` 

| Scope | Key | Default | Description |
| ----- | --- | ------- | ----------- |
| **GLOBAL** | studies | `[[130, "regression"], [135, "classification"]]`  | This is a list of lists of which the first item in the inner list is a OpenML study ID and the second item is a string tag. By default we use our collected studies.
|  | cores | 2 | Number of cores to allocate to allocate to the AutoML framework if supported. Number of cores allocated to batch. *Does not affect baremetal machines*
|  | memory | 3500 | Memory in MB allocated to AWS Batch machines. Also minimum memory allocated to AutoML frameworks if supported (JVM for H2O)
|  | runtime | 10800 | Soft runtime limit for AutoML frameworks. (seconds)
|  | hard_limit | 12600 | Hard runtime limit before killing process (seconds)
|  | s3_bucket_root | None | Bucket id of S3 bucket to store persistent data
|  | s3_folder | `""` | Folder within bucket to store data (excludes leading slash and includes trailing slash i.e. `"data/"`)
|  | openml_api_key | None | API key to get OpenML studies and download data
|  | repo_ssh_key | None | File path to RSA key with authentication to git repo
|  | ec2_ssh_key | None | File path to RSA key associated with AWS Batch cluster and EC2 Instance Template
| **BATCH** | job_queue_id | None | Id of Job Queue in AWS Batch (Must be associated with compute cluster)
|  | job_definition_id | None | Id of Job Definition to template job
|  | job_name | None | Name of job to create from job definition (arbitrary)
|  | ecr_repo | None | URI of ECR repo to upload docker image to i.e. `000.dkr.ecr.us-east-1.amazonaws.com/name`
| **BAREMETAL** | cluster_image | `ami-14c5486b` | AMI ID of EC2 instances to spin up (default is latest AmazonLinux)
|  | ec2_template | None | Template ID of launch template defining properties of cluster spot instances
|  | cluster_size | 100 | Max cluster size of bare-metal spot fleet
|  | cluster_type | c4.xlarge | Defines instance type of spot fleet (sets hard mem / cpu limits for baremetal)


## Executing

The following script (assuming your configuration file is correct and RSA keys are valid) will generate tests, download the data,
and start an AWS Run.

```bash

./main.py download_data # Downloads datasets
./main.py init-compute # Uploads all neccesary files to S3
./main.py execute-compute # Will execute all compute tasks
./main.py fetch-results # Downloads results
./main.py perform-analysis # Compiles results and exports figures

```

*See Usage for more details commands*

## Extending

#### Adding Models / Frameworks

Additional models and frameworks can be added to `benchmark/analysis/benchmark.py` and all other systems will automatically update
Ensure that `generate_tests()` is generating tests for your new framework.

#### Adding Compute Methods

Included are two compute methods. AWS Batch and AWS Baremetal. To add a new compute method we reccomend following
the existing folder structure and adding a folder to `benchmark/compute` which includes a `scripts` folder if neccesary.
Within this folder create a `method.py` file and a `method_wrapper.py` file. The method file should include a
class that inherits from `BatchDispatcher` like so.

```python

@AutoMLMethods('framework')
class CustomDispatcher(Dispatcher):

    @classmethod
    def process(cls, tests):
        pass

```

Notice the AutoMLMethods decorator that is passed parameters of the framework names to be executed via that method.
Also notice `process` which must be overriden. This method is called to preocess the list of tests passed in via the `tests` parameter.

The `method_wrapper.py` file should contain a method that is executed when the file is called. The `method.py` file should
provision a system which eventually executes this file. Within this file the corresponding tests should be executed
and uploaded to S3 in addition to logs on failure. Please review baremetal and batch for examples.

## Contributing

This project was developed to solve the problem of large-scale benchmarking of python machine learning frameworks. We are no longer maintaining this repository but we hope it may be useful in the future. Feel free to create pull requests or issues and we will attempt to address them. Thanks! 