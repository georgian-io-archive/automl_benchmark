# A Benchmark of Automatic Machine Learning Frameworks
Replicating the results

## Installation Steps
1. Create a virtualenv and install required libraries
  * Note: do this in the repo folder  
```bash
$ virtualenv -p python3 automl_benchmark
$ source automl_benchmark/bin/activate
(automl_benchmark) $ pip install -r requirements.txt
```
* TODO: Check if sklearn has fixed multi-class support and update requirements
2. Clone down openml and follow steps to setup library
  * Recomendation: create a library repos folder in your home directory  
```bash
(automl_benchmark) $ cd
(automl_benchmark) $ mkdir library_repos; cd library_repos
(automl_benchmark) $ git clone https://github.com/openml/openml-python.git
(automl_benchmark) $ cd openml; pip install -e .
```

## Run Get Data
* Go back to the repo dir and get the data  
```bash
(automl_benchmark) $ mkdir datasets
(automl_benchmark) $ ./get_datasets.py
```

## Run the benchmarking
* TODO