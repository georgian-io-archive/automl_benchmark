# Benchmarking Automatic Machine Learning Frameworks

## Installation Steps
```bash
$ virtualenv -p python3 automl_benchmark
$ source automl_benchmark/bin/activate
(automl_benchmark) $ pip install -r pre-requirements.txt # Required for auto-sklearn and openml to install correctly
(automl_benchmark) $ pip install -r requirements.txt
```

## Run Get Data
* Go back to the repo dir and get the data  
```bash
(automl_benchmark) $ mkdir datasets
(automl_benchmark) $ ./get_datasets.py
```

Note: For MacOS use TkAgg
https://github.com/pypa/virtualenv/issues/54#issuecomment-223204279