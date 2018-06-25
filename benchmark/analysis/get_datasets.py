import argparse
import os

import boto3
import openml
import numpy as np
import scipy
import pandas as pd
from tqdm import tqdm

from ..config import load_config

def _make_data_dir():
    data_dir = './datasets'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

def _save_dataset_data(d_id):
    """Takes a OpenMLDataset object and saves the dataset associated with it
    Args:
        d_id (int): Dataset id
    Returns:
        A tuple of dataset features (dataset id, 
                                     dataset name, 
                                     number of rows, 
                                     number of features, 
                                     number of classes)
    """
    d = openml.datasets.get_dataset(d_id)
    X, y, col_types, col_names = d.get_data(target=d.default_target_attribute,
                                            return_categorical_indicator=True,
                                            return_attribute_names=True)
    if scipy.sparse.issparse(X):
        X = X.todense() # convert sparse matrix to dense

    # Build data df
    df_dict = {n: X[:, i] for i, n in enumerate(col_names)}
    df_dict['target'] = y
    df = pd.DataFrame(df_dict)

    # Build categories df
    types_df = pd.DataFrame([(n, 'categorical' if t else 'numerical') for n, t in zip(col_names, 
                                                                                      col_types)],
                             columns=['NAME', 'TYPE'])
    types_df.loc[len(types_df)] = ['target', 'target']

    # Save dfs
    df.to_csv('./datasets/{0}.csv'.format(d.dataset_id), index=False)
    types_df.to_csv('./datasets/{0}_types.csv'.format(d.dataset_id), index=False)

    classes = np.unique(df['target'].values.ravel()).size
    return (d_id, d.name, df.shape[0], df.shape[1], classes)


def _get_study(s_id, s_name):
    """Uses the standard OpenML 100 as a testing suite to test different AutoML algorithms
    Args:
        s_id (int): number that represents the openml study id
        s_name (str): a string used to identify the study
    """

    benchmark_study = openml.study.get_study(s_id)
    dataset_info = []

    print('Getting data for study: {}'.format(s_name))
    for d_id in tqdm(benchmark_study.data):
        tqdm.write('Getting dataset ({})'.format(d_id))
        dataset_info.append(_save_dataset_data(d_id))

    study_df_cols = ['DATASET_ID', 'DATASET_NAME', 'ROWS', 'FEATURES', 'CLASSES']
    study_df = pd.DataFrame(dataset_info, columns=study_df_cols)
    study_df.to_csv('./datasets/study_{}_info.csv'.format(s_name), index=False)


def _get_from_s3(d_id):
    """Downloads dataset from robust S3 bucket cache"""
    s3_bucket = os.getenv("S3_BUCKET","")
    s3_folder = os.getenv("S3_FOLDER", "")

    s3 = boto3.resource('s3')
    with open("datasets/" + str(d_id) + ".csv", "wb") as f:
        s3.Bucket(s3_bucket).download_fileobj(s3_folder + "datasets/" + str(d_id) + ".csv",  f)
    with open("datasets/" + str(d_id) + "_types.csv", "wb") as f:
        s3.Bucket(s3_bucket).download_fileobj(s3_folder + "datasets/" + str(d_id) + "_types.csv",  f)

def upload_datasets():
    """Uploads datasets to S3 cache"""
    config = load_config()
    s3_bucket = config["s3_bucket_root"]
    s3_folder = config["s3_folder"]

    s3 = boto3.resource('s3')

    for subdir, dirs, files in os.walk("datasets"):
        for file in tqdm(files):
            try:
                with open("datasets/" + file, 'rb') as data:
                    data.seek(0)
                    key = s3_folder + 'datasets/' + file
                    s3.Bucket(s3_bucket).put_object(Key=key, Body=data)
            except Exception as e:
                print('Error Saving File: ,', str(e))

def get_studies():
    """Collects the data for a specified set of openml 'studies'"""
    studies = [[130, 'regression'],
               [14, 'classification']]

    _make_data_dir()

    for s in studies:
        _get_study(*s)

def single_dataset(d_id, use_cache=False):
    """Downloads a single dataset"""
    _make_data_dir()

    if not use_cache:
        _save_dataset_data(d_id)
    else:
        _get_from_s3(d_id) 
