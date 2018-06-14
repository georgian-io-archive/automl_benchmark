#!/usr/bin/env python

if __name__ == '__main__':
    # this needs to be here because other libs import mp
    import multiprocessing as mp
    try:
        mp.set_start_method('forkserver')
    except RuntimeError:
        print('Failed to set forkserver')
import signal

import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from tqdm import tqdm

TIME_PER_TASK = 10800 # seconds (3 hours)
GRACE_PERIOD = 300
MIN_MEM = '6g'
MAX_MEM = '6g'
N_CORES = 2

def process_auto_sklearn(X_train, X_test, y_train, df_types, m_type, seed, *args):
    """Function that trains and tests data using auto-sklearn"""

    

    from autosklearn.classification import AutoSklearnClassifier
    from autosklearn.regression import AutoSklearnRegressor

    categ_cols = df_types[df_types.NAME != 'target']['TYPE'].values.ravel()

    if m_type == 'classification':
        automl = AutoSklearnClassifier(time_left_for_this_task=TIME_PER_TASK,
                                       per_run_time_limit=int(TIME_PER_TASK/8),
                                       seed=seed,
                                       resampling_strategy='cv',
                                       resampling_strategy_arguments={'folds': 5},
                                       delete_tmp_folder_after_terminate=False)
    else:
        automl = AutoSklearnRegressor(time_left_for_this_task=TIME_PER_TASK,
                                      per_run_time_limit=int(TIME_PER_TASK/8),
                                      seed=seed,
                                      resampling_strategy='cv',
                                      resampling_strategy_arguments={'folds': 5},
                                      delete_tmp_folder_after_terminate=False)
    
    automl.fit(X_train.copy(), y_train.copy(), feat_type=categ_cols)
    automl.refit(X_train.copy(), y_train.copy())

    pdb.set_trace()

    return (automl.predict_proba(X_test) if m_type == 'classification' else 
            automl.predict(X_test))

def process_tpot(X_train, X_test, y_train, df_types, m_type, seed, *args):
    """Function that trains and tests data using tpot"""

    from tpot import TPOTClassifier
    from tpot import TPOTRegressor
    from tpot_config import classifier_config_dict

    # Register Timer
    def handler(signum, frame):
        raise SystemExit('Time limit exceeded, sending system exit...')

    signal.signal(signal.SIGALRM, handler)

    # default cv is 5
    if m_type == 'classification':
        automl = TPOTClassifier(generations=100,
                                population_size=100,
                                config_dict=classifier_config_dict,
                                verbosity=3,
                                max_time_mins=int(TIME_PER_TASK/60),
                                scoring='f1_weighted',
                                n_jobs=N_CORES,
                                random_state=seed)
    else:
        automl = TPOTRegressor(generations=100, 
                               population_size=100,
                               verbosity=3,
                               max_time_mins=int(TIME_PER_TASK/60),
                               n_jobs=N_CORES,
                               random_state=seed)

    # Set timer
    # for long running processes TPOT sometimes does not end even with generations
    signal.alarm(TIME_PER_TASK+GRACE_PERIOD)
    automl.fit(X_train.values, y_train.values)

    return (automl.predict_proba(X_test.values) if m_type == 'classification' else 
            automl.predict(X_test.values))

def process_h2o(X_train, X_test, y_train, df_types, m_type, seed,*args):
    """Function that trains and tests data using h2o's AutoML"""

    import h2o
    from h2o.automl import H2OAutoML

    ip = args[0] if len(args) > 0 else '127.0.0.1'
    port = np.random.randint(5555,8888)

    h2o.init(ip=ip, port=port, min_mem_size=MIN_MEM, max_mem_size=MAX_MEM, nthreads=N_CORES, ice_root='/tmp/')
    aml = H2OAutoML(max_runtime_secs=TIME_PER_TASK, seed=seed)
    dd = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
    td = h2o.H2OFrame(X_test)

    # set categorical columns as 'factors'
    categ_cols = df_types[df_types['TYPE'] == 'categorical']['NAME'].values.tolist()
    if len(categ_cols) > 0:
        dd[categ_cols] = dd[categ_cols].asfactor()
        td[categ_cols] = td[categ_cols].asfactor()
    if m_type == 'classification':
        dd['target'] = dd['target'].asfactor()

    aml.train(y = 'target', training_frame = dd)
    response = aml.predict(td)
    return (response[1:].as_data_frame().values if m_type == 'classification' else 
            response.as_data_frame().values.ravel())

def process_auto_ml(X_train, X_test, y_train, df_types, m_type, seed, *args):
    """Function that trains and tests data using auto_ml"""

    from auto_ml import Predictor

    # convert column names to numbers to avoid column name collisions (bug)
    names = {c: str(i) for i, c in enumerate(X_train.columns)}
    X_train.columns = names
    X_test.columns = names

    df_types.loc[df_types['NAME'] == 'target', 'TYPE'] = 'output'
    df_types = df_types[df_types['TYPE'] != 'numerical'].set_index('NAME')
    df_types = df_types.rename(index=names)['TYPE'].to_dict()
    X_train['target'] = y_train

    cmodels = ['AdaBoostClassifier', 'ExtraTreesClassifier', 'RandomForestClassifier', 'XGBClassifier']
    rmodels = ['BayesianRidge', 'ElasticNet', 'Lasso', 'LassoLars', 'LinearRegression', 
        'Perceptron', 'LogisticRegression', 'AdaBoostRegressor', 'ExtraTreesRegressor', 
        'PassiveAggressiveRegressor', 'RandomForestRegressor', 'SGDRegressor', 'XGBRegressor']
    
    automl = Predictor(type_of_estimator='classifier' if m_type == 'classification' else 'regressor',
                             column_descriptions=df_types)

    automl.train(X_train, model_names=cmodels if m_type == 'classification' else rmodels,
        cv=5, verbose=False)

    return (automl.predict_proba(X_test) if m_type == 'classification' else 
            automl.predict(X_test))

def parse_open_ml(d_id, seed):
    """Function that processes each dataset into an interpretable form
    Args:
        d_id (int): dataset id
        seed (int): random seed for replicable results
    Returns:
        A tuple of the train / test split data along with the column types
    """

    df = pd.read_csv('./datasets/{0}.csv'.format(d_id))
    df_types = pd.read_csv('./datasets/{0}_types.csv'.format(d_id))

    df_valid = df[~df['target'].isnull()]

    x_cols = [c for c in df_valid.columns if c != 'target']
    X = df_valid[x_cols]
    y = df_valid['target']

    X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, test_size = 0.3, random_state=seed)

    return X_train, X_test, y_train, y_test, df_types


def process(m_name, d_id, m_type, seed, *args):
    """Routing function to call and process the results of each ml model
    Args:
        m_name (str): name of the automl model
        d_id (int): the data set id
        m_type (str): 'regression' or 'classification' representing the type of automl probelm
        seed (int): random seed for replicable results
    """
    def error():
        raise Exception('Invalid model framework!')

    np.random.seed(seed) # set numpy random seed

    model_dict = {'auto-sklearn': process_auto_sklearn,
                  'tpot': process_tpot,
                  'h2o': process_h2o,
                  'auto_ml': process_auto_ml}
    X_train, X_test, y_train, y_test, df_types = parse_open_ml(d_id, seed)
    y_hat = model_dict.get(m_name, error)(X_train, X_test, y_train, df_types, m_type, seed, *args)
    rmse, r2_score = (np.nan, np.nan)
    log_loss, f1_score = (np.nan, np.nan)
    if m_type == 'classification':
        ll_y = (y_test if np.unique(y_test).size == 2 else 
                OneHotEncoder().fit(y_train.values.reshape((-1, 1))).transform(y_test.values.reshape((-1,1))))
        log_loss = metrics.log_loss(ll_y, y_hat)
        f1_score = metrics.f1_score(y_test, y_hat.argmax(axis=1), average='weighted')
    else:
        r2_score = metrics.r2_score(y_test, y_hat)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_hat))

    return (m_name, d_id, m_type, seed, rmse, r2_score, log_loss, f1_score)

def save_results(m_name, d_id, m_type, seed, rmse, r2_score, log_loss, f1_score):
    """Saves the results to a local file"""

    with open('compiled_results.csv', 'a') as fopen:
        fopen.write('0,{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(m_name, d_id, m_type, seed, rmse, 
            r2_score, log_loss, f1_score))

def load_datasets():
    cinfo_df = pd.read_csv('./datasets/study_classification_info.csv')
    rinfo_df = pd.read_csv('./datasets/study_regression_info.csv')

    datasets = np.vstack((np.array([[d_id, 'classification'] for d_id in cinfo_df['DATASET_ID'].values]), 
                          np.array([[d_id, 'regression'] for d_id in rinfo_df['DATASET_ID'].values])))

    return datasets

def generate_tests():
    """Generates test data for benchmarking"""

    np.random.seed(1400)
    seeds = list(map(int, list(np.random.randint(1000, size=10)))) # Generate 10 random 'seeds'
    datasets = load_datasets().tolist()
    models = ['auto-sklearn', 'tpot', 'h2o', 'auto_ml']

    tests = [ [i,j[0],j[1],k] for i in models for j in datasets for k in seeds] 
    tests = [ [i] + d for i, d in enumerate(tests) ]
    return tests

def benchmark():
    """Main function to benchmark each function"""

    with open('compiled_results.csv', 'w') as fopen:
        fopen.write('ID,MODEL,DATASET_ID,TYPE,SEED,RMSE,R2_SCORE,LOGLOSS,F1_SCORE\n')

    test = generate_tests()

    for i, m, d_id, t, s in test:
        rslts = process(m, d_id, t, s)
        save_results(*rslts)

if __name__ == '__main__':
    benchmark() # run benchmarking locally
