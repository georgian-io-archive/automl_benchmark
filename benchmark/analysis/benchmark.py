
"""
if __name__ == '__main__':
    # this needs to be here because other libs import mp
    import multiprocessing as mp
    try:
        mp.set_start_method('forkserver')
    except RuntimeError:
        print('Failed to set forkserver')
"""

import os
import time
import json
import signal
from typing import Optional, Dict

import numpy as np
import pandas as pd
import comet_ml
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from tqdm import tqdm

from ..config import load_config


def process_auto_sklearn(X_train, X_test, y_train, df_types, m_type,
                         training_time_mins, n_cores, seed,
                         exp, **kwargs):
    """Function that trains and tests data using auto-sklearn"""

    from autosklearn.classification import AutoSklearnClassifier
    from autosklearn.regression import AutoSklearnRegressor
    from autosklearn.metrics import f1_weighted
    from autosklearn.metrics import mean_squared_error

    categ_cols = df_types[df_types.NAME != 'target']['TYPE'].values.ravel()

    start_time = time.time()
    if m_type == 'classification':
        automl = AutoSklearnClassifier(time_left_for_this_task=training_time_mins * 60,
                                       per_run_time_limit=training_time_mins * 60 // 8,
                                       seed=seed,
                                       resampling_strategy='cv',
                                       resampling_strategy_arguments={'folds': 5},
                                       delete_tmp_folder_after_terminate=False)
    else:
        automl = AutoSklearnRegressor(time_left_for_this_task=training_time_mins * 60,
                                      per_run_time_limit=training_time_mins * 60 // 8,
                                      seed=seed,
                                      resampling_strategy='cv',
                                      resampling_strategy_arguments={'folds': 5},
                                      delete_tmp_folder_after_terminate=False)

    automl.fit(X_train.copy(),
        y_train.copy(),
        feat_type=categ_cols,
        metric=f1_weighted if m_type == 'classification' else mean_squared_error)
    automl.refit(X_train.copy(), y_train.copy())
    end_time = time.time()

    with exp.train():
        metrics_dict = compute_metrics(m_type=m_type,
                                       y_true=y_train,
                                       y_pred=automl.predict_proba(X_train)
                                              if m_type == 'classification'
                                              else automl.predict(X_train))
        exp.log_metrics(metrics_dict)

    exp.log_metrics({
        'processing_time(ms)': 1000 * max(0, (end_time - start_time - training_time_mins * 60)),
    })


    return (automl.predict_proba(X_test) if m_type == 'classification' else
            automl.predict(X_test))


def process_tpot(X_train, X_test, y_train, df_types, m_type,
                 training_time_mins, n_cores, seed,
                 exp, **kwargs):
    """Function that trains and tests data using tpot"""

    from tpot import TPOTClassifier
    from tpot import TPOTRegressor
    from ..config import classifier_config_dict

    # Register Timer
    def handler(signum, frame):
        raise SystemExit('Time limit exceeded, sending system exit...')

    signal.signal(signal.SIGALRM, handler)

    start_time = time.time()
    # default cv is 5
    if m_type == 'classification':
        automl = TPOTClassifier(generations=100,
                                population_size=100,
                                config_dict=classifier_config_dict,
                                verbosity=3,
                                max_time_mins=training_time_mins,
                                scoring='f1_weighted',
                                n_jobs=n_cores,
                                random_state=seed)
    else:
        automl = TPOTRegressor(generations=100,
                               population_size=100,
                               verbosity=3,
                               max_time_mins=training_time_mins,
                               scoring='neg_mean_squared_error',
                               n_jobs=n_cores,
                               random_state=seed)

    # Set timer
    # for long running processes TPOT sometimes does not end even with generations
    signal.alarm(training_time_mins * 60 + kwargs['grace_period_secs'])
    automl.fit(X_train.values, y_train.values)
    end_time = time.time()
    signal.alarm(0)

    with exp.train():
        metrics_dict = compute_metrics(m_type=m_type,
                                       y_true=y_train,
                                       y_pred=automl.predict_proba(X_train)
                                              if m_type == 'classification'
                                              else automl.predict(X_train))
        exp.log_metrics(metrics_dict)

    exp.log_metrics({
        'processing_time(ms)': 1000 * max(0, (end_time - start_time - training_time_mins * 60)),
    })

    return (automl.predict_proba(X_test.values) if m_type == 'classification' else
            automl.predict(X_test.values))


def process_h2o(X_train, X_test, y_train, df_types, m_type,
                training_time_mins, n_cores, seed,
                exp, **kwargs):
    """Function that trains and tests data using h2o's AutoML"""

    import h2o
    from h2o.automl import H2OAutoML

    ip = '127.0.0.1'
    port = np.random.randint(5555,8888)

    start_time = time.time()
    h2o.init(ip=ip, port=port, nthreads=n_cores,
             min_mem_size=kwargs['min_mem'], max_mem_size=kwargs['max_mem'],
             ice_root='/tmp/')

    aml = None

    if(m_type == 'classification'):
        aml = H2OAutoML(max_runtime_secs=training_time_mins * 60,
                        seed=seed, sort_metric='AUTO')
    else:
        aml = H2OAutoML(max_runtime_secs=training_time_mins * 60,
                        seed=seed, sort_metric='MSE')

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
    end_time = time.time()

    with exp.train():
        train_response = aml.predict(h2o.H2OFrame(X_train))
        metrics_dict = compute_metrics(m_type=m_type,
                                       y_true=y_train,
                                       y_pred=train_response[1:].as_data_frame().values
                                              if m_type == 'classification'
                                              else train_response.as_data_frame().values.ravel())
        exp.log_metrics(metrics_dict)

    exp.log_metrics({
        'processing_time(ms)': 1000 * max(0, (end_time - start_time - training_time_mins * 60)),
    })

    response = aml.predict(td)
    pred = (response[1:].as_data_frame().values if m_type == 'classification' else
           response.as_data_frame().values.ravel())

    h2o.cluster().shutdown()

    return pred


def process_auto_ml(X_train, X_test, y_train, df_types, m_type,
                    training_time_mins, n_cores, seed,
                    exp, **kwargs):
    """Function that trains and tests data using auto_ml"""

    from auto_ml import Predictor

    start_time = time.time()
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

    end_processing_time = time.time()
    automl.train(X_train, model_names=cmodels if m_type == 'classification' else rmodels,
        scoring='f1_score' if m_type == 'classification' else 'mean_squared_error',
        cv=5, verbose=False)
    end_training_time = time.time()

    with exp.train():
        metrics_dict = compute_metrics(m_type=m_type,
                                       y_true=y_train,
                                       y_pred=automl.predict_proba(X_train)
                                              if m_type == 'classification'
                                              else automl.predict(X_train))
        exp.log_metrics(metrics_dict)

    exp.log_metrics({
        'processing_time(ms)': 1000 * (end_processing_time - start_time),
        'automl_training_time(mins)': (end_training_time - end_processing_time) / 60,
    })

    return (automl.predict_proba(X_test) if m_type == 'classification' else
            automl.predict(X_test))


def process_foreshadow(X_train, X_test, y_train, df_types, m_type,
                       training_time_mins, n_cores, seed,
                       exp, **kwargs):
    """Function that trains and tests data using foreshadow"""

    import foreshadow as fs

    start_time = time.time()
    estimator = fs.estimators.AutoEstimator(problem_type=m_type, auto='tpot')
    estimator.estimator_kwargs.update({
        'max_time_mins': training_time_mins,
        'n_jobs': n_cores,
        'random_state': seed,
    })

    shadow = fs.Foreshadow(problem_type=m_type, estimator=estimator)

    shadow.fit(X_train, y_train)
    end_time = time.time()

    with exp.train():
        metrics_dict = compute_metrics(m_type=m_type,
                                       y_true=y_train,
                                       y_pred=shadow.predict_proba(X_train)
                                              if m_type == 'classification'
                                              else shadow.predict(X_train))
        exp.log_metrics(metrics_dict)

    # Log experiment info
    n_trials = len(shadow.estimator.estimator.estimator.evaluated_individuals_)

    exp.log_metrics({
        'num_trials': n_trials,
        'processing_time(ms)': 1000 * max(0, (end_time - start_time - training_time_mins * 60)),
        'avg_trial_runtime(ms)': (training_time_mins * 60 * 1000) / n_trials / n_cores,
    })
    exp.log_asset_data(json.dumps(shadow.serialize('dict'), indent=4),
                       file_name='foreshadow_config.json')

    return (shadow.predict_proba(X_test)
            if m_type == 'classification'
            else shadow.predict(X_test))


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


def process(exp_id, m_name, d_id, m_type, seed, **kwargs):
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

    model_dict = {
        'auto_sklearn': process_auto_sklearn,
        'tpot': process_tpot,
        'h2o': process_h2o,
        'auto_ml': process_auto_ml,
        'foreshadow': process_foreshadow
    }

    if len(kwargs) == 0:
        kwargs = load_config()

    # Load data
    X_train, X_test, y_train, y_test, df_types = parse_open_ml(d_id, seed)

    # Initialize comet_ml experiment
    exp = initialise_comet(exp_id, m_name, d_id,
                           X_train, X_test, y_train, y_test,
                           m_type, seed, **kwargs)

    # Perform AutoML
    print(f'Benchmarking for {m_name} | D{d_id} | S{seed} started.')
    y_hat = model_dict.get(m_name, error)(X_train=X_train, X_test=X_test, y_train=y_train,
                                          df_types=df_types, m_type=m_type,
                                          seed=seed, exp=exp, **kwargs)
    print(f'Benchmarking for {m_name} | D{d_id} | S{seed} ended.')

    # Calculate scores
    with exp.test():
        metrics_dict = compute_metrics(m_type=m_type,
                                       y_true=y_test,
                                       y_pred=y_hat,
                                       y_ref=y_train)
        exp.log_metrics(metrics_dict)
    exp.log_parameter('experiment_end_time', time.time())

    return (m_name, d_id, m_type, seed,
            *metrics_dict.values()) # (mse, r2, log_loss, F1)


def compute_metrics(m_type: str,
                    y_true: pd.Series,
                    y_pred: pd.Series,
                    y_ref: Optional[pd.Series] = None
                   ) -> Dict[str, int]:
    """Compute benchmarking metrics.

    Metrics calculated include MSE and R2 for regression, and
    log_loss and F1 for classification.

    Arguments:
        m_type {str} -- Problem type. Valid values are
                        {'classification', 'regression'}.
        y_true {pd.Series} -- True label values.
        y_pred {pd.Series} -- Predicted label values.

    Keyword Arguments:
        y_ref {Optional[pd.Series]}
            -- To be provided if y_true corresponds to test labels for
               non-binary classification. Used to train the OneHotEncoder.
               (default: {None})

    Returns:
        Dict[str, int] -- A dictionary for metrics. Default values are np.nan.
    """

    metrics_dict = {
        'MSE': np.nan,
        'R2': np.nan,
        'log_loss': np.nan,
        'F1': np.nan,
    }

    if m_type == 'classification':
        if y_true.unique().size == 2:
            ll_y = y_true
        else:
            if y_ref is not None:
                ll_y = (
                    OneHotEncoder()
                    .fit(y_ref.to_frame())
                    .transform(y_true.to_frame())
                )
            else:
                ll_y = OneHotEncoder().fit_transform(y_true.to_frame())

        metrics_dict['log_loss'] = metrics.log_loss(ll_y, y_pred)
        metrics_dict['F1'] = metrics.f1_score(y_true, y_pred.argmax(axis=1), average='weighted')
    else:
        metrics_dict['R2'] = metrics.r2_score(y_true, y_pred)
        metrics_dict['MSE'] = metrics.mean_squared_error(y_true, y_pred)

    return metrics_dict


def save_results(m_name, d_id, m_type, seed, mse, r2_score, log_loss, f1_score):
    """Saves the results to a local file"""

    with open('compiled_results.csv', 'a') as fopen:
        fopen.write('0,{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(m_name, d_id, m_type, seed, mse,
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
    models = ['auto_sklearn', 'tpot', 'h2o', 'auto_ml', 'foreshadow']

    tests = [ [i,j[0],j[1],k] for i in models for j in datasets for k in seeds]
    tests = [ [i] + d for i, d in enumerate(tests) ]
    return tests


def benchmark():
    """Main function to benchmark each function"""

    with open('compiled_results.csv', 'w') as fopen:
        fopen.write('ID,MODEL,DATASET_ID,TYPE,SEED,MSE,R2_SCORE,LOGLOSS,F1_SCORE\n')

    test = generate_tests()

    for i, m, d_id, t, s in test:
        rslts = process(i, m, d_id, t, s)
        save_results(*rslts)


def initialise_comet(exp_id, m_name, d_id,
                     X_train, X_test, y_train, y_test,
                     m_type, seed, debug=False, **kwargs,
                    ) -> comet_ml.Experiment:
    # HACK to filter clean parameters for saving
    extra_params = {f'benchmark/{k}': v for k, v in kwargs.items() if k not in ('command', 'debug')}
    extra_params['benchmark/allocated_training_time(mins)'] = extra_params.pop('benchmark/training_time_mins')

    exp = comet_ml.Experiment(
        project_name='automl-framework-benchmarks',
        workspace='gp-internal',
        auto_param_logging=False,
        auto_output_logging=None,
        disabled=debug
    )
    exp.set_name(f'E{exp_id} | D{d_id} | S{seed}')

    # Add model framework and problem type as tags
    exp.add_tags((m_name, m_type))

    exp.log_parameters({
        'run/id': exp_id,
        'dataset/id': d_id,
        'dataset/num_rows': len(X_train) + len(X_test),
        'dataset/num_features': X_train.shape[1],
        'dataset/memory(kB)': sum(df.memory_usage(deep=True, index=False).sum()
                                  for df in (X_train, X_test, y_train.to_frame(), y_test.to_frame())) / 1e3,
        'dataset/size': sum(x.size for x in (X_train, X_test, y_train, y_test)),
        'dataset/problem_type': m_type,
        'seed': seed,
        'experiment_start_time': time.time(),
        'framework': m_name,
        **extra_params
    })

    return exp