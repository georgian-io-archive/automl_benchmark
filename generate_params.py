import argparse

import numpy as np
import pandas as pd

from benchmark.analysis import generate_tests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=10, type=int,
                        help='Number of experiments to generate per framework')
    parser.add_argument('--seed', default=11, type=int,
                        help='Random seed in sampling experiments')
    parser.add_argument('--foreshadow', action='store_true',
                        help='Use the Foreshadow framework. '
                             'Otherwise, defaults to using all other methods in '
                             '`automl_benchmark` repo')
    return parser.parse_args()


def add_framework_field(df: pd.DataFrame, framework: str) -> pd.DataFrame:
    df = df.copy()
    df['m_name'] = framework
    return df


def generate_test_subset(n: int = 10, seed: int = 0) -> pd.DataFrame:
    FRAMEWORKS = ('auto_ml', 'auto_sklearn', 'foreshadow', 'h2o', 'tpot')
    COLUMNS = ['m_name', 'd_id', 'm_type', 'seed']

    df = pd.DataFrame(generate_tests(),
                      columns=('id', 'm_name', 'd_id', 'm_type', 'seed'))

    df = (
        df
        .set_index('id')
        .drop(['m_name'], axis=1)
        .drop_duplicates()
        .sample(n=n, random_state=seed)
    )

    df = pd.concat([add_framework_field(df, f) for f in FRAMEWORKS], axis=0)
    df = (
        df[COLUMNS]
        .reset_index()
        .sort_values(['id', 'm_name'])
    )

    return df


def output_texts(df: pd.DataFrame):
    [print(' '.join([str(val) for val in row])) for row in df.itertuples(index=False)]


if __name__ == '__main__':
    args = parse_args()
    df = generate_test_subset(n=args.n, seed=args.seed)

    # Filter experiments that (do not) contain `foreshadow`
    if args.foreshadow:
        df = df[df['m_name'].isin(['foreshadow'])]
    else:
        df = df[~df['m_name'].isin(['foreshadow'])]

    output_texts(df)
