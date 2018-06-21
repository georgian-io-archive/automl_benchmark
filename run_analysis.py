#!/usr/bin/env python

import itertools
import os

from colour import Color, color_scale, hsl2hex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(['fivethirtyeight'])
import matplotlib as mpl
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

from benchmark import generate_tests


def set_print_options(rows=None, cols=None):
    """Sets the print options for pandas to show all columns or rows"""

    if not rows:
        pd.set_option('display.max_rows', rows)
    if not cols:
        pd.set_option('display.max_columns', cols)


def compute_missing_runs(runs_df):
    """Computes the runs which don't have results
    Args:
        runs_df (pd.Dataframe): A list of all the runs 
    Returns:
        A pandas dataframe of all the missing runs
    """

    tests = generate_tests()

    test_len = len(tests)
    keyed = {test_id: t for test_id, t in zip(range(test_len), tests)}

    missing = [keyed[i] for i in range(test_len) if i not in runs_df['ID'].tolist()]

    missing_df = pd.DataFrame(missing, columns=['ID', 'MODEL', 'DATASET_ID', 'TYPE', 'SEED'])
    missing_df['DATASET_ID'] = missing_df['DATASET_ID'].astype(int)

    return missing_df

def drop_missing_datasets(runs_df, missing_df, missing_thresh):
    """If a dataset is missing more than or equal to the missing_thresh for a specific combination of 
       model and dataset, the dataset and its data is dropped from all models
    Args:
        runs_df (pd.Dataframe): A list of all computed runs
        missing_df (pd.Dataframe): A list of all missing runs
        missing_thresh (int): missing threshold (0-10)
    Returns:
        An augmented pandas dataframe with removed datasets
    """

    counts = missing_df.groupby(['TYPE', 'MODEL'])['DATASET_ID'].value_counts()
    counts = counts[counts >= missing_thresh]
    drop_datasets = counts.index.get_level_values('DATASET_ID').values
    drop_dids = np.unique(drop_datasets).tolist()
    runs_df = runs_df[~runs_df['DATASET_ID'].isin(drop_dids)]

    return runs_df

def drop_missing_runs(runs_df, missing_df):
    """In order to make the comparisons even across models, all runs that did not complete in one
       model are removed from all models
    Args:
        missing_df (pd.Dataframe): A list of all missing runs
    Returns:
        A index list 
    """
    drop_tuples = list(set(missing_df.set_index(['DATASET_ID', 'SEED']).index.values.tolist()))
    dataset_missing = pd.DataFrame(drop_tuples, columns=['DATASET_ID', 'SEED'])['DATASET_ID'].value_counts()
    drop_dids = dataset_missing[dataset_missing > 5].index.values.tolist()
    runs_df = runs_df[~runs_df['DATASET_ID'].isin(drop_dids)]
    runs_df = runs_df.set_index(['DATASET_ID', 'SEED'])
    runs_df = runs_df.drop(index=drop_tuples).reset_index()
    runs_df = runs_df[['ID', 'MODEL', 'DATASET_ID', 'TYPE', 'SEED', 'RMSE', 'R2_SCORE', 'LOGLOSS', 
                       'F1_SCORE']]

    return runs_df


def split_by_type(runs_df):
    runs_grouped = runs_df.groupby('TYPE')
    return (runs_grouped.get_group('classification').drop(columns=['RMSE', 'R2_SCORE']),
            runs_grouped.get_group('regression').drop(columns=['F1_SCORE', 'LOGLOSS']))

def data_distributions(data_df, target):
    """Plots the spread of multiple runs (seeds) across a dataframe
    Args:
        data_df (pd.Dataframe): A dataframe holding the results of the runs
        target (str): a pandas column header to represent the response variable
    """

    grouped = data_df.groupby(['MODEL', 'DATASET_ID'])
    for k, df in grouped:
        plt.hist(df['F1_SCORE'].values, alpha=0.5, label=k)
    plt.legend(loc='upper right')
    plt.show()


def pairwise_comp_viz(mu_df, target):
    """Creates a pariwise interaction visualization plot comparing each model against the other
    Args:
        mu_df (pd.Dataframe): A dataframe of valid runs with type and model as indicies with aggregated
                              means across runs
        c_df_info (pd.Dataframe): A dataframe with information about each dataset type
    """

    def square_fac(n):
        upper_bound = int(n**0.5)+1
        for c in range(upper_bound, 0, -1):
            if n % c == 0: break
        rslts = [c, int(n/c)]
        return min(rslts), max(rslts)

    def plot_comp(mu_df, m1, m2, target, vmin, vmax, cmap, ax):
        m1_values = mu_df.xs(m1, level=1).values
        m2_values = mu_df.xs(m2, level=1).values

        # difference from y=x color mapping (not magnitude because independent)
        colors = np.array([m_2 - m_1 for m_2, m_1 in zip(m2_values, m1_values)])

        sc = ax.scatter(m1_values, m2_values, alpha=0.7, s=15, c=colors, cmap=cmap, zorder=10, 
            norm=MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0))
        ax.set_xlabel(m1)
        ax.set_ylabel(m2)
        ax.axhline(c='black', lw=1, alpha=0.5)
        ax.axvline(c='black', lw=1, alpha=0.5)

        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'k-', lw=2, alpha=0.2, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        return sc

    class MidpointNormalize(mpl.colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    def get_color_range(c1, c2, bins):
        MAX_L = 0.7
        c1h, c1s, c1l = c1.hsl
        c2h, c2s, c2l = c2.hsl
        c1_bins = [Color(hsl=(c1h, c1s, var_l)) for var_l in np.linspace(c1l, MAX_L, int(bins/2))]
        c2_bins = [Color(hsl=(c2h, c2s, var_l)) for var_l in np.linspace(MAX_L, c2l, int(bins/2))]
        color_range = [c.hex_l for c in (c1_bins + c2_bins)]
        return color_range

    mu_df = mu_df[target]
    models = np.unique(mu_df.index.get_level_values('MODEL').values)
    combos = list(itertools.combinations(models, 2))
    plot_count = len(combos)
    rows, cols = square_fac(plot_count)
    fig, ax_list = plt.subplots(rows, cols)
    fig.set_size_inches(18, 10)
    metric_name = target.replace('_', ' ').title() if target == 'F1_SCORE' else target
    base_colors = [hsl2hex(c) for c in color_scale((0, 0.7, 0.4), (1, 0.7, 0.4), plot_count)]
    model_colors = {m: c for m, c in zip(models, base_colors)}
    color_bins = 10
    scatters = []

    # get min-max of differences
    vmin = np.inf
    vmax = -np.inf
    for m1, m2 in combos:
        m1_values = mu_df.xs(m1, level=1).values
        m2_values = mu_df.xs(m2, level=1).values
        colors = np.array([m_2 - m_1 for m_2, m_1 in zip(m2_values, m1_values)])
        if np.max(colors) > vmax:
            vmax = np.max(colors)
        if np.min(colors) < vmin:
            vmin = np.min(colors)

    for combo, ax in zip(combos, ax_list.ravel()):
        m1, m2 = combo
        color_range = get_color_range(Color(model_colors[m1]), Color(model_colors[m2]), color_bins)
        cmap = mpl.colors.ListedColormap(color_range)
        scatters.append(plot_comp(mu_df, m1, m2, target, vmin, vmax, cmap, ax))

    for sc, ax in zip(scatters, ax_list.ravel()):
        cbar = fig.colorbar(sc, ax=ax) # fraction=0.046, pad=0.04,
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Normalized {} Difference'.format(metric_name), rotation=90, fontsize=8,
            labelpad=-55) # if target == 'F1_SCORE' else -65)

    fig.suptitle('Dataset Mean {} Across Frameworks'.format(metric_name))
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/DatasetMean{}.png'.format(metric_name.replace(' ', '')), dpi=1000)
    # plt.show()


def standardize_rmse(runs_df):
    print('Standardizing and scaling RMSE...')
    regression_dids = np.unique(runs_df[runs_df['TYPE'] == 'regression']['DATASET_ID'].values)
    for d_id in regression_dids:
        runs_df.loc[runs_df['DATASET_ID'] == d_id, 'RMSE'] = 1 - MinMaxScaler().fit_transform(zscore(
            runs_df[runs_df['DATASET_ID'] == d_id]['RMSE'].values).reshape((-1, 1))).ravel()

    return runs_df

def per_model_mean_std(runs_df):
    """Computes the grouped mean and median by model type
    Args: 
        runs_df (pd.Dataframe): A list of all the runs
    Returns:
        A tuple of pandas Dataframes that represent the mean and variance of each model
    """

    overall = runs_df.drop(columns=['SEED', 'ID']).groupby(['TYPE', 'MODEL', 'DATASET_ID'], 
        as_index=False).mean()
    collected = overall.drop(columns=['DATASET_ID']).groupby(['TYPE', 'MODEL'])
    return collected.mean(), collected.std()


def per_dataset_mean_std(runs_df):
    """Computes the overall mean and median of each dataset grouped by model
    Args: 
        runs_df (pd.Dataframe): A list of all the runs
    Returns:
        A tuple of pandas Dataframes that represent the mean and variance of each dataset by model
    """

    processed = runs_df.drop(columns=['SEED', 'ID']).groupby(['TYPE', 'MODEL', 'DATASET_ID'])
    return processed.mean(), processed.std()


def analysis_suite():
    """An automatic suite that performs analysis on the computed results of the benchmarking process"""

    runs_df = pd.read_csv('./compiled_results.csv')
    runs_df['R2_SCORE'] = runs_df['R2_SCORE'].abs()
    missing_df = compute_missing_runs(runs_df)
    runs_df = drop_missing_datasets(runs_df, missing_df, 10)
    runs_df = drop_missing_runs(runs_df, missing_df)
    runs_df = standardize_rmse(runs_df)
    c_df, r_df = split_by_type(runs_df)
    cd_mu, cd_std = per_dataset_mean_std(c_df)
    rd_mu, rd_std = per_dataset_mean_std(r_df)
    c_mu, c_std = per_model_mean_std(c_df)
    r_mu, r_std = per_model_mean_std(r_df)

    deduplicated_missing = list(set([tuple(v) for v in missing_df[['DATASET_ID', 
                                                                   'SEED']].values]))
    deduplicated_df = pd.DataFrame(deduplicated_missing, columns=['DATASET_ID', 'COUNT'])
    drop_counts = deduplicated_df['DATASET_ID'].value_counts()
    dropped_dataset_count = len(drop_counts[drop_counts > 5])
    dropped_points = drop_counts[drop_counts <= 5].sum()
    total_dropped_points = dropped_dataset_count*10+dropped_points*4
    print('Dropped items per datasets (>5 drop entire dataset)...')
    print(drop_counts)
    print('Total dropped datasets: ', dropped_dataset_count)
    print('Other dropped points: ', dropped_points)
    print('percentage {}/5200: {}'.format(total_dropped_points, 
                                        total_dropped_points/5200))
    
    print('Classification per model means...\n', c_mu)
    print('Classification per model standard deviation...\n', c_std)
    print('Regression per model means...\n', r_mu)
    print('Regression per model standard deviation...\n', r_std)

    print('Creating classification visualization...')
    pairwise_comp_viz(cd_mu, target='F1_SCORE')
    print('Creating regression visualization...')
    pairwise_comp_viz(rd_mu, target='RMSE')


if __name__ == '__main__':
    set_print_options()
    analysis_suite()
