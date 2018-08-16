import itertools
from operator import itemgetter
import os

from colour import Color, color_scale, hsl2hex
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
plt.style.use(['fivethirtyeight'])


from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

from .benchmark import generate_tests


def set_print_options(rows=None, cols=None):
    """Sets the print options for pandas to show all columns or rows"""

    if not rows:
        pd.set_option('display.max_rows', rows)
    if not cols:
        pd.set_option('display.max_columns', cols)


def square_fac(n):
    """Gets the factors closest to square of a number n"""

    upper_bound = int(n**0.5)+1
    for c in range(upper_bound, 0, -1):
        if n % c == 0: break
    rslts = [c, int(n/c)]
    return min(rslts), max(rslts)


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

    return missing_df, len(tests)

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
    drop_dids = pd.unique(drop_datasets).tolist()
    drop_num = len(set(runs_df['DATASET_ID'].values.tolist()) & set(drop_dids))
    runs_df = runs_df[~runs_df['DATASET_ID'].isin(drop_dids)]

    return runs_df, drop_num

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
    runs_df = runs_df.set_index(['DATASET_ID', 'SEED'])
    runs_df = runs_df.drop(index=drop_tuples).reset_index()
    runs_df = runs_df[['ID', 'MODEL', 'DATASET_ID', 'TYPE', 'SEED', 'MSE', 'R2_SCORE', 'LOGLOSS', 
                       'F1_SCORE']]

    return runs_df, len(drop_tuples)


def split_by_type(runs_df):
    runs_grouped = runs_df.groupby('TYPE')
    return (runs_grouped.get_group('classification').drop(columns=['MSE', 'R2_SCORE']),
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

def correlation_viz(mu_df, targets):
    """Creates scatterplots of correlation betwene dataset stats and model performance 
    Args:
        mu_df (pd.DataFrame): A dataframe holding all failures
        targets (dict(str,list(str))): Column names from mu_df to perform analysis on
    """

    def get_true_features(d_id):
        df = pd.read_csv('datasets/{}.csv'.format(d_id))
        df_types = pd.read_csv('datasets/{}_types.csv'.format(d_id))

        #Get categorical encoded column count
        df_types_cat = df_types.loc[df_types['TYPE'] == 'categorical']['NAME']
        df_cat = df[df.columns.intersection(df_types_cat.values)]
        uniques = [len(df_cat[col].unique()) for col in df_cat]

        df_types_num = df_types.loc[df_types['TYPE'] == 'numerical']['NAME']
        df_num = df[df.columns.intersection(df_types_num.values)]
        
        count = np.sum(uniques) + len(df_num.columns)

        return count

    plt.gcf().set_size_inches(20, 15)  
    meta_c_df = pd.read_csv('datasets/study_classification_info.csv')
    meta_r_df = pd.read_csv('datasets/study_regression_info.csv')
    meta_df = pd.concat([meta_c_df, meta_r_df])
    meta_df['DIMENSIONALITY'] = meta_df.apply(lambda row: get_true_features(row['DATASET_ID']), axis=1)
    full_data = pd.merge(mu_df, meta_df, how='left')    
    models = full_data['MODEL'].unique()

    row_size = max([len(x) for x in targets.values()])
    base_colors = [hsl2hex(c) for c in color_scale((0., 0.8, 0.6), (0.8, 0.8, 0.6), len(models))]
    lines = None
    for j, TYPE in enumerate(targets):
        for i, BASE in enumerate(targets[TYPE][1]): 
     
            all_data = full_data.loc[full_data['TYPE'] == TYPE]
            all_data = all_data[['MODEL','DATASET_ID',BASE,targets[TYPE][0]]]
            all_data = all_data.groupby(['MODEL','DATASET_ID',BASE], as_index=False).mean()
            all_data = all_data.sort_values(BASE)

            plt.subplot(len(targets),row_size,row_size*j+i+1)

            ylabel_str = targets[TYPE][0]
            if ylabel_str.lower() == 'mse':
                label_str = 'standardized negated mse'

            plt.xlabel(BASE.replace('_',' ').capitalize())
            plt.ylabel("{} {}".format(TYPE.replace('_',' ').capitalize(), 
                                      ylabel_str.replace('_',' ').capitalize()))
            local_lines = []
            for k, m in enumerate(models):
                ss = all_data.loc[all_data['MODEL'] == m]
                ss[BASE] = ss[BASE].rolling(int(len(ss[BASE])/2)).median()
                ss[targets[TYPE][0]] = ss[targets[TYPE][0]].rolling(int(len(ss[BASE])/2)).median()

                x = ss[BASE]
                y = ss[targets[TYPE][0]]
                line, = plt.plot(x, y, color=base_colors[k], alpha=0.7, label=m)
                local_lines.append(line)
            if lines == None:
                lines = local_lines
    
    plt.figlegend(lines, models, fancybox=True, framealpha=0.0)
    plt.gcf().suptitle('Dataset Dependent Performance Analysis') 

    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/DatasetPerformance.pdf', dpi=plt.gcf().dpi, transparent=True)
    # plt.show()
    

def dataset_viz(mu_df, targets):
    """Creates histograms for given dataset, type filter and targets
    Args:
        mu_df (pd.DataFrame): A dataframe holding all failures
        targets (dict(str,list(str))): Column names from mu_df to perform analysis on
    """

    def lengths(x):
        if isinstance(x,list):
            yield len(x)
            for y in x:
                yield from lengths(y)

    rows = len(targets.keys())
    cols = max(lengths(list(targets.values())))
    fig, axes_list = plt.subplots(rows, cols)
    axes_list[-1, -1].axis('off')
    fig.set_size_inches(18, 8)

    meta_c_df = pd.read_csv('datasets/study_classification_info.csv')
    meta_r_df = pd.read_csv('datasets/study_regression_info.csv')
    meta_df = pd.concat([meta_c_df, meta_r_df])

    row_size = max([len(x) for x in targets.values()])
    base_colors = [hsl2hex(c) for c in color_scale((0., 0.8, 0.6), (0.8, 0.8, 0.6), row_size)]

    for j, TYPE in enumerate(targets):
        for i, BASE in enumerate(targets[TYPE]):             
            all_data = pd.merge(mu_df.loc[mu_df['TYPE']==TYPE], meta_df, how='left')   
            full_data = pd.merge(mu_df, meta_df, how='left')

            ax = axes_list[j][i]
            ax.set_xlabel(BASE.capitalize() + " Count (Log Scale)")
            ax.set_ylabel("{} Frequency".format(BASE.capitalize()))
            counts, bins, bars = ax.hist(all_data[BASE], 
                                         bins=np.logspace(np.log10(np.min(full_data[BASE])), 
                                                          np.log10(np.max(full_data[BASE])), 
                                                          30), 
                                         stacked=True, 
                                         color=base_colors[i],
                                         alpha=0.7,
                                         edgecolor='black',
                                         linewidth=0.6)
            ax.set_xscale('log')

    fig.suptitle('Content Analysis of Datasets')
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/DatasetShapes.pdf', dpi=fig.dpi, transparent=True)
    # plt.show()

def pairwise_comp_viz(mu_df, target):
    """Creates a pariwise interaction visualization plot comparing each model against the other
    Args:
        mu_df (pd.Dataframe): A dataframe of valid runs with type and model as indicies with aggregated
                              means across runs
        c_df_info (pd.Dataframe): A dataframe with information about each dataset type
    """

    def plot_comp(mu_df, m1, m2, target, vmin, vmax, cmap, ax):
        m1_values = mu_df.xs(m1, level=1).values
        m2_values = mu_df.xs(m2, level=1).values

        # difference from y=x color mapping (not magnitude because independent)
        colors = np.array([m_2 - m_1 for m_2, m_1 in zip(m2_values, m1_values)])

        sc = ax.scatter(m1_values, m2_values, alpha=0.7, s=15, c=colors, cmap=cmap, zorder=10, 
            norm=MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0))
        ax.set_xlabel(m1)
        ax.set_ylabel(m2)

        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'k-', lw=0.7, alpha=0.7, zorder=0)
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

    sort_order = {'auto-sklearn': 1, 'tpot': 2, 'h2o': 3, 'auto_ml': 4}
    mu_df = mu_df[target]
    models = sorted(pd.unique(mu_df.index.get_level_values('MODEL').values), key=lambda x: sort_order[x])
    combos = list(itertools.combinations(models, 2))
    sorted_combos = list(sorted(combos, key=lambda x: (sort_order[x[0]], sort_order[x[1]])))
    plot_count = len(sorted_combos)
    rows, cols = square_fac(plot_count)
    fig, ax_list = plt.subplots(rows, cols)
    fig.set_size_inches(17, 8)
    metric_name = target.replace('_', ' ').title() if target == 'F1_SCORE' else target
    base_colors = [hsl2hex(c) for c in color_scale((0, 0.7, 0.4), (1, 0.7, 0.4), plot_count)]
    model_colors = {m: c for m, c in zip(models, base_colors)}
    color_bins = 10
    scatters = []

    # get min-max of differences
    vmin = np.inf
    vmax = -np.inf
    for m1, m2 in sorted_combos:
        m1_values = mu_df.xs(m1, level=1).values
        m2_values = mu_df.xs(m2, level=1).values
        colors = np.array([m_2 - m_1 for m_2, m_1 in zip(m2_values, m1_values)])
        if np.max(colors) > vmax:
            vmax = np.max(colors)
        if np.min(colors) < vmin:
            vmin = np.min(colors)

    for combo, ax in zip(sorted_combos, ax_list.ravel()):
        m1, m2 = combo
        color_range = get_color_range(Color(model_colors[m1]), Color(model_colors[m2]), color_bins)
        cmap = mpl.colors.ListedColormap(color_range)
        scatters.append(plot_comp(mu_df, m1, m2, target, vmin, vmax, cmap, ax))

    for sc, ax in zip(scatters, ax_list.ravel()):
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.08)
        cbar.ax.tick_params(labelsize=10)
        ax_str = '{} Difference' if target == 'F1_score' else 'Standardized Inverted {} Difference'
        cbar.set_label(ax_str.format(metric_name), rotation=90, fontsize=8,
            labelpad=-57) # if target == 'F1_SCORE' else -65)

    fig.suptitle('Dataset Mean {} Across Frameworks'.format(metric_name))

    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/DatasetMean{}.pdf'.format(metric_name.replace(' ', '')), dpi=fig.dpi, 
                                                                                  transparent=True)
    # plt.show()

def boxplot_viz(clean_df, target):
    clean_df = clean_df[target]
    models = pd.unique(clean_df.index.values)
    data_arr = np.array([clean_df[m].values for m in models]).T
    base_colors = [hsl2hex(c) for c in color_scale((0., 0.8, 0.6), (0.8, 0.8, 0.6), len(models))]
    plt.figure(figsize=(7, 3.5))
    title_str = "Raw Per Model {} Comparison ({})".format('Classification' if target=='F1_SCORE' else 'Regression', target)
    plt.title(title_str, size=12)
    bplot = plt.boxplot(data_arr, vert=False, patch_artist=True, notch=True, labels="    ", positions=list(reversed(range(1, len(models)+1))))

    for p, c in zip(bplot['boxes'], base_colors):
        p.set_facecolor(c)

    plt.legend(bplot['boxes'], models, loc='lower left', prop={'size': 8}, fancybox=True, framealpha=0.6)
    plt.setp(bplot['fliers'], markeredgecolor='grey')
    plt.setp(bplot['medians'], color='black')

    # plt.show()
    plt.savefig('figures/RawDataBoxPlot{}.pdf'.format(target), dpi=plt.gcf().dpi, transparent=True)

def standardize_scale(runs_df, target, invert=False):
    runs_df = runs_df.copy()
    print('Standardizing and scaling {}...'.format(target))
    m_type = 'classification' if target == 'F1_SCORE' else 'regression'
    d_ids = pd.unique(runs_df[runs_df['TYPE'] == m_type]['DATASET_ID'].values)
    for d_id in d_ids:
        transformation = MinMaxScaler().fit_transform(zscore(
            runs_df[runs_df['DATASET_ID'] == d_id][target].values).reshape((-1, 1))).ravel()
        runs_df.loc[runs_df['DATASET_ID'] == d_id, target] = 1 - transformation if invert else transformation
    return runs_df

def per_model_median_confidence(runs_df):
    """Computes the grouped median and iqr by model type
    Args: 
        runs_df (pd.Dataframe): A list of all the runs
    Returns:
        A tuple of pandas Dataframes that represent the median and iqr of each model
    """

    overall = runs_df.drop(columns=['SEED', 'ID']).groupby(['TYPE', 'MODEL', 'DATASET_ID'], 
        as_index=False).mean()
    collected = overall.drop(columns=['DATASET_ID']).groupby(['TYPE', 'MODEL'])
    N = len(runs_df)/4
    return collected.median(), 1.57*(collected.quantile(0.75)-collected.quantile(0.25))/np.sqrt(N)

def per_model_mean(runs_df):
    """Computes the grouped mean and std by model type
    Args: 
        runs_df (pd.Dataframe): A list of all the runs
    Returns:
        A tuple of pandas Dataframes that represent the mean and std of each model
    """

    overall = runs_df.drop(columns=['SEED', 'ID']).groupby(['TYPE', 'MODEL', 'DATASET_ID'], 
        as_index=False).mean()
    collected = overall.drop(columns=['DATASET_ID']).groupby(['TYPE', 'MODEL'])
    return collected.mean()


def per_dataset_mean_std(runs_df):
    """Computes the overall mean and median of each dataset grouped by model
    Args: 
        runs_df (pd.Dataframe): A list of all the runs
    Returns:
        A tuple of pandas Dataframes that represent the mean and variance of each dataset by model
    """

    processed = runs_df.drop(columns=['SEED', 'ID']).groupby(['TYPE', 'MODEL', 'DATASET_ID'])
    return processed.mean(), processed.std()


def original_dataset_clean(runs_df):
    return runs_df.drop(columns=['SEED', 'ID']).set_index(['TYPE', 'MODEL'])


def analysis_suite():
    """An automatic suite that performs analysis on the computed results of the benchmarking process"""

    runs_df = pd.read_csv('./compiled_results.csv')
    runs_df['R2_SCORE'] = runs_df['R2_SCORE'].abs()
    missing_df, total_run_count = compute_missing_runs(runs_df)
    runs_df, drop_d_count = drop_missing_datasets(runs_df, missing_df, 10)
    runs_df, drop_r_count = drop_missing_runs(runs_df, missing_df)
    scaled_df = standardize_scale(runs_df, 'MSE', invert=True)
    scaled_df = standardize_scale(scaled_df, 'F1_SCORE')
    c_df, r_df = split_by_type(scaled_df)
    raw_c_df, raw_r_df = split_by_type(runs_df)
    c_mu, c_std = per_dataset_mean_std(c_df)
    r_mu, r_std = per_dataset_mean_std(r_df)
    c_median, c_iqr = per_model_median_confidence(c_df)
    r_median, r_iqr = per_model_median_confidence(r_df)
    raw_c_mu = per_model_mean(raw_c_df)
    raw_r_mu = per_model_mean(raw_r_df)
    total_dropped_points = drop_d_count*40+drop_r_count*4

    print('Missing by Model...\n', missing_df['MODEL'].value_counts())
    print('Total dropped datasets: ', drop_d_count)
    print('Other dropped points: ', drop_r_count)
    print('percentage {}/{}: {}'.format(total_dropped_points,
                                        total_run_count,
                                        total_dropped_points/total_run_count))
    
    print('Classification per model medians...\n', c_median.round(3))
    print('Classification per model iqrs...\n', c_iqr.round(3))
    print('Regression per model medians...\n', r_median.round(3))
    print('Regression per model iqrs...\n', r_iqr.round(3))

    print('Raw Classification per model means...\n', raw_c_mu.round(3))
    print('Raw Regression per model means...\n', raw_r_mu.round(3))

    print('Creating classification pairwise visualization...')
    pairwise_comp_viz(c_mu, target='F1_SCORE')
    print('Creating regression pairwise visualization...')
    pairwise_comp_viz(r_mu, target='MSE')
    
    print('Creating dataset visualization...')
    dataset_viz(scaled_df, targets={'classification':['FEATURES','ROWS','CLASSES'],
                                  'regression':['FEATURES','ROWS']}) 

    print('Creating metric correlation visualization...')
    correlation_viz(scaled_df, targets={'classification':('F1_SCORE',['DIMENSIONALITY','ROWS']),
                                       'regression':('MSE',['DIMENSIONALITY','ROWS'])})

    print('Creating classification boxplot visualization...')
    boxplot_viz(c_df.drop(columns=['ID', 'SEED', 'TYPE']).set_index(['MODEL']), target='F1_SCORE')
    print('Creating regression boxplot visualization...')
    boxplot_viz(r_df.drop(columns=['ID', 'SEED', 'TYPE']).set_index(['MODEL']), target='MSE')