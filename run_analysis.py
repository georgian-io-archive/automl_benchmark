#!/usr/bin/env python

import numpy as np
import pandas as pd
from benchmark import generate_tests


def set_print_options(rows=None, cols=None):
	if not rows:
		pd.set_option('display.max_rows', rows)
	if not cols:
		pd.set_option('display.max_columns', cols)


def compute_missing_runs(runs_df):
	tests = generate_tests()
	test_len = len(tests)
	keyed = {test_id: t for test_id, t in zip(range(test_len), tests)}

	missing = [keyed[i] for i in range(test_len) if i not in runs_df['ID'].tolist()]

	missing_df = pd.DataFrame(missing, columns=['ID', 'MODEL', 'DATASET_ID', 'TYPE', 'SEED'])

	return missing_df


def model_mean_var(runs_df):
	overall = runs_df.drop(columns=['SEED', 'ID']).groupby(['TYPE', 'MODEL', 'DATASET_ID'], 
		as_index=False).mean()
	collected = overall.drop(columns=['DATASET_ID']).groupby(['TYPE', 'MODEL'])
	return collected.mean(), collected.var()


def overall_mean_var(runs_df):
	processed = runs_df.drop(columns=['SEED', 'ID']).groupby(['TYPE', 'MODEL', 'DATASET_ID'])
	return processed.mean(), processed.var()


def analysis_suite():
	runs_df = pd.read_csv('./compiled_results.csv', header=0)

	missing_df = compute_missing_runs(runs_df)
	print('Missing Counts...')
	print(missing_df.groupby('MODEL')['DATASET_ID'].value_counts())
	print('Total Count: ', len(missing))

	overall_mu, overall_sigma2 = overall_mean_var(runs_df)
	print('Means...\n', overall_mu)
	print('Variances...\n', overall_sigma2)

	model_mu, model_sigma2 = model_mean_var(runs_df)
	print('Means...\n', model_mu)
	print('Variances...\n', model_sigma2)


if __name__ == '__main__':
	set_print_options()
	analysis_suite()
