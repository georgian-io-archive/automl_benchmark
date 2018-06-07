#!/usr/bin/env python

import pandas as pd

def set_print_options(rows=None, cols=None):
	if not rows:
		pd.set_option('display.max_rows', rows)
	if not cols:
		pd.set_option('display.max_columns', cols)

def analyze_runs():
	runs_df = pd.read_csv('./compiled_results.csv', header=0)

	print('MEAN')
	print(runs_df.drop(columns=['SEED']).groupby(['TYPE', 'MODEL', 'DATASET_ID']).mean())
	print('VARIANCE')
	print(runs_df.drop(columns=['SEED']).groupby(['TYPE', 'MODEL', 'DATASET_ID']).var())

def get_nan_rows()
	runs_df = pd.read_csv('./compiled_results.csv', header=0)

	grouped = runs_df.groupby('TYPE')
	nan_ids = []

	for g, df in grouped:
		if g == 'classification':
			df = df.drop(columns=['RMSE', 'R2_SCORE'])
			nan_ids.extend(df[df.isnull().any(axis=1)]['ID'].values.ravel().tolist())
		else:
			df = df.drop(columns=['LOGLOSS', 'F1_SCORE'])
			nan_ids.extend(df[df.isnull().any(axis=1)]['ID'].values.ravel().tolist())

	return nan_ids

if __name__ == '__main__':
	# analyze_runs()
	set_print_options()
	get_nan_rows()
