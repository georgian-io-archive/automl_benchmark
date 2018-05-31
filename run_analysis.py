#!/usr/bin/env python

import pandas as pd

def analyze_runs():
	runs_df = pd.read_csv('./compiled_results.csv', header=0)

	print('MEAN')
	print(runs_df.drop(columns=['SEED']).groupby(['TYPE', 'MODEL', 'DATASET_ID']).mean())
	print('VARIANCE')
	print(runs_df.drop(columns=['SEED']).groupby(['TYPE', 'MODEL', 'DATASET_ID']).var())

if __name__ == '__main__':
	analyze_runs()