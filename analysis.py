#!/usr/bin/env python3

"""
Generate the figures and results for the paper: "OrionBench:
User Centric Benchmarking Framework for Unsupervised Time 
Series Anomaly Detection."
"""

import ast
import itertools
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from orion.benchmark import BENCHMARK_DATA

DATASET_FAMILY = {
    "MSL": "NASA",
    "SMAP": "NASA",
    "YAHOOA1": "YAHOO",
    "YAHOOA2": "YAHOO",
    "YAHOOA3": "YAHOO",
    "YAHOOA4": "YAHOO",
    "artificialWithAnomaly": "NAB",
    "realAWSCloudwatch": "NAB",
    "realAdExchange": "NAB",
    "realTraffic": "NAB",
    "realTweets": "NAB"
}

DATASET_ABBREVIATION = {
    "MSL": "MSL",
    "SMAP": "SMAP",
    "YAHOOA1": "A1",
    "YAHOOA2": "A2",
    "YAHOOA3": "A3",
    "YAHOOA4": "A4",
    "artificialWithAnomaly": "Art",
    "realAWSCloudwatch": "AWS",
    "realAdExchange": "AdEx",
    "realTraffic": "Traf",
    "realTweets": "Tweets"
}

OUTPUT_PATH = Path('output')

BUCKET = 'sintel-orion'
S3_URL = 'https://{}.s3.amazonaws.com/{}'
GITHUB_URL = 'https://raw.githubusercontent.com/sintel-dev/Orion/master/benchmark/results/{}.csv'

DATA_MAP = {signal: data for data, signals in BENCHMARK_DATA.items() for signal in signals}

_VERSION = ['0.1.3', '0.1.4', '0.1.5', '0.1.6', '0.1.7', '0.2.0', '0.2.1', '0.3.0', '0.3.1', '0.3.2', '0.4.0', '0.4.1', '0.5.0']
_ORDER = ['aer', 'lstm_dynamic_threshold', 'arima', 'lstm_autoencoder', 'tadgan', 'vae', 'dense_autoencoder', 'ganf', 'azure']
_LABELS = ['AER', 'LSTM DT', 'ARIMA', 'LSTM AE', 'TadGAN', 'VAE', 'Dense AE', 'GANF', 'Azure AD']
_COLORS = ["#ED553B", "#F6D55C", "#3CAEA3", "#4287F5", "#20639B", "#173F5F", "#5E3A94", "#4A4A4A", "#8F96A2"]
_MARKERS = ['o', 's', 'v', 'X', 'p', '^', 'd', 'P', '>']

_PALETTE = sns.color_palette("Spectral")
_PALETTE.append("#173F5F")
_PALETTE.append("#4A4A4A")
_PALETTE.append("#8F96A2")
# _PALETTE = sns.color_palette(_COLORS)

# ------------------------------------------------------------------------------
# Saving results
# ------------------------------------------------------------------------------

def _savefig(fig, name, figdir=OUTPUT_PATH):
    figdir = Path(figdir)
    for ext in ['.png', '.pdf', '.eps']:
        fig.savefig(figdir.joinpath(name+ext),
                    bbox_inches='tight')

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def _get_f1_scores(results):
    df = results.groupby(['dataset', 'pipeline'])[['fp', 'fn', 'tp']].sum().reset_index()

    precision = df['tp'] / (df['tp'] + df['fp'])
    recall = df['tp'] / (df['tp'] + df['fn'])
    df['f1'] = 2 * (precision * recall) / (precision + recall)

    df = df.set_index(['dataset', 'pipeline'])[['f1']].unstack().T.droplevel(0)

    df.columns = [DATASET_ABBREVIATION[col] for col in df.columns]
    df.columns = pd.MultiIndex.from_tuples(list(zip(DATASET_FAMILY.values(), df.columns)))

    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)

    df.insert(0, 'Pipeline', df.index)
    df = df.reset_index(drop=True)

    return df

# ------------------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------------------

def table_2():
	data = pd.read_csv('data_summary.csv')
	data['anomaly_len'] = data['anomaly_len'].apply(ast.literal_eval)
	summary = data.groupby('dataset')['count'].agg(['count', 'sum'])
	signals = data.groupby('dataset')['signal_len'].mean()
	anomaly = data.groupby('dataset')['anomaly_len'].apply(lambda x: np.mean(list(itertools.chain.from_iterable(x))))
	summary = pd.concat([summary, signals, anomaly], axis=1)
	return summary

def table_3():
	df = pd.read_csv('benchmark.csv')
	df = df.groupby(['dataset', 'pipeline', 'iteration'])[['fp', 'fn', 'tp']].sum().reset_index()

	precision = df['tp'] / (df['tp'] + df['fp'])
	recall = df['tp'] / (df['tp'] + df['fn'])
	df['f1'] = 2 * (precision * recall) / (precision + recall)

	df = df.set_index(['dataset', 'pipeline', 'iteration'])[['f1']].reset_index()
	df['family'] = df['dataset'].apply(lambda x: DATASET_FAMILY[x])
	df['dataset'] = df['dataset'].apply(lambda x: DATASET_ABBREVIATION[x])

	data = df.groupby(['dataset', 'family', 'pipeline'])[['f1']].agg(["mean", "std"]).droplevel(0, axis=1)
	data['value'] = data["mean"].round(3).astype("str") + "+-" + data["std"].round(2).astype("str")
	data = data[['value']].unstack().T.droplevel(0)
	data = data[['MSL', 'SMAP', 'A1', 'A2', 'A3', 'A4', 'Art', 'AWS', 'AdEx', 'Traf', 'Tweets']]
	data = data.swaplevel(axis=1).loc[_ORDER]
	data.index = _LABELS
	return data

# ------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------

def figure_3():
	df = pd.read_csv('benchmark.csv')
	df = df.groupby(['dataset', 'pipeline', 'iteration'])[['fp', 'fn', 'tp']].sum().reset_index()

	precision = df['tp'] / (df['tp'] + df['fp'])
	recall = df['tp'] / (df['tp'] + df['fn'])
	df['f1'] = 2 * (precision * recall) / (precision + recall)

	df = df.set_index(['dataset', 'pipeline', 'iteration'])[['f1']].reset_index()
	df['family'] = df['dataset'].apply(lambda x: DATASET_FAMILY[x])
	df['dataset'] = df['dataset'].apply(lambda x: DATASET_ABBREVIATION[x])

	fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(12, 3))
	fig.subplots_adjust(wspace=0)

	data = df.set_index('pipeline').loc[_ORDER].reset_index()

	sns.boxplot(data[data['family'] == 'NASA'], x='pipeline', y='f1', palette=_PALETTE, ax=axes[0])
	sns.boxplot(data[data['dataset'].isin(['A1', 'A2'])], x='pipeline', y='f1', palette=_PALETTE, ax=axes[1])
	sns.boxplot(data[data['dataset'].isin(['A3', 'A4'])], x='pipeline', y='f1', palette=_PALETTE, ax=axes[2])
	sns.boxplot(data[data['family'] == 'NAB'], x='pipeline', y='f1', palette=_PALETTE, ax=axes[3])

	for i in range(4):
	    axes[i].grid(True, linestyle='--')
	    axes[i].set_xticklabels(_LABELS, rotation=90)
	    axes[i].set_xlabel('')
	    axes[i].set_ylabel('')
	    
	axes[0].set_xlim(-1, 9)
	axes[0].set_ylabel('F1 Score')

	axes[0].set_title("NASA")
	axes[1].set_title("Yahoo S5 (A1 & A2)")
	axes[2].set_title("Yahoo S5 (A3 & A4)")
	axes[3].set_title("NAB")

	# handles = [
	# 	 mpatches.Patch(color=_PALETTE[i], label=_LABELS[i])
	# 	 for i in range(len(_LABELS))
	# ]

	# plt.legend(handles=handles, bbox_to_anchor=(1.05, 0.95), edgecolor='black')

	_savefig(fig, 'benchmark', figdir=OUTPUT_PATH)

def figure_4():
	def map(x):
		if x == "0":
			return None, 0, 0, 0

		values = ast.literal_eval(x)
		return values

	results = []
	for version in _VERSION:
		df = pd.read_csv(GITHUB_URL.format(version))
	
		try:
			scores = _get_f1_scores(df)
		except:
			df['confusion_matrix'] = df['confusion_matrix'].apply(map)
			df[['tn', 'fp', 'fn', 'tp']] = pd.DataFrame(df['confusion_matrix'].tolist(), index=df.index)
			scores = _get_f1_scores(df)

		scores.columns = scores.columns.get_level_values(0)
		scores = scores[['Pipeline', 'mean']]
		scores.columns = ['pipeline', version]
		results.append(scores.set_index('pipeline'))

	df = pd.concat(results, axis=1)
	df = df.reindex(sorted(df.columns), axis=1)
	df = df.loc[_ORDER]

	fig = plt.figure(figsize=(5, 3))
	ax = plt.gca()

	for i, pipeline in enumerate(df.T.columns):
	  ax.plot(df.T[pipeline], marker=_MARKERS[i], markersize=7, color=_PALETTE[i], label=_LABELS[i])

	plt.grid(True, linestyle='--')
	plt.legend(bbox_to_anchor=(1.01, 0.93), edgecolor='black')
	plt.ylim([0.15, 0.8])
	plt.xticks(rotation=90)
	# plt.yticks(size=12)
	plt.ylabel('F1 Score')
	plt.xlabel('Version')
	plt.title('F1 Score Across Releases')
	
	_savefig(fig, 'version', figdir=OUTPUT_PATH)


def figure_5():
	df = pd.read_csv('benchmark.csv')
	df['family'] = df['dataset'].apply(lambda x: DATASET_FAMILY[x])
	df['dataset'] = df['dataset'].apply(lambda x: DATASET_ABBREVIATION[x])
	df = df.set_index('pipeline').loc[_ORDER].reset_index()

	fig = plt.figure(figsize=(5, 3))
	ax = plt.gca()
	    
	sns.barplot(data=df, x="pipeline", y="elapsed", palette=_PALETTE, 
				saturation=0.7, linewidth=0.5, edgecolor='k', ax=ax)

	ax.set_xticklabels(_LABELS, rotation=90)

	plt.grid(True, linestyle='--')
	plt.yscale('log')
	plt.ylim([0.1e1, 0.2e3])
	plt.ylabel('time in seconds (log)')
	plt.xlabel('')
	plt.title("Pipeline Elapsed Time")

	_savefig(fig, 'elapsed', figdir=OUTPUT_PATH)


def figure_6():
	results = []
	for version in _VERSION:
		df = pd.read_csv(GITHUB_URL.format(version))
		time = df.groupby(['dataset', 'pipeline'])['elapsed'].mean().reset_index()
		time = time.set_index(['dataset', 'pipeline'])[['elapsed']].unstack().T.droplevel(0)

		time.columns = [DATASET_ABBREVIATION[col] for col in time.columns]
		time.columns = pd.MultiIndex.from_tuples(list(zip(DATASET_FAMILY.values(), time.columns)))

		time['mean'] = time.mean(axis=1)
		time['std'] = time.std(axis=1)

		time.insert(0, 'Pipeline', time.index)
		time = time.reset_index(drop=True)


		time.columns = time.columns.get_level_values(0)
		time = time[['Pipeline', 'mean']]
		time.columns = ['pipeline', version]
		results.append(time.set_index('pipeline'))

	df = pd.concat(results, axis=1)
	df = df.reindex(sorted(df.columns), axis=1)
	df = df.loc[_ORDER]

	fig = plt.figure(figsize=(7, 4.5))
	ax = plt.gca()

	for i, pipeline in enumerate(df.T.columns):
	  ax.plot(df.T[pipeline], marker=_MARKERS[i], markersize=7, color=_PALETTE[i], label=_LABELS[i])

	ax.set_yscale('log')

	plt.grid(True, linestyle='--')
	plt.legend(bbox_to_anchor=(1.05, 0.85), edgecolor='black')
	plt.xticks(size=12, rotation=90)
	plt.yticks(size=12)
	plt.ylabel('Time (s)', size=14)
	plt.xlabel('Version', size=14)
	plt.title('Average Runtime Across Releases', size=18)
	
	_savefig(fig, 'runtime-version', figdir=OUTPUT_PATH)
