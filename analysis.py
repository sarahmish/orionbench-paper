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

DATASET_ABBREVIATION = {
    "MSL": "MSL",
    "SMAP": "SMAP",
    "natural": "Natural",
    "DISTORTED": "Distorted",
    "NOISE": "Noise",
    "UCR": "UCR",
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

DATASET_FAMILY = {
    "MSL": "NASA",
    "SMAP": "NASA",
    "natural": "UCR",
    "DISTORTED": "UCR",
    "NOISE": "UCR",
    "UCR": "UCR",
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

OUTPUT_PATH = Path('output')

BUCKET = 'sintel-orion'
S3_URL = 'https://{}.s3.amazonaws.com/{}'
GITHUB_URL = 'https://raw.githubusercontent.com/sintel-dev/Orion/master/benchmark/results/{}.csv'

DATA_MAP = {signal: data for data, signals in BENCHMARK_DATA.items() for signal in signals}

_VERSION = ['0.1.3', '0.1.4', '0.1.5', '0.1.6', '0.1.7', '0.2.0', '0.2.1', '0.3.0', '0.3.1', '0.3.2', '0.4.0', '0.4.1', '0.5.0', '0.5.1', '0.5.2']
_ORDER = ['aer', 'lstm_dynamic_threshold', 'arima', 'matrixprofile', 'lstm_autoencoder', 'tadgan', 'vae', 'dense_autoencoder', 'ganf', 'lnn', 'azure', 'anomaly_transformer']
_LABELS = ['AER', 'LSTM DT', 'ARIMA', 'MP', 'LSTM AE', 'TadGAN', 'VAE', 'Dense AE', 'GANF', 'LNN', 'Azure AD', 'AT']
_COLORS = ['#9B2226', '#AE2012', '#BB3E03', '#CA6702', '#EE9B00', '#E9D8A6', '#BFD5B2', '#94D2BD', '#0A9396', '#005F73']
_NEW_COLORS = ['#9B2226', '#AE2012', '#BB3E03', '#CA6702', '#EE9B00', '#E0BE67', '#E9D8A6', '#BFD5B2', '#94D2BD', '#0A9396', '#005F73', '#001219']
_MARKERS = ['o', 's', 'v', 'X', 'p', '^', 'd', 'P', '>', '<', 'H']
_PALETTE = _NEW_COLORS


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

def _compute_f1(df, iteration=True):
    columns = ['dataset', 'pipeline', 'iteration']
    if not iteration:
        columns = ['dataset', 'pipeline']

    df = df.groupby(columns)[['fp', 'fn', 'tp']].sum().reset_index()

    df['precision'] = df.eval('tp / (tp + fp)')
    df['recall'] = df.eval('tp / (tp + fn)')
    df['f1'] = df.eval('2 * (precision * recall) / (precision + recall)')

    df['family'] = df['dataset'].apply(lambda x: DATASET_FAMILY[x])
    df['dataset'] = df['dataset'].apply(lambda x: DATASET_ABBREVIATION[x])

    return df

def _get_f1_scores(results, iteration=True):
    df = _compute_f1(results, iteration=iteration)
    df = df.set_index(['dataset', 'pipeline'])[['f1']].unstack().T.droplevel(0)

    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)

    df.insert(0, 'Pipeline', df.index)
    df = df.reset_index(drop=True)

    return df

# ------------------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------------------

def table_data_summary():
    data = pd.read_csv('data_summary.csv')
    data['anomaly_len'] = data['anomaly_len'].apply(ast.literal_eval)
    summary = data.groupby('dataset')['count'].agg(['count', 'sum'])
    signals = data.groupby('dataset')['signal_len'].mean()
    anomaly = data.groupby('dataset')['anomaly_len'].apply(lambda x: np.mean(list(itertools.chain.from_iterable(x))))
    summary = pd.concat([summary, signals, anomaly], axis=1)
    return summary

def table_leaderboard():
    def _get_wins(df):
        df = df.groupby(['dataset', 'pipeline'])[['fp', 'fn', 'tp']].sum().reset_index()
        precision = df['tp'] / (df['tp'] + df['fp'])
        recall = df['tp'] / (df['tp'] + df['fn'])
        df['f1'] = 2 * (precision * recall) / (precision + recall)

        summary = dict()

        # number of wins over arima
        arima_pipeline = 'arima'
        intermediate = df.set_index(['pipeline', 'dataset'])['f1'].unstack().T
        arima = intermediate.pop(arima_pipeline)

        summary['wins'] = (intermediate.T > arima).sum(axis=1)
        summary['f1'] = intermediate.mean()
        summary['wins'][arima_pipeline] = None
                                                     

        summary = pd.DataFrame(summary)
        summary.index.name = 'Pipeline'

        return summary.reset_index().sort_values(['wins', 'f1'], ascending=False)

    df = pd.read_csv('benchmark.csv')

    wins = {}
    for i in range(5):
        sub = df[df['iteration'] == i]
        score = _get_wins(sub)
        score = score.reset_index(drop=True)
        score.index.name = 'rank'
        score.reset_index(drop=False, inplace=True)
        score['rank'] += 1
        wins[i] = score
        
    wins = pd.concat([x.set_index('Pipeline').loc[_ORDER]['wins'] for x in wins.values()], axis=1).median(axis=1)
    wins.index = _LABELS
    wins.drop('ARIMA', inplace=True)
    wins.sort_values(ascending=False, inplace=True)
    wins = wins.reset_index()
    wins.columns = ['Pipeline', 'Wins']
    return wins

def table_performance():
    def _format_table(df, metric):
        df = df.groupby(['dataset', 'family', 'pipeline'])[[metric]].agg(["mean", "std"]).droplevel(0, axis=1)
        df['value'] = df["mean"].round(3).astype("str") + "+-" + df["std"].round(2).astype("str")
        df = df[['value']].unstack().T.droplevel(0)
        df = df[['MSL', 'SMAP', 'Natural', 'Distorted', 'Noise', 'A1', 'A2', 'A3', 'A4', 'Art', 'AWS', 'AdEx', 'Traf', 'Tweets']]
        df = df.swaplevel(axis=1).loc[_ORDER]
        df.index = _LABELS
        df.name = metric.title()
        return df

    df = pd.read_csv('benchmark.csv')
    df = _compute_f1(df)

    f1 = _format_table(df, metric='f1')
    pre = _format_table(df, metric='precision')
    rec = _format_table(df, metric='recall')
    
    return f1, pre, rec


# ------------------------------------------------------------------------------
# Main Figures
# ------------------------------------------------------------------------------

def figure_f1_boxplot():
    df = pd.read_csv('benchmark.csv')
    df = _compute_f1(df)
    df = df.set_index(['dataset', 'pipeline', 'iteration'])[['f1', 'family']].reset_index()

    fig, axes = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(13, 3))
    fig.subplots_adjust(wspace=0)
    axes = axes.flatten()

    data = df.set_index('pipeline').loc[_ORDER].reset_index()

    sns.boxplot(data[data['family'] == 'NASA'], x='pipeline', y='f1', palette=_PALETTE, ax=axes[0])
    sns.boxplot(data[data['family'] == 'NAB'], x='pipeline', y='f1', palette=_PALETTE, ax=axes[1])
    sns.boxplot(data[data['family'] == 'UCR'], x='pipeline', y='f1', palette=_PALETTE, ax=axes[2])
    sns.boxplot(data[data['dataset'].isin(['A1', 'A2'])], x='pipeline', y='f1', palette=_PALETTE, ax=axes[3])
    sns.boxplot(data[data['dataset'].isin(['A3', 'A4'])], x='pipeline', y='f1', palette=_PALETTE, ax=axes[4])

    for i in range(5):
        axes[i].grid(True, linestyle='--')
        axes[i].set_xticklabels(_LABELS, rotation=90)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    # axes[2].get_xaxis().set_visible(False)
    # axes[2].get_yaxis().set_visible(False)

    axes[0].set_xlim(-1, 12)
    axes[0].set_ylabel('F1 Score')
    
    axes[0].set_title("NASA")
    axes[1].set_title("NAB")
    axes[2].set_title("UCR")
    axes[3].set_title("Yahoo S5 (A1 & A2)")
    axes[4].set_title("Yahoo S5 (A3 & A4)")

    # handles = [
    #      mpatches.Patch(color=_PALETTE[i], label=_LABELS[i])
    #      for i in range(len(_LABELS))
    # ]

    # plt.legend(handles=handles, bbox_to_anchor=(0.98, 2.1), frameon=False, ncol=2)
    
    _savefig(fig, 'figure_4', figdir=OUTPUT_PATH)


def figure_runtime():
    BINS = ['<= 10,000', '> 10,000, <= 100,000', '> 100,000']

    df = pd.read_csv('benchmark.csv')
    signal_meta = pd.read_csv('data_summary.csv').set_index('signal')['signal_len'].to_dict()

    df['elapsed'] = df['elapsed'] / 60
    df['signal_len'] = df['signal'].apply(lambda x: signal_meta[x])
    df['bin'] = pd.cut(df['signal_len'], [0, 10000, 100000, 1000000], labels=BINS)
    df['family'] = df['dataset'].apply(lambda x: DATASET_FAMILY[x])
    df['dataset'] = df['dataset'].apply(lambda x: DATASET_ABBREVIATION[x])
    df = df.set_index('pipeline').loc[_ORDER].reset_index()

    fig = plt.figure(figsize=(6, 3))
    ax = plt.gca()
        
    sns.barplot(data=df, x="bin", y="elapsed", hue='pipeline', palette=_PALETTE,
                saturation=0.7, linewidth=1, edgecolor='k', ax=ax)

#     ax.set_xticklabels(_LABELS, rotation=90)
    handles = [
         mpatches.Patch(color=_PALETTE[i], label=_LABELS[i])
         for i in range(len(_LABELS))
    ]
    
    plt.grid(True, linestyle='--')
    plt.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.31, 1.05), edgecolor='black')
    plt.yscale('log')
#     plt.ylim([0.1e1, 0.2e5])
    plt.ylabel('Time in Minutes (log)')
    plt.xlabel('Signal Length')
    plt.title("Pipeline Elapsed Time")

    _savefig(fig, 'figure_5', figdir=OUTPUT_PATH)


def figure_f1_progression():
    def get_value(x):
        if x == "0":
            return None, 0, 0, 0

        values = ast.literal_eval(x)
        return values

    results = []
    for version in _VERSION:
        df = pd.read_csv(GITHUB_URL.format(version))
    
        try:
            scores = _get_f1_scores(df, iteration=False)
        except KeyError as e:
            df['confusion_matrix'] = df['confusion_matrix'].apply(get_value)
            df[['tn', 'fp', 'fn', 'tp']] = pd.DataFrame(df['confusion_matrix'].tolist(), index=df.index)
            scores = _get_f1_scores(df, iteration=False)

        scores.columns = scores.columns.get_level_values(0)
        scores = scores[['Pipeline', 'mean']]
        scores.columns = ['pipeline', version]
        results.append(scores.set_index('pipeline'))

    labels = ['AER', 'LSTM DT', 'ARIMA', 'MP', 'LSTM AE', 'TadGAN', 'VAE', 'Dense AE', 'GANF', 'Azure AD']
    order = ['aer', 'lstm_dynamic_threshold', 'arima', 'matrixprofile', 'lstm_autoencoder', 'tadgan', 'vae', 'dense_autoencoder', 'ganf', 'azure']

    df = pd.concat(results, axis=1)
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.loc[order]

    fig = plt.figure(figsize=(5, 2.5))
    ax = plt.gca()

    for i, pipeline in enumerate(df.T.columns):
        if pipeline not in ['aer', 'lstm_dynamic_threshold', 'arima', 'azure']:
            continue
        
        ax.plot(df.T[pipeline], color=_COLORS[i], label=labels[i], lw=4)

    plt.grid(True, linestyle='--')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), edgecolor='black', ncol=4, fontsize='small', fancybox=True)
    plt.ylim([0.15, 0.825])
    plt.xticks(rotation=90)
    plt.ylabel('F1 Score')
    plt.xlabel('Version')
    plt.title('Pipeline Performance Across Releases\n\n')
    
    _savefig(fig, 'figure_5b', figdir=OUTPUT_PATH)


# ------------------------------------------------------------------------------
# Supplementary Figures
# ------------------------------------------------------------------------------


def figure_f1_details():
    df = pd.read_csv('benchmark.csv')
    df = _compute_f1(df)

    fig, axes = plt.subplots(4, 5, sharex=True, sharey=True, figsize=(14, 11))
    fig.subplots_adjust(wspace=0)
    axes = axes.flatten()
    skip = [2, 3, 4, 8, 9, 14]
    for i in skip:
        fig.delaxes(axes[i])

    data = df.set_index('pipeline').loc[_ORDER].reset_index()

    i = 0
    for d in DATASET_ABBREVIATION.values():
        if d == "UCR":
            continue
            
        while i in skip:
            i += 1
            
        sns.boxplot(data[data['dataset'] == d], x='pipeline', y='f1', palette=_PALETTE, ax=axes[i])
        axes[i].grid(True, linestyle='--')
        axes[i].set_xticklabels(_LABELS, rotation=90)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].set_title(d)
        i += 1
            
    axes[0].set_xlim(-1, 11)
    axes[0].set_ylabel('F1 Score')
    axes[5].set_ylabel('F1 Score')
    axes[10].set_ylabel('F1 Score')

    # fig.suptitle('F1 Scores per Dataset')

    handles = [
         mpatches.Patch(color=_PALETTE[i], label=_LABELS[i])
         for i in range(len(_LABELS))
    ]

    plt.legend(handles=handles, bbox_to_anchor=(0.8, 2.9), edgecolor='black')
    _savefig(fig, 'figure_9', figdir=OUTPUT_PATH)


def figure_runtime_details():
    df = pd.read_csv('benchmark.csv')
    df['family'] = df['dataset'].apply(lambda x: DATASET_FAMILY[x])
    df['dataset'] = df['dataset'].apply(lambda x: DATASET_ABBREVIATION[x])
    df = df.set_index('pipeline').loc[_ORDER].reset_index()

    fig, axes = plt.subplots(4, 5, sharex=True, sharey=True, figsize=(14, 11))
    fig.subplots_adjust(wspace=0)
    axes = axes.flatten()
    skip = [2, 3, 4, 8, 9, 14]
    for i in skip:
        fig.delaxes(axes[i])

    data = df.set_index('pipeline').loc[_ORDER].reset_index()

    i = 0
    for d in DATASET_ABBREVIATION.values():
        if d == "UCR":
            continue
            
        while i in skip:
            i += 1
            
        sns.boxplot(data[data['dataset'] == d], x='pipeline', y='elapsed', palette=_PALETTE, ax=axes[i])
        axes[i].grid(True, linestyle='--')
        axes[i].set_xticklabels(_LABELS, rotation=90)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].set_title(d)
        axes[i].set_yscale('log')
        i += 1
        
    axes[0].set_xlim(-1, 11)
    axes[0].set_ylabel('Time (s)')
    axes[5].set_ylabel('Time (s)')
    axes[10].set_ylabel('Time (s)')

    # fig.suptitle('Runtime per Dataset')

    handles = [
         mpatches.Patch(color=_PALETTE[i], label=_LABELS[i])
         for i in range(len(_LABELS))
    ]

    plt.legend(handles=handles, bbox_to_anchor=(0.8, 2.9), edgecolor='black');
    _savefig(fig, 'figure_10', figdir=OUTPUT_PATH)


def figure_runtime_progression():
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

    labels = ['AER', 'LSTM DT', 'ARIMA', 'MP', 'LSTM AE', 'TadGAN', 'VAE', 'Dense AE', 'GANF', 'Azure AD']
    order = ['aer', 'lstm_dynamic_threshold', 'arima', 'matrixprofile', 'lstm_autoencoder', 'tadgan', 'vae', 'dense_autoencoder', 'ganf', 'azure']
    
    df = pd.concat(results, axis=1)
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.loc[order]

    fig = plt.figure(figsize=(5, 3))
    ax = plt.gca()

    for i, pipeline in enumerate(df.T.columns):
        ax.plot(df.T[pipeline], marker=_MARKERS[i], markersize=7, color=_COLORS[i], label=labels[i])

    ax.set_yscale('log')

    plt.grid(True, linestyle='--')
    plt.legend(bbox_to_anchor=(1.01, 0.93), edgecolor='black')
    plt.xticks(rotation=90)
    plt.ylabel('Time (s)')
    plt.xlabel('Version')
    plt.title('Average Runtime Across Releases')
    
    _savefig(fig, 'figure_11', figdir=OUTPUT_PATH)
