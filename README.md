# orionbench-paper

Reproducing Figures & Tables in OrionBench paper. 

## Installation

Experiments were made in **python 3.8**.
To run the benchmark, create a virtual environment and install required packages, then run the following script.

```bash
conda create --name orionbench-env python=3.8
conda activate orionbench-env
pip install -r requirements.txt
```

## Figures

To reproduce figures, refer to `Figures & Tables.ipynb`. All results will be saved to `./output` directory.

## Usage

File `benchmark.csv` contains the latest benchmark results with the following setup.

```python3
import os
from functools import partial

from orion.benchmark import benchmark
from orion.evaluation import CONTEXTUAL_METRICS as METRICS
from orion.evaluation import contextual_confusion_matrix

# directory to store pipelines and intermediate results
pipeline_dir = 'pipelines'
cache_dir = 'cache'

# metrics using overlapping segment
METRICS['confusion_matrix'] = contextual_confusion_matrix
metrics = {k: partial(fun, weighted=False) for k, fun in METRICS.items()}

# run
results = benchmark(
    metrics=metrics, iterations=5, 
    pipeline_dir=pipeline_dir, cache_dir=cache_dir
)
```

> :warning: running the full benchmark requires a long time to compute.

To specify a subset of pipelines and datasets, you need to pass them as arguments

```python3
pipelines = [
    'arima',
    'lstm_dynamic_threshold'
]

signals = ['S-1', 'P-1']

datasets = {
    'NASA-subset': signals
}

results = benchmark(pipelines=pipelines, datasets=datasets)
```

## Resources

Code and benchmark is available in our unsupervised time series anomaly detection library [Orion](https://github.com/sintel-dev/Orion).