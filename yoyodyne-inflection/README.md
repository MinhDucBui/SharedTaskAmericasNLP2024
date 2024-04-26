THIS IS A FORK FROM [Yoyodyne](https://github.com/CUNY-CL/yoyodyne).

# Yoyodyne 🪀

[![PyPI
version](https://badge.fury.io/py/yoyodyne.svg)](https://pypi.org/project/yoyodyne)
[![Supported Python
versions](https://img.shields.io/pypi/pyversions/yoyodyne.svg)](https://pypi.org/project/yoyodyne)
[![CircleCI](https://circleci.com/gh/CUNY-CL/yoyodyne/tree/master.svg?style=svg&circle-token=37883deeb03d32c8a7b2aa7c34e5143bf514acdd?)](https://circleci.com/gh/CUNY-CL/yoyodyne/tree/master)

Yoyodyne provides neural models for small-vocabulary sequence-to-sequence
generation with and without feature conditioning.

These models are implemented using [PyTorch](https://pytorch.org/) and
[Lightning](https://www.pytorchlightning.ai/).

While we provide classic LSTM and transformer models, some of the provided
models are particularly well-suited for problems where the source-target
alignments are roughly monotonic (e.g., `transducer`) and/or where source and
target vocabularies have substantial overlap (e.g., `pointer_generator_lstm`).

## Installation

### Local installation

Yoyodyne currently supports Python 3.9 and 3.10.
[#60](https://github.com/CUNY-CL/yoyodyne/issues/60) is a known blocker to
Python \> 3.10 support.

First install dependencies:

    pip install -r requirements.txt

Then install:

    pip install .

It can then be imported like a regular Python module:

```python
import yoyodyne
```

## Usage

### Training

Training is performed by the [`yoyodyne-train`](yoyodyne/train.py) script. One
must specify the following required arguments:

-   `--model_dir`: path for model metadata and checkpoints
-   `--experiment`: name of experiment (pick something unique)
-   `--train`: path to TSV file containing training data
-   `--val`: path to TSV file containing validation data

The user can also specify various optional training and architectural arguments.
See below or run [`yoyodyne-train --help`](yoyodyne/train.py) for more
information.

### Prediction

Prediction is performed by the [`yoyodyne-predict`](yoyodyne/predict.py) script.
One must specify the following required arguments:

-   `--model_dir`: path for model metadata
-   `--experiment`: name of experiment
-   `--checkpoint`: path to checkpoint
-   `--predict`: path to file containing data to be predicted
-   `--output`: path for predictions

The `--predict` file can either be a TSV file or an ordinary TXT file with one
source string per line; in the latter case, specify `--target_col 0`. Run
[`yoyodyne-predict --help`](yoyodyne/predict.py) for more information.

Conference of the North American Chapter of the Association for Computational
Linguistics (Demonstrations)*, pages 48-53.

Smith, L. N. 2017. [Cyclical learning rates for training neural
networks](https://ieeexplore.ieee.org/abstract/document/7926641). In *2017 IEEE
Winter Conference on Applications of Computer Vision*, pages 464-472.
