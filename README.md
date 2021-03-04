<img src="https://github.com/Nixtla/nixtla/blob/a36f9988575a3e23ed14c8c8fe2b343cdbe5019c/nixtla_logo.png" width="200" height="160">

# **Nixtla** - Machine Learning for Time Series Forecasting
> [nikstla] (noun, nahuatl) Period of time.


[![Build](https://github.com/kdgutier/esrnn_torch/workflows/Python%20package/badge.svg?branch=master)](https://github.com/kdgutier/esrnn_torch/tree/master)
[![PyPI version fury.io](https://badge.fury.io/py/ESRNN.svg)](https://pypi.python.org/pypi/ESRNN/)
[![Downloads](https://pepy.tech/badge/esrnn)](https://pepy.tech/project/esrnn)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/kdgutier/esrnn_torch/blob/master/LICENSE)

**Nixtla** is a **forecasting library** for **state of the art** statistical and **machine learning models**, currently hosting the [ESRNN](https://www.sciencedirect.com/science/article/pii/S0169207019301153), and the [NBEATSX](https://arxiv.org/abs/1905.10437) models.

Nixtla aims to store tools and models with the capacity to provide **highly accurate forecasts** with **comparable** performance to that of [Amazon Web Services](https://aws.amazon.com/es/forecast/), [Azure](https://docs.microsoft.com/en-us/azure/machine-learning/), or [DataRobot](https://www.datarobot.com/platform/automated-time-series/).

* [Documentation (stable version)]()
* [Documentation (latest)]()
* [JMLR MLOSS Paper]()
* [ArXiv Paper]()

## Installation

### Stable version

This code is a work in progress, any contributions or issues are welcome on
GitHub at: https://github.com/Nixtla/nixtla

You can install the *released version* of `Nixtla` from the [Python package index](https://pypi.org) with:

```python
pip install nixtla
```

(installing inside a python virtualenvironment or a conda environment is warmly recommended).

### Development version

You may want to test the current development version; follow the steps below in that case (clone the git repository and install the Python requirements):
```bash
git clone https://github.com/Nixtla/nixtla.git
cd nixtla
python setup.py bdist_wheel
cd ../
pip install nixtla/dist/nixtla-XX.whl
```
where XX is the latest version downloaded.

#### Development version in development mode

If you want to make some modifications to the code and see the effects in real time (without reinstalling), follow the steps below:

```bash
git clone https://github.com/Nixtla/nixtla.git
cd nixtla
pip install -e .
```

## Currently available models

* [Exponential Smoothing Recurrent Neural Network (ESRNN)](https://www.sciencedirect.com/science/article/pii/S0169207019301153), a hybrid model that combines the expressivity of non linear models to capture the trends while it normalizes using a Holt-Winters inspired model for the levels and seasonals.  This model is the winner of the M4 forecasting competition.

<!--- ![ESRNN](/indx_imgs/ESRNN.png) -->

* [Neural Basis Expansion Analysis with Exogenous Variables (NBEATSX)](https://arxiv.org/abs/1905.10437) is a model from Element-AI (Yoshua Bengio’s lab) that has proven to achieve state of the art performance on benchmark large scale forecasting datasets like Tourism, M3, and M4. The model is fast to train an has an interpretable configuration.

<!--- ![NBEATSX](/indx_imgs/NBEATSX.png) -->

## Usage

Here we show a usage example, where we load data, instantiate loaders and model, train and make predictions.


Here load daily electricity prices from the Nord Pool dataset.
```python
import pandas as pd
from nixtla.data.datasets.epf import EPF, EPFInfo
from nixtla.data.tsloader_general import TimeSeriesLoader
from nixtla.experiments.utils import create_datasets

import pylab as plt
from pylab import rcParams
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'serif'

Y_df, X_df, S_df = EPF.load_groups(directory='../data', groups=['NP'])

fig = plt.figure(figsize=(15, 6))
plt.plot(Y_df.ds, Y_df.y.values, color='#628793', linewidth=0.4)
plt.ylabel('Price [EUR/MWh]', fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.show()
```


Here we declare model and data hyperparameters.
```python
# Architecture parameters
mc = {}
mc['model'] = 'nbeatsx'
mc['input_size_multiplier'] = 1
mc['output_size'] = 24*7
mc['stack_types'] = ['trend', 'seasonality', 'exogenous_wavenet']
mc['activation'] = 'selu'
mc['n_blocks'] = [1, 1, 1]
mc['n_layers'] = [2, 2, 2]
mc['n_hidden'] = 256
mc['exogenous_n_channels'] = 20
mc['x_s_n_hidden'] = 0
mc['shared_weights'] = False
mc['n_harmonics'] = 2
mc['n_polynomials'] = 10

# Optimization and regularization parameters
mc['initialization'] = 'lecun_normal'
mc['learning_rate'] = 0.001
mc['batch_size'] = 512
mc['lr_decay'] = 0.5
mc['n_lr_decay_steps'] = 3
mc['n_iterations'] = 3000
mc['early_stopping'] = 10
mc['eval_freq'] = 200
mc['batch_normalization'] = False
mc['dropout_prob_theta'] = 0
mc['dropout_prob_exogenous'] = 0
mc['l1_theta'] = 0
mc['weight_decay'] = 0.00005
mc['loss'] = 'MAE'
mc['loss_hypar'] = 0.5
mc['val_loss'] = mc['loss']
mc['random_seed'] = 1

# Data Parameters
mc['idx_to_sample_freq'] = 1
mc['n_val_weeks'] = 52
mc['window_sampling_limit'] = 500_000
mc['normalizer_y'] = None
mc['normalizer_x'] = 'median'
mc['complete_inputs'] = False
mc['frequency'] = 'H'
mc['seasonality'] = 24

print(65*'=')
print(pd.Series(mc))
print(65*'=')

mc['n_hidden'] = len(mc['stack_types']) * [ [int(mc['n_hidden']), int(mc['n_hidden'])] ]
```

Here we instantiate the model and dataloaders.

```python
train_ts_dataset, outsample_ts_dataset, scaler_y = create_datasets(mc, Y_df, X_df, S_df, 728*24, False, 0)

train_ts_loader = TimeSeriesLoader(ts_dataset=train_ts_dataset,
                                    model='nbeats',
                                    offset=0,
                                    window_sampling_limit=int(mc['window_sampling_limit']),
                                    input_size=int(mc['input_size_multiplier']*mc['output_size']),
                                    output_size=int(mc['output_size']),
                                    idx_to_sample_freq=int(mc['idx_to_sample_freq']),
                                    batch_size=int(mc['batch_size']),
                                    complete_inputs=mc['complete_inputs'],
                                    complete_sample=False,
                                    shuffle=True)

val_ts_loader = TimeSeriesLoader(#ts_dataset=outsample_ts_dataset,
                                 ts_dataset=train_ts_dataset,
                                 model='nbeats',
                                 offset=0,
                                 window_sampling_limit=int(mc['window_sampling_limit']),
                                 input_size=int(mc['input_size_multiplier']*mc['output_size']),
                                 output_size=int(mc['output_size']),
                                 idx_to_sample_freq=24*7,
                                 batch_size=int(mc['batch_size']),
                                 complete_inputs=False,
                                 complete_sample=False,
                                 shuffle=False)

model = Nbeats(input_size_multiplier=mc['input_size_multiplier'],
                output_size=int(mc['output_size']),
                shared_weights=mc['shared_weights'],
                initialization=mc['initialization'],
                activation=mc['activation'],
                stack_types=mc['stack_types'],
                n_blocks=mc['n_blocks'],
                n_layers=mc['n_layers'],
                n_hidden=mc['n_hidden'],
                n_harmonics=int(mc['n_harmonics']),
                n_polynomials=int(mc['n_polynomials']),
                x_s_n_hidden=int(mc['x_s_n_hidden']),
                exogenous_n_channels=int(mc['exogenous_n_channels']),
                batch_normalization = mc['batch_normalization'],
                dropout_prob_theta=mc['dropout_prob_theta'],
                dropout_prob_exogenous=mc['dropout_prob_exogenous'],
                learning_rate=float(mc['learning_rate']),
                lr_decay=float(mc['lr_decay']),
                n_lr_decay_steps=float(mc['n_lr_decay_steps']),
                weight_decay=mc['weight_decay'],
                l1_theta=mc['l1_theta'],
                n_iterations=int(mc['n_iterations']),
                early_stopping=int(mc['early_stopping']),
                loss=mc['loss'],
                loss_hypar=float(mc['loss_hypar']),
                val_loss=mc['val_loss'],
                frequency=mc['frequency'],
                seasonality=int(mc['seasonality']),
                random_seed=int(mc['random_seed']))
```

```python
model.fit(train_ts_loader=train_ts_loader, val_ts_loader=val_ts_loader, eval_freq=mc['eval_freq'])
```


```python
y_true, y_hat, _ = model.predict(ts_loader=val_ts_loader, return_decomposition=False)

# Date stamps
window_pred_idx = 20 #25
meta_data = val_ts_loader.ts_dataset.meta_data
input_size = mc['input_size_multiplier'] * mc['output_size']
x_outsample = meta_data[0]['last_ds'].values[input_size:]
start = window_pred_idx*mc['output_size']
end   = (window_pred_idx+1)*mc['output_size']
x_plot = x_outsample[start:end]

fig = plt.figure(figsize=(10, 4))
plt.plot(x_plot, y_true[0, window_pred_idx, :])
plt.plot(x_plot, y_hat[0, window_pred_idx, :])
plt.ylabel('Price [EUR/MWh]', fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.show()
```

<img src="https://github.com/Nixtla/nixtla/blob/98471b1cf43ed294f83ec0796b8d28ac71db2ac8/nbs/results/NP_predictions.png" width="800" height="480">


## Authors
This repository was developed with joint efforts from AutonLab researchers at Carnegie Mellon University and Abraxas data scientists.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/kdgutier/esrnn_torch/blob/master/LICENSE) file for details.

## How to cite

If you use `Nixtla` in a scientific publication, we encourage you to add
the following references to the related papers:


```bibtex
@article{nixtla_arxiv,
  author  = {XXXX},
  title   = {{Nixtla: Machine learning for time series forecasting}},
  journal = {arXiv preprint arXiv:XXX.XXX},
  year    = {2021}
}
```

<!---

## Citing

```bibtex
@article{,
    author = {},
    title = {{}},
    journal = {},
    year = {}
}
```
-->
