<img src="https://github.com/Nixtla/nixtla/blob/a36f9988575a3e23ed14c8c8fe2b343cdbe5019c/nixtla_logo.png" width="200" height="160"> # **Nixtla** - Machine learning for time series forecasting
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

KIN

```python
# Code
```

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
