# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/losses__numpy.ipynb (unless otherwise specified).

__all__ = ['mae', 'mse', 'rmse', 'mape', 'smape']

# Cell
from math import sqrt

import numpy as np

from .pytorch import divide_no_nan

# Cell
def mae(y: np.ndarray, y_hat: np.ndarray, weights=None):
    """Calculates Mean Absolute Error.

    The mean absolute error

    Parameters
    ----------
    y: numpy array
        actual test values
    y_hat: numpy array
        predicted values
    weights: numpy array
        weights

    Return
    ------
    scalar: MAE
    """
    assert (weights is None) or (np.sum(weights)>0), 'Sum of weights cannot be 0'
    assert (weights is None) or (len(weights)==len(y)), 'Wrong weight dimension'
    mae = np.average(np.abs(y - y_hat), weights=weights)
    return mae

# Cell
def mse(y: np.ndarray, y_hat: np.ndarray, weights=None) -> float:
    """Calculates Mean Squared Error.
    MSE measures the prediction accuracy of a
    forecasting method by calculating the squared deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.

    Parameters
    ----------
    y: numpy array
        actual test values
    y_hat: numpy array
        predicted values
    weights: numpy array
        weights

    Returns
    -------
    scalar: MSE
    """
    assert (weights is None) or (np.sum(weights)>0), 'Sum of weights cannot be 0'
    assert (weights is None) or (len(weights)==len(y)), 'Wrong weight dimension'
    mse = np.average(np.square(y - y_hat), weights=weights)

    return mse

# Cell
def rmse(y: np.ndarray, y_hat: np.ndarray, weights=None) -> float:
    """Calculates Root Mean Squared Error.
    RMSE measures the prediction accuracy of a
    forecasting method by calculating the squared deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.
    Finally the RMSE will be in the same scale
    as the original time series so its comparison with other
    series is possible only if they share a common scale.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values

    Returns
    -------
    scalar: RMSE
    """
    rmse = sqrt(mse(y, y_hat, weights))

    return rmse

# Cell
def mape(y: np.ndarray, y_hat: np.ndarray, weights=None) -> float:
    #TODO: weights no hace nada
    """Calculates Mean Absolute Percentage Error.
    MAPE measures the relative prediction accuracy of a
    forecasting method by calculating the percentual deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values

    Returns
    -------
    scalar: MAPE
    """
    delta_y = np.abs(y - y_hat)
    scale = np.abs(y)
    mape = divide_no_nan(delta_y, scale)
    mape = np.mean(mape)
    mape = 100 * mape

    return mape

# Cell
def smape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculates Symmetric Mean Absolute Percentage Error.

    SMAPE measures the relative prediction accuracy of a
    forecasting method by calculating the relative deviation
    of the prediction and the true value scaled by the sum of the
    absolute values for the prediction and true value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 200% which is desireble compared to normal MAPE that
    may be undetermined.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values

    Returns
    -------
    scalar: SMAPE
    """
    delta_y = np.abs(y - y_hat)
    scale = np.abs(y) + np.abs(y_hat)
    smape = divide_no_nan(delta_y, scale)
    smape = 200 * np.mean(smape)

    assert smape <= 200, 'SMAPE should be lower than 200'

    return smape