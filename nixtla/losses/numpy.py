# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/losses__numpy.ipynb (unless otherwise specified).

__all__ = ['metric_protections', 'mae', 'mse', 'rmse', 'mape', 'smape', 'pinball_loss', 'accuracy_logits']

# Cell
from math import sqrt
import numpy as np
from .pytorch import divide_no_nan

# TODO: Think more efficient way of masking y_mask availability, without indexing (maybe 0s)

# Cell
def metric_protections(y: np.ndarray, y_hat: np.ndarray,
                       y_mask, weights):

    assert (weights is None) or (np.sum(weights)>0), 'Sum of weights cannot be 0'
    assert (weights is None) or (len(weights)==len(y)), 'Wrong weight dimension'
    assert (y_mask is None) or (len(y_mask)==len(y)), 'Wrong mask dimension'

    if y_mask is not None:
        y = y[y_mask]
        y_hat = y_hat[y_mask]
        if weights is not None:
            weights = delta_y[y_mask]

    return y, y_hat, y_mask, weights

# Cell
def mae(y: np.ndarray, y_hat: np.ndarray,
        y_mask=None, weights=None):
    """Calculates Mean Absolute Error.

    The mean absolute error

    Parameters
    ----------
    y: numpy array
        actual test values
    y_hat: numpy array
        predicted values
    y_mask: numpy array
      optional mask, 1 keep 0 omit
    weights: numpy array
      weights for weigted average

    Return
    ------
    scalar: MAE
    """
    y, y_hat, y_mask, weights = metric_protections(y=y, y_hat=y_hat,
                                                   y_mask=y_mask, weights=weights)

    delta_y = np.abs(y - y_hat)
    mae = np.average(np.abs(y - y_hat), weights=weights)
    return mae

# Cell
def mse(y: np.ndarray, y_hat: np.ndarray,
        y_mask=None, weights=None) -> float:
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
    y_mask: numpy array
      optional mask, 1 keep 0 omit
    weights: numpy array
      weights for weigted average

    Returns
    -------
    scalar: MSE
    """
    y, y_hat, y_mask, weights = metric_protections(y=y, y_hat=y_hat,
                                                   y_mask=y_mask, weights=weights)

    mse = np.average(np.square(y - y_hat), weights=weights)
    return mse

# Cell
def rmse(y: np.ndarray, y_hat: np.ndarray,
         y_mask=None, weights=None) -> float:
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
    y_mask: numpy array
      optional mask, 1 keep 0 omit
    weights: numpy array
      weights for weigted average

    Returns
    -------
    scalar: RMSE
    """
    y, y_hat, y_mask, weights = metric_protections(y=y, y_hat=y_hat,
                                                   y_mask=y_mask, weights=weights)

    rmse = sqrt(mse(y, y_hat, weights))

    return rmse

# Cell
def mape(y: np.ndarray, y_hat: np.ndarray,
         y_mask=None, weights=None) -> float:
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
    y_mask: numpy array
      optional mask, 1 keep 0 omit
    weights: numpy array
      weights for weigted average

    Returns
    -------
    scalar: MAPE
    """
    y, y_hat, y_mask, weights = metric_protections(y=y, y_hat=y_hat,
                                                   y_mask=y_mask, weights=weights)

    delta_y = np.abs(y - y_hat)
    scale = np.abs(y)
    mape = divide_no_nan(delta_y, scale)
    mape = np.average(mape, weights=weights)
    mape = 100 * mape

    return mape

# Cell
def smape(y: np.ndarray, y_hat: np.ndarray,
          y_mask=None, weights=None) -> float:
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
    y_mask: numpy array
      optional mask, 1 keep 0 omit
    weights: numpy array
      weights for weigted average

    Returns
    -------
    scalar: SMAPE
    """
    y, y_hat, y_mask, weights = metric_protections(y=y, y_hat=y_hat,
                                                   y_mask=y_mask, weights=weights)

    delta_y = np.abs(y - y_hat)
    scale = np.abs(y) + np.abs(y_hat)
    smape = divide_no_nan(delta_y, scale)
    smape = 200 * np.average(smape, weights=weights)

    assert smape <= 200, 'SMAPE should be lower than 200'

    return smape

# Cell
def pinball_loss(y: np.ndarray, y_hat: np.ndarray, tau: float=0.5,
                 y_mask=None, weights=None) -> np.ndarray:
    """Calculates the Pinball Loss.

    The Pinball loss measures the deviation of a quantile forecast.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.
    A common value for tau is 0.5 for the deviation from the median.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array of len h (forecasting horizon)
      predicted values
    y_mask: numpy array
      optional mask, 1 keep 0 omit
    weights: numpy array
      weights for weigted average
    tau: float
      Fixes the quantile against which the predictions are compared.
    Return
    ------
    return: pinball_loss
    """
    y, y_hat, y_mask, weights = metric_protections(y=y, y_hat=y_hat,
                                                   y_mask=y_mask, weights=weights)

    delta_y = y - y_hat
    pinball = np.maximum(tau * delta_y, (tau - 1) * delta_y)
    pinball = np.average(pinball, weights=weights) #pinball.mean()
    return pinball

# Cell
def accuracy_logits(y: np.ndarray, y_hat: np.ndarray, weights=None, thr=0.5) -> np.ndarray:
    """Calculates the Accuracy.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array of len h (forecasting horizon)
      predicted values
    weights: numpy array
      weights for weigted average
    Return
    ------
    return accuracy
    """
    y, y_hat, y_mask, weights = metric_protections(y=y, y_hat=y_hat,
                                                   y_mask=y_mask, weights=weights)

    y_hat = ((1/(1 + np.exp(-y_hat))) > thr) * 1
    accuracy = np.average(y_hat==y, weights=weights) * 100
    return accuracy