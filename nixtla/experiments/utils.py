# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/experiments_utils.ipynb (unless otherwise specified).

__all__ = ['get_mask_df', 'train_val_split', 'prepare_dataset', 'instantiate_nbeats', 'model_fit_predict',
           'evaluate_model', 'hyperopt_tunning']

# Cell
import time
import os
# Limit number of threads in numpy and others to avoid throttling
os.environ["OMP_NUM_THREADS"] = "2" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "2" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "3" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "2" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "3" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import pandas as pd
import argparse
import pickle
import glob
import itertools
import random
from datetime import datetime
from functools import partial

from ..data.scalers import Scaler
from ..data.tsdataset import TimeSeriesDataset
from ..data.tsloader_fast import TimeSeriesLoader
from ..losses.numpy import mae, mape, smape, rmse, pinball_loss

# Models
from ..models.nbeats.nbeats import Nbeats
from ..models.esrnn.esrnn import ESRNN

# import warnings
# warnings.filterwarnings("ignore")

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Cell
def get_mask_df(Y_df, timestamps_in_outsample):
    # Creates outsample_mask
    # train 1 validation 0

    last_df = Y_df.copy()[['unique_id', 'ds']]
    last_df.sort_values(by=['unique_id', 'ds'], inplace=True, ascending=False)
    last_df.reset_index(drop=True, inplace=True)

    last_df = last_df.groupby('unique_id').head(timestamps_in_outsample)
    last_df['sample_mask'] = 0

    last_df = last_df[['unique_id', 'ds', 'sample_mask']]

    mask_df = Y_df.merge(last_df, on=['unique_id', 'ds'], how='left')
    mask_df['sample_mask'] = mask_df['sample_mask'].fillna(1)    # The first len(Y)-n_hours used as train

    mask_df = mask_df[['unique_id', 'ds', 'sample_mask']]
    mask_df.sort_values(by=['unique_id', 'ds'], inplace=True)
    mask_df['available_mask'] = 1

    assert len(mask_df)==len(Y_df), \
        f'The mask_df length {len(mask_df)} is not equal to Y_df length {len(Y_df)}'

    return mask_df

# Cell
def train_val_split(len_series, offset, window_sampling_limit, n_val_weeks, ds_per_day):
    last_ds = len_series - offset
    first_ds = max(last_ds - window_sampling_limit, 0)

    last_day = int(last_ds/ds_per_day)
    first_day = int(first_ds/ds_per_day)

    days = set(range(first_day, last_day)) # All days, to later get train days
    # Sample weeks from here, -7 to avoid sampling from last week
    # To not sample first week and have inputs
    sampling_days = set(range(first_day + 7, last_day - 7))
    validation_days = set({}) # Val days set

    # For loop for n of weeks in validation
    for i in range(n_val_weeks):
        # Sample random day, init of week
        init_day = random.sample(sampling_days, 1)[0]
        # Select days of sampled init of week
        sampled_days = list(range(init_day, min(init_day+7, last_day)))
        # Add days to validation days
        validation_days.update(sampled_days)
        # Remove days from sampling_days, including overlapping resulting previous week
        days_to_remove = set(range(init_day-6, min(init_day+7, last_day)))
        sampling_days = sampling_days.difference(days_to_remove)

    train_days = days.difference(validation_days)

    train_days = sorted(list(train_days))
    validation_days = sorted(list(validation_days))

    train_idx = []
    for day in train_days:
        hours_idx = range(day*ds_per_day,(day+1)*ds_per_day)
        train_idx += hours_idx

    val_idx = []
    for day in validation_days:
        hours_idx = range(day*ds_per_day,(day+1)*ds_per_day)
        val_idx += hours_idx

    assert all([idx < last_ds for idx in val_idx]), 'Leakage!!!!'

    return train_idx, val_idx

# Cell
def prepare_dataset(mc, Y_df, X_df, S_df, timestamps_in_outsample, shuffle_outsample, offset):
    #TODO: offset not implemented
    #TODO: shuffle_outsample

    # n_timestamps_pred defines number of hours ahead to predict
    # offset defines the shift of the data to simulate rolling window
    # assert offset % n_timestamps_pred == 0, 'Avoid overlap of predictions, redefine n_timestamps_pred or offset' <-- restriccion poco general

    #------------------------------------- Available and Validation Mask ------------------------------------#
    # mask: 1 last_n_timestamps, 0 timestamps until last_n_timestamps
    train_mask_df = get_mask_df(Y_df=Y_df, timestamps_in_outsample=timestamps_in_outsample)
    outsample_mask_df = train_mask_df.copy()
    outsample_mask_df['sample_mask'] = 1 - outsample_mask_df['sample_mask']

    #---------------------------------------------- Scale Data ----------------------------------------------#
    # Scale data # TODO: write sample_mask conditional/groupby(['unique_id]) scaling
    Y_df, X_df, scaler_y = scale_data(Y_df=Y_df, X_df=X_df, mask_df=train_mask_df,
                                                    normalizer_y=mc['normalizer_y'], normalizer_x=mc['normalizer_x'])

    #----------------------------------------- Declare Dataset and Loaders ----------------------------------#
    train_ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, S_df=S_df, mask_df=train_mask_df)
    if timestamps_in_outsample == 0:
        outsample_ts_dataset = None
    else:
        outsample_ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, S_df=S_df, mask_df=outsample_mask_df)

    mc['t_cols'] = train_ts_dataset.t_cols
    return mc, train_ts_dataset, outsample_ts_dataset, scaler_y

# Cell
def instantiate_nbeats(mc, train_ts_dataset, outsample_ts_dataset, scaler_y):
    #TODO: window_sampling_limit interaccion con offset
    #TODO: idx_to_sample_freq de validation??
    #TODO: que time_series_loader???
    #TODO: n_hidden e include_var_dict cambiar por parser

    mc['n_hidden'] = len(mc['stack_types']) * [ [int(mc['n_hidden']), int(mc['n_hidden'])] ]

    include_var_dict = {'y': [],
                    'Exogenous1': [],
                    'Exogenous2': [],
                    'week_day': []}

    if mc['incl_pr1']: include_var_dict['y'].append(-2)
    if mc['incl_pr2']: include_var_dict['y'].append(-3)
    if mc['incl_pr3']: include_var_dict['y'].append(-4)
    if mc['incl_pr7']: include_var_dict['y'].append(-8)

    if mc['incl_ex1_0']: include_var_dict['Exogenous1'].append(-1)
    if mc['incl_ex1_1']: include_var_dict['Exogenous1'].append(-2)
    if mc['incl_ex1_7']: include_var_dict['Exogenous1'].append(-8)

    if mc['incl_ex2_0']: include_var_dict['Exogenous2'].append(-1)
    if mc['incl_ex2_1']: include_var_dict['Exogenous2'].append(-2)
    if mc['incl_ex2_7']: include_var_dict['Exogenous2'].append(-8)

    if mc['incl_day']: include_var_dict['week_day'].append(-1)

    mc['include_var_dict'] = include_var_dict

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

    if outsample_ts_dataset is not None:
        val_ts_loader = TimeSeriesLoader(ts_dataset=outsample_ts_dataset,
                                        model='nbeats',
                                        offset=0,
                                        window_sampling_limit=int(mc['window_sampling_limit']),
                                        input_size=int(mc['input_size_multiplier']*mc['output_size']),
                                        output_size=int(mc['output_size']),
                                        idx_to_sample_freq=24,
                                        batch_size=int(mc['batch_size']),
                                        complete_inputs=False,
                                        complete_sample=False,
                                        shuffle=False)
    else:
        val_ts_loader = None

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
                    include_var_dict=mc['include_var_dict'],
                    t_cols=mc['t_cols'],
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
                    #scaler_y=scaler_y,
                    loss=mc['loss'],
                    loss_hypar=float(mc['loss_hypar']),
                    val_loss=mc['val_loss'],
                    frequency=mc['frequency'],
                    seasonality=int(mc['seasonality']),
                    random_seed=int(mc['random_seed']))

    return train_ts_loader, val_ts_loader, model

# Cell
def model_fit_predict(mc, Y_df, X_df, S_df, timestamps_in_outsample, shuffle_outsample, offsets):
    #TODO: pensar si mejorar for loop
    #TODO: no me convence la funcion como esta, hace demasiado, tal vez separar diferente

    X_df = X_df.copy()
    Y_df = Y_df.copy()

    #-------------------------------------- Rolling prediction on outsample --------------------------------------#
    y_true = []
    y_hat = []
    mask = []
    n_splits = len(offsets)
    for split, offset in enumerate(offsets):
        print(10*'-', f'Split {split+1}/{n_splits}', 10*'-')

        #----------------------------------------------- Datasets -----------------------------------------------#
        #TODO: offset verdadero no hecho, hackeado por fuera
        Y_split_df = Y_df.head(len(Y_df) - offset)
        X_split_df = X_df.head(len(X_df) - offset)
        mc, train_ts_dataset, outsample_ts_dataset, scaler_y = prepare_dataset(mc=mc, Y_df=Y_split_df, X_df=X_split_df,
                                                                               S_df=S_df,
                                                                               timestamps_in_outsample=timestamps_in_outsample,
                                                                               shuffle_outsample=shuffle_outsample,
                                                                               offset=offset)

        #--------------------------------------- Instantiate, fit, predict ---------------------------------------#
        train_ts_loader, val_ts_loader, model = instantiate_nbeats(mc=mc, train_ts_dataset=train_ts_dataset,
                                                                   outsample_ts_dataset=outsample_ts_dataset,
                                                                   scaler_y=scaler_y)

        model.fit(train_ts_loader=train_ts_loader, val_ts_loader=val_ts_loader, eval_steps=mc['eval_steps'])

        y_true_split, y_hat_split, mask_split = model.predict(ts_loader=val_ts_loader, eval_mode=True)

        y_true.append(y_true_split)
        y_hat.append(y_hat_split)
        mask.append(mask_split)

    y_true = np.vstack(y_true)
    y_hat = np.vstack(y_hat)
    mask = np.vstack(mask)

    print(f'y_true.shape (#n_windows, #lt) {y_true.shape}')
    print(f'y_hat.shape (#n_windows, #lt) {y_hat.shape}')
    print("\n")

    # Reshape for univariate and panel model compatibility
    n_series = train_ts_loader.ts_dataset.n_series
    n_fcds = len(y_true) // n_series
    y_true = y_true.reshape(n_series, n_fcds, mc['output_size'])
    y_hat = y_hat.reshape(n_series, n_fcds, mc['output_size'])

    print("y_true.shape (#n_series, #n_fcds, #lt) ", y_true.shape)
    print("y_hat.shape (#n_series, #n_fcds, #lt) ", y_hat.shape)
    print("\n")

    meta_data = val_ts_loader.ts_dataset.meta_data

    return y_true, y_hat, mask, meta_data, model

# Cell
def evaluate_model(mc, loss_function, Y_df, X_df, S_df, timestamps_in_outsample, shuffle_outsample, offsets):
    #TODO: mask shape esta mal

    # Make predictions
    start = time.time()
    y_true, y_hat, mask, meta_data, model = model_fit_predict(mc, Y_df, X_df, S_df, timestamps_in_outsample, shuffle_outsample, offsets)
    run_time = time.time() - start
    # Evaluate predictions
    loss = loss_function(y=y_true, y_hat=y_hat) #weights=mask

    result =  {'loss': loss,
                'mc': mc,
                'trajectories': model.trajectories,
                'run_time': run_time,
                'status': STATUS_OK}
    return result

# Cell
def hyperopt_tunning(space, hyperopt_iters, loss_function, Y_df, X_df, S_df, timestamps_in_outsample, shuffle_outsample, offsets, save_trials=False):
    #TODO: mc parser!!!!!!

    trials = Trials()
    fmin_objective = partial(evaluate_model, loss_function=loss_function, Y_df=Y_df, X_df=X_df, S_df=S_df,
                             timestamps_in_outsample=timestamps_in_outsample,
                             shuffle_outsample=shuffle_outsample, offsets=offsets)

    fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=hyperopt_iters, trials=trials, verbose=True)

    return trials