# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/experiments__utils.ipynb (unless otherwise specified).

__all__ = ['ENV_VARS', 'get_default_mask_df', 'get_train_val_mask_df', 'scale_data', 'train_val_split',
           'create_datasets', 'instantiate_loaders', 'instantiate_nbeats', 'instantiate_esrnn', 'instantiate_mqesrnn',
           'instantiate_rnn', 'instantiate_tcn', 'instantiate_model', 'model_fit_predict', 'evaluate_model',
           'hyperopt_tunning']

# Cell
ENV_VARS = dict(OMP_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2',
                MKL_NUM_THREADS='3',
                VECLIB_MAXIMUM_THREADS='2',
                NUMEXPR_NUM_THREADS='3')

# Cell
import os
# Limit number of threads in numpy and others to avoid throttling
os.environ.update(ENV_VARS)
import random
import time
from functools import partial

import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from ..data.scalers import Scaler
from ..data.tsdataset import TimeSeriesDataset
from ..data.tsloader_general import TimeSeriesLoader
from ..models.esrnn.esrnn import ESRNN
from ..models.esrnn.mqesrnn import MQESRNN
from ..models.esrnn.rnn import RNN
from ..models.nbeats.nbeats import Nbeats
from ..models.tcn.tcn import TCN

# Cell
def get_default_mask_df(Y_df, ds_in_test, is_test):
    # Creates outsample_mask
    # train 1 validation 0
    last_df = Y_df.copy()[['unique_id', 'ds']]
    last_df.sort_values(by=['unique_id', 'ds'], inplace=True, ascending=False)
    last_df.reset_index(drop=True, inplace=True)

    last_df = last_df.groupby('unique_id').head(ds_in_test)
    last_df['sample_mask'] = 0

    last_df = last_df[['unique_id', 'ds', 'sample_mask']]

    mask_df = Y_df.merge(last_df, on=['unique_id', 'ds'], how='left')
    mask_df['sample_mask'] = mask_df['sample_mask'].fillna(1)

    mask_df = mask_df[['unique_id', 'ds', 'sample_mask']]
    mask_df.sort_values(by=['unique_id', 'ds'], inplace=True)
    mask_df['available_mask'] = 1

    assert len(mask_df)==len(Y_df), \
        f'The mask_df length {len(mask_df)} is not equal to Y_df length {len(Y_df)}'

    if is_test:
        mask_df['sample_mask'] = 1 - mask_df['sample_mask']

    return mask_df

# Cell
def get_train_val_mask_df(Y_df, ds_in_test, n_val, n_uids, periods, freq):
    """
    Parameters
    ----------
    ds_in_test: int
        Number of ds in test.
    n_val: int
        Number of windows for validation.
    periods: int
        ds_in_test multiplier
    """
    # Creates outsample_mask
    # train 1 validation 0
    mask_df = Y_df.copy()[['unique_id', 'ds']]
    mask_df.sort_values(by=['unique_id', 'ds'], inplace=True)
    mask_df.reset_index(drop=True, inplace=True)

    mask_df['sample_mask'] = 1
    mask_df['available_mask'] = 1

    idx_test = mask_df.groupby('unique_id').tail(ds_in_test).index
    mask_df.loc[idx_test, 'sample_mask'] = 0

    val_mask_df = mask_df.copy()
    val_mask_df['sample_mask'] = 0

    assert len(mask_df)==len(Y_df), \
        f'The mask_df length {len(mask_df)} is not equal to Y_df length {len(Y_df)}'

    uids = mask_df['unique_id'].unique()
    val_uids = np.random.choice(uids, n_uids, replace=False)

    min_ds = mask_df['ds'].min()
    available_ds = mask_df.query('ds >= @min_ds') \
                          .loc[~mask_df.index.isin(idx_test)]['ds'] \
                          .unique()
    val_init_ds = np.random.choice(available_ds, n_val, replace=False)

    val_ds = [pd.date_range(init, periods=periods * ds_in_test, freq=freq) for init in val_init_ds]
    val_ds = np.concatenate(val_ds)

    val_idx = mask_df.query('unique_id in @val_uids & ds in @val_ds').index
    mask_df.loc[val_idx, 'sample_mask'] = 0
    val_mask_df.loc[val_idx, 'sample_mask'] = 1

    return mask_df, val_mask_df

# Cell
def scale_data(Y_df, X_df, mask_df, normalizer_y, normalizer_x):
    y_shift = None
    y_scale = None

    # mask = mask.astype(int)
    mask = mask_df['available_mask'].values * mask_df['sample_mask'].values

    if normalizer_y is not None:
        scaler_y = Scaler(normalizer=normalizer_y)
        Y_df['y'] = scaler_y.scale(x=Y_df['y'].values, mask=mask)
    else:
        scaler_y = None

    if normalizer_x is not None:
        X_cols = [col for col in X_df.columns if col not in ['unique_id','ds']]
        for col in X_cols:
            scaler_x = Scaler(normalizer=normalizer_x)
            X_df[col] = scaler_x.scale(x=X_df[col].values, mask=mask)

    return Y_df, X_df, scaler_y

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
def create_datasets(mc, Y_df, X_df, S_df, ds_in_test, shuffle_outsample):
    #TODO: shuffle_outsample

    # n_timestamps_pred defines number of hours ahead to predict

    #------------------------------------- Available and Validation Mask ------------------------------------#
    # mask: 1 last_n_timestamps, 0 timestamps until last_n_timestamps
    train_mask_df = get_default_mask_df(Y_df=Y_df, ds_in_test=ds_in_test, is_test=False)
    outsample_mask_df = get_default_mask_df(Y_df=Y_df, ds_in_test=ds_in_test, is_test=True)

    #---------------------------------------------- Scale Data ----------------------------------------------#
    # Scale data # TODO: write sample_mask conditional/groupby(['unique_id]) scaling
    Y_df, X_df, scaler_y = scale_data(Y_df=Y_df, X_df=X_df, mask_df=train_mask_df,
                                      normalizer_y=mc['normalizer_y'], normalizer_x=mc['normalizer_x'])

    #----------------------------------------- Declare Dataset and Loaders ----------------------------------#
    train_ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, S_df=S_df, mask_df=train_mask_df, verbose=True)
    if ds_in_test == 0:
        outsample_ts_dataset = None
    else:
        outsample_ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, S_df=S_df,
                                                 mask_df=outsample_mask_df, verbose=True)

    return train_ts_dataset, outsample_ts_dataset, scaler_y

# Cell
def instantiate_loaders(mc, train_ts_dataset, outsample_ts_dataset):
    train_ts_loader = TimeSeriesLoader(ts_dataset=train_ts_dataset,
                                       model=mc['model'],
                                       window_sampling_limit=int(mc['window_sampling_limit']),
                                       input_size=int(mc['input_size_multiplier']*mc['output_size']),
                                       output_size=int(mc['output_size']),
                                       idx_to_sample_freq=int(mc['idx_to_sample_freq']),
                                       len_sample_chunks=mc['len_sample_chunks'],
                                       batch_size=int(mc['batch_size']),
                                       n_series_per_batch=mc['n_series_per_batch'],
                                       complete_inputs=mc['complete_inputs'],
                                       complete_sample=mc['complete_sample'],
                                       shuffle=True,
                                       verbose=True)

    if outsample_ts_dataset is not None:
        val_ts_loader = TimeSeriesLoader(ts_dataset=outsample_ts_dataset,
                                        model=mc['model'],
                                        window_sampling_limit=int(mc['window_sampling_limit']),
                                        input_size=int(mc['input_size_multiplier']*mc['output_size']),
                                        output_size=int(mc['output_size']),
                                        idx_to_sample_freq=mc['val_idx_to_sample_freq'],
                                        len_sample_chunks=mc['len_sample_chunks'],
                                        batch_size=1,
                                        n_series_per_batch=mc['n_series_per_batch'],
                                        complete_inputs=mc['complete_inputs'],
                                        complete_sample=mc['complete_sample'],
                                        shuffle=False,
                                        verbose=True)
    else:
        val_ts_loader = None

    return train_ts_loader, val_ts_loader


# Cell
def instantiate_nbeats(mc):
    mc['n_hidden_list'] = len(mc['stack_types']) * [ mc['n_layers'][0]*[mc['n_hidden']] ]
    model = Nbeats(input_size_multiplier=mc['input_size_multiplier'],
                   output_size=int(mc['output_size']),
                   shared_weights=mc['shared_weights'],
                   initialization=mc['initialization'],
                   activation=mc['activation'],
                   stack_types=mc['stack_types'],
                   n_blocks=mc['n_blocks'],
                   n_layers=mc['n_layers'],
                   n_pooling_kernel=mc['n_pooling_kernel'],
                   n_freq_downsample=mc['n_freq_downsample'],
                   n_hidden=mc['n_hidden_list'],
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
    return model

# Cell
def instantiate_esrnn(mc):
    model = ESRNN(# Architecture parameters
                  input_size=int(mc['input_size_multiplier']*mc['output_size']),
                  output_size=int(mc['output_size']),
                  es_component=mc['es_component'],
                  cell_type=mc['cell_type'],
                  state_hsize=int(mc['state_hsize']),
                  dilations=mc['dilations'],
                  add_nl_layer=mc['add_nl_layer'],
                  # Optimization parameters
                  n_iterations=int(mc['n_iterations']),
                  early_stopping=int(mc['early_stopping']),
                  learning_rate=mc['learning_rate'],
                  lr_scheduler_step_size=int(mc['lr_scheduler_step_size']),
                  lr_decay=mc['lr_decay'],
                  per_series_lr_multip=mc['per_series_lr_multip'],
                  gradient_eps=mc['gradient_eps'],
                  gradient_clipping_threshold=mc['gradient_clipping_threshold'],
                  rnn_weight_decay=mc['rnn_weight_decay'],
                  noise_std=mc['noise_std'],
                  level_variability_penalty=mc['level_variability_penalty'],
                  testing_percentile=mc['testing_percentile'],
                  training_percentile=mc['training_percentile'],
                  loss=mc['loss'],
                  val_loss=mc['val_loss'],
                  seasonality=mc['seasonality'],
                  random_seed=int(mc['random_seed'])
                  # Data parameters
                  )
    return model

# Cell
def instantiate_mqesrnn(mc):
    model = MQESRNN(# Architecture parameters
                    input_size=int(mc['input_size_multiplier']*mc['output_size']),
                    output_size=int(mc['output_size']),
                    es_component=mc['es_component'],
                    cell_type=mc['cell_type'],
                    state_hsize=int(mc['state_hsize']),
                    dilations=mc['dilations'],
                    add_nl_layer=mc['add_nl_layer'],
                    # Optimization parameters
                    n_iterations=int(mc['n_iterations']),
                    early_stopping=int(mc['early_stopping']),
                    learning_rate=mc['learning_rate'],
                    lr_scheduler_step_size=int(mc['lr_scheduler_step_size']),
                    lr_decay=mc['lr_decay'],
                    gradient_eps=mc['gradient_eps'],
                    gradient_clipping_threshold=mc['gradient_clipping_threshold'],
                    rnn_weight_decay=mc['rnn_weight_decay'],
                    noise_std=mc['noise_std'],
                    testing_percentiles=list(mc['testing_percentiles']),
                    training_percentiles=list(mc['training_percentiles']),
                    loss=mc['loss'],
                    val_loss=mc['val_loss'],
                    random_seed=int(mc['random_seed'])
                    # Data parameters
                  )
    return model

# Cell
def instantiate_rnn(mc):
    model = RNN(input_size=int(mc['input_size_multiplier']*mc['output_size']),
                output_size=int(mc['output_size']),
                max_epochs=int(mc['max_epochs']),
                learning_rate=mc['learning_rate'],
                lr_scheduler_step_size=int(mc['lr_scheduler_step_size']),
                lr_decay=mc['lr_decay'],
                gradient_eps=mc['gradient_eps'],
                gradient_clipping_threshold=mc['gradient_clipping_threshold'],
                rnn_weight_decay=mc['rnn_weight_decay'],
                noise_std=mc['noise_std'],
                testing_percentile=mc['testing_percentile'],
                training_percentile=mc['training_percentile'],
                cell_type=mc['cell_type'],
                state_hsize=int(mc['state_hsize']),
                dilations=mc['dilations'],
                add_nl_layer=mc['add_nl_layer'],
                loss=mc['loss'],
                random_seed=int(mc['random_seed']))
    return model

# Cell
def instantiate_tcn(mc):
    model = TCN(output_size=int(mc['output_size']),
                n_channels=mc['n_channels'],
                kernel_size=int(mc['kernel_size']),
                initialization=mc['initialization'],
                learning_rate=mc['learning_rate'],
                lr_decay=mc['lr_decay'],
                n_lr_decay_steps=mc['n_lr_decay_steps'],
                weight_decay=mc['weight_decay'],
                dropout_prob=mc['dropout_prob'],
                n_iterations=int(mc['n_iterations']),
                early_stopping=int(mc['early_stopping']),
                loss=mc['loss'],
                val_loss=mc['val_loss'],
                frequency=mc['frequency'],
                random_seed=int(mc['random_seed']),
                seasonality=mc['seasonality'])
    return model

# Cell
def instantiate_model(mc):
    MODEL_DICT = {'nbeats': instantiate_nbeats,
                  'esrnn': instantiate_esrnn,
                  'new_rnn': instantiate_esrnn,
                  'mqesrnn': instantiate_mqesrnn,
                  'rnn': instantiate_rnn,
                  'tcn': instantiate_tcn}
    return MODEL_DICT[mc['model']](mc)

# Cell
def model_fit_predict(mc, Y_df, X_df, S_df, ds_in_test, shuffle_outsample):
    #TODO: rolling forecast
    #TODO: expected_fcds

    Y_df = Y_df.copy()
    if X_df is not None:
        X_df = X_df.copy()
    if S_df is not None:
        S_df = S_df.copy()

    #----------------------------------------------- Datasets -----------------------------------------------#
    train_ts_dataset, outsample_ts_dataset, scaler_y = create_datasets(mc=mc, Y_df=Y_df, X_df=X_df,
                                                                       S_df=S_df,
                                                                       ds_in_test=ds_in_test,
                                                                       shuffle_outsample=shuffle_outsample)

    #--------------------------------------- Instantiate, fit, predict ---------------------------------------#
    train_ts_loader, val_ts_loader = instantiate_loaders(mc=mc, train_ts_dataset=train_ts_dataset,
                                                         outsample_ts_dataset=outsample_ts_dataset)
    model = instantiate_model(mc=mc)

    # Val loader not implemented during training for ESRNN and RNN
    model.fit(train_ts_loader=train_ts_loader, val_ts_loader=val_ts_loader, verbose=True,
                eval_freq=mc['eval_freq'])
    y_true, y_hat, mask = model.predict(ts_loader=val_ts_loader, return_decomposition=False)

    print("y_true.shape (#n_series, #n_fcds, #lt) ", y_true.shape)
    print("y_hat.shape (#n_series, #n_fcds, #lt) ", y_hat.shape)
    print("\n")

    meta_data = val_ts_loader.ts_dataset.meta_data

    return y_true, y_hat, mask, meta_data, model

# Cell
def evaluate_model(mc, loss_function, Y_df, X_df, S_df, ds_in_test, shuffle_outsample,
                   kwargs_loss):

    # Some asserts due to work in progress
    assert mc['normalizer_y'] is None, 'Scaling Y not iplemented (inverse Y missing for loss)'
    print(30*'-')
    print(pd.Series(mc))
    print(30*'-')

    n_series = Y_df['unique_id'].nunique()
    if n_series > 1:
        assert mc['normalizer_x'] is None, 'Data scaling not implemented with multiple time series'
    assert shuffle_outsample == False, 'Shuffle outsample not implemented'

    assert ds_in_test % mc['val_idx_to_sample_freq']==0, 'outsample size should be multiple of val_idx_to_sample_freq'

    # Make predictions
    start = time.time()
    y_true, y_hat, mask, meta_data, model = model_fit_predict(mc=mc, Y_df=Y_df, X_df=X_df,
                                                              S_df=S_df, ds_in_test=ds_in_test,
                                                              shuffle_outsample=shuffle_outsample)
    run_time = time.time() - start

    # Evaluate predictions
    loss = loss_function(y=y_true, y_hat=y_hat, weights=mask, **kwargs_loss)

    result =  {'loss': loss,
               'mc': mc,
            #    'y_true': y_true,
            #    'y_hat': y_hat,
               'trajectories': model.trajectories,
               'run_time': run_time,
               'status': STATUS_OK}
    return result

# Cell
def hyperopt_tunning(space, hyperopt_iters, loss_function, Y_df, X_df, S_df, ds_in_test,
                     shuffle_outsample, save_trials=False,
                     kwargs_loss=None):
    trials = Trials()
    fmin_objective = partial(evaluate_model, loss_function=loss_function, Y_df=Y_df, X_df=X_df, S_df=S_df,
                             ds_in_test=ds_in_test,
                             shuffle_outsample=shuffle_outsample,
                             kwargs_loss=kwargs_loss or {})

    fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=hyperopt_iters, trials=trials, verbose=True)

    return trials