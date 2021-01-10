# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/experiments_nbeats__hyperopt_epf.ipynb (unless otherwise specified).

__all__ = ['forecast_evaluation_table', 'run_val_nbeatsx', 'get_experiment_space', 'parse_trials', 'main', 'parse_args']

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

from ...data.scalers import Scaler
from ...data.datasets.epf import EPF, EPFInfo
from ...data.tsdataset import TimeSeriesDataset
from ...data.tsloader_fast import TimeSeriesLoader
from ...losses.numpy import mae, mape, smape, rmse, pinball_loss

# Models
from ...models.nbeats.nbeats import Nbeats

import warnings
warnings.filterwarnings("ignore")

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Cell
def forecast_evaluation_table(y_true, y_hat):
    y_true = y_true.reshape(-1)
    y_hat = y_hat.reshape(-1)

    _pinball50 = np.round(pinball_loss(y_true, y_hat, tau=0.5),5)
    _mae   = np.round(mae(y_true, y_hat),5)
    _mape  = np.round(mape(y_true, y_hat),5)
    _smape = np.round(smape(y_true, y_hat),5)
    _rmse  = np.round(rmse(y_true, y_hat),5)

    evaluations = pd.DataFrame({'metric': ['pinball50', 'mae', 'mape', 'smape', 'rmse'],
                                'nbeatsx': [_pinball50, _mae, _mape, _smape, _rmse]})

    return evaluations

def run_val_nbeatsx(mc, train_loader, val_loader, trials, trials_file_name, final_evaluation=False):
    # Save trials, can analyze progress
    save_every_n_step = 5
    current_step = len(trials.trials)
    if (current_step % save_every_n_step==0):
        with open(trials_file_name, "wb") as f:
            pickle.dump(trials, f)

    start_time = time.time()

    model = Nbeats(input_size_multiplier=int(mc['input_size_multiplier']),
                   output_size=int(mc['output_size']),
                   shared_weights=int(mc['shared_weights']),
                   activation=mc['activation'],
                   initialization=mc['initialization'],
                   stack_types=mc['stack_types'], #2*['identity'],
                   n_blocks=mc['n_blocks'], #2*[1],
                   n_layers=mc['n_layers'], #2*[2],
                   n_hidden=2*[2*[int(mc['n_hidden'])]], #2*[[256,256]]
                   n_polynomials=mc['n_polynomials'], #2,
                   n_harmonics=int(mc['n_harmonics']), #1,
                   exogenous_n_channels=int(mc['exogenous_n_channels']), #9,
                   include_var_dict={'y': [-2, -3, -8],
                                     'Exogenous1': [-1, -2, -8],
                                     'Exogenous2': [-1, -2, -8],
                                     'week_day': [-1]}, #TODO: mc['include_var_dict]
                   t_cols=train_loader.ts_dataset.t_cols,
                   batch_normalization=mc['batch_normalization'], #False,
                   dropout_prob_theta=float(mc['dropout_prob_theta']), #0.01,
                   dropout_prob_exogenous=float(mc['dropout_prob_exogenous']), #0.01,
                   x_s_n_hidden=int(mc['x_s_n_hidden']), #0,
                   learning_rate=float(mc['learning_rate']), #0.007,
                   lr_decay=float(mc['lr_decay']), #0.5,
                   n_lr_decay_steps=int(mc['n_lr_decay_steps']), #3,
                   weight_decay=float(mc['weight_decay']), #0.0000001,
                   l1_theta=float(mc['l1_theta']), #0.0001,
                   n_iterations=int(mc['n_iterations']), #200,
                   early_stopping=int(mc['early_stopping']), #40,
                   loss=mc['loss'], #'PINBALL',
                   loss_hypar=float(mc['loss_hypar']), #0.5,
                   val_loss=mc['val_loss'], #'MAE',
                   frequency=mc['frequency'], #'H',
                   random_seed=int(mc['random_seed']), #1,
                   seasonality=int(mc['seasonality'])) #24)

    model.fit(train_ts_loader=train_loader, val_ts_loader=val_loader, verbose=True, eval_steps=10) # aqui val_loader==Test

    # TODO: Pytorch numerical error hacky protection
    hyperopt_reported_loss = model.final_outsample_loss
    if np.isnan(model.final_insample_loss):
        hyperopt_reported_loss = 100
    if model.final_insample_loss<=0:
        hyperopt_reported_loss = 100

    if np.isnan(model.final_outsample_loss):
        hyperopt_reported_loss = 100
    if model.final_outsample_loss<=0:
        hyperopt_reported_loss = 100

    results =  {'loss': model.final_outsample_loss,
                'loss_name': mc['val_loss'], #val_mae <--------
                'mc': mc,
                'final_insample_loss': model.final_insample_loss,
                'final_outsample_loss': model.final_outsample_loss,
                'trajectories': model.trajectories,
                'run_time': time.time() - start_time,
                'status': STATUS_OK}

    if final_evaluation:
        print('Best Model Hyperpars')
        print(75*'=')
        print(pd.Series(mc))
        print(75*'='+'\n')

        print('Best Model Evaluation')
        y_true, y_hat, _ = model.predict(ts_loader=val_loader, eval_mode=True)
        print(forecast_evaluation_table(y_true, y_hat))

    return results

# Cell
def get_experiment_space(args):
    if args.space=='nbeats_extended1':
        space = {# Architecture parameters
                 'input_size_multiplier': hp.choice('input_size_multiplier', [7]),  #<------- TODO: Change for n_xt
                 'output_size': hp.choice('output_size', [24]),
                 'shared_weights': hp.choice('shared_weights', [False]),
                 'activation': hp.choice('activation', ['relu','softplus','tanh','selu','lrelu','prelu','sigmoid']),
                 'initialization':  hp.choice('initialization', ['orthogonal','he_uniform','he_normal',
                                                                 'glorot_uniform','glorot_normal','lecun_normal']),
                 'stack_types': hp.choice('stack_types', [ ['identity'],
                                                            1*['identity']+['exogenous_wavenet'],
                                                                ['exogenous_wavenet']+1*['identity'],
                                                            1*['identity']+['exogenous_tcn'],
                                                                ['exogenous_tcn']+1*['identity'] ]),
                 'n_blocks': hp.choice('n_blocks', [ [1, 1] ]),
                 'n_layers': hp.choice('n_layers', [ [2, 2] ]),
                 'n_hidden': hp.quniform('n_hidden_1', 50, 500, 1), #<------- TODO: Change for n_theta_list
                 'n_harmonics': hp.choice('n_harmonics', [1]), #<--------- TODO: Eliminate unnecesary hypar
                 'n_polynomials': hp.choice('n_polynomials', [2]), #<----- TODO: Eliminate unnecesary hypar
                 'exogenous_n_channels': hp.quniform('exogenous_n_channels', 1, 10, 1), #<------- TODO: Change for n_xt_channels
                 'x_s_n_hidden': hp.choice('x_s_n_hidden', [0]), #<------- TODO: Change for n_xs_hidden
                 # Regularization and optimization parameters
                 'batch_normalization': hp.choice('batch_normalization', [True, False]),
                 'dropout_prob_theta': hp.uniform('dropout_prob_theta', 0, 1),
                 'dropout_prob_exogenous': hp.uniform('dropout_prob_exogenous', 0, 0.5),
                 'learning_rate': hp.loguniform('learning_rate', np.log(5e-4), np.log(0.1)),
                 'lr_decay': hp.choice('lr_decay', [0.5]),
                 'n_lr_decay_steps': hp.choice('n_lr_decay_steps', [3]),
                 'weight_decay': hp.loguniform('weight_decay', np.log(5e-4), np.log(0.01)),
                 'n_iterations': hp.choice('n_iterations', [args.max_epochs]), #<------- TODO: Change for max_epochs
                 'early_stopping': hp.choice('early_stopping', [40]),
                 'loss': hp.choice('loss', ['PINBALL']),
                 'loss_hypar': hp.uniform('loss_hypar', 0.45, 0.55),
                 'val_loss': hp.choice('val_loss', [args.val_loss]),
                 'l1_theta': hp.choice('l1_theta', [0, hp.loguniform('lambdal1', np.log(1e-5), np.log(1))]),
                 # Data parameters
                 'frequency': hp.choice('frequency', ['H']),
                 'seasonality': hp.choice('seasonality', [24]),
                 'include_var_dict': hp.choice('include_var_dict', [{'y': [-2, -3, -8],
                                                                     'Exogenous1': [-1, -2, -8],
                                                                     'Exogenous2': [-1, -2, -8],
                                                                     'week_day': [-1]}]),
                 'random_seed': hp.quniform('random_seed', 1, 20, 1)}


    elif args.space=='nbeats_collapsed':
        space= {# Architecture parameters
                'input_size_multiplier': hp.choice('input_size_multiplier', [7]),  #<------- TODO: Change for n_xt
                'output_size': hp.choice('output_size', [24]),
                'shared_weights': hp.choice('shared_weights', [False]),
                'activation': hp.choice('activation', ['relu','softplus','tanh','selu','lrelu','prelu','sigmoid']),
                'initialization':  hp.choice('initialization', ['orthogonal','he_uniform','he_normal',
                                                                'glorot_uniform','glorot_normal','lecun_normal']),
                'stack_types': hp.choice('stack_types', [ #['identity'],
                                                          #  1*['identity']+['exogenous_wavenet'],
                                                            ['exogenous_wavenet']+1*['identity'],
                                                          #  1*['identity']+['exogenous_tcn'],
                                                            ['exogenous_tcn']+1*['identity'] ]),
                'n_blocks': hp.choice('n_blocks', [ [1, 1] ]),
                'n_layers': hp.choice('n_layers', [ [2, 2] ]),
                'n_hidden': hp.quniform('n_hidden_1', 50, 500, 1), #<------- TODO: Change for n_theta_list
                'n_harmonics': hp.choice('n_harmonics', [1]), #<--------- TODO: Eliminate unnecesary hypar
                'n_polynomials': hp.choice('n_polynomials', [2]), #<----- TODO: Eliminate unnecesary hypar
                'exogenous_n_channels': hp.quniform('exogenous_n_channels', 1, 10, 1), #<------- TODO: Change for n_xt_channels
                'x_s_n_hidden': hp.choice('x_s_n_hidden', [0]), #<------- TODO: Change for n_xs_hidden
                # Regularization and optimization parameters
                'batch_normalization': hp.choice('batch_normalization', [False]),
                'dropout_prob_theta': hp.uniform('dropout_prob_theta', 0, 1),
                'dropout_prob_exogenous': hp.uniform('dropout_prob_exogenous', 0, 0.5),
                'learning_rate': hp.loguniform('learning_rate', np.log(5e-4), np.log(0.1)),
                'lr_decay': hp.uniform('lr_decay', 0.3, 1.0),
                'n_lr_decay_steps': hp.choice('n_lr_decay_steps', [3]),
                'weight_decay': hp.loguniform('weight_decay', np.log(5e-5), np.log(5e-3)),
                'n_iterations': hp.choice('n_iterations', [args.max_epochs]), #<------- TODO: Change for max_epochs
                'early_stopping': hp.choice('early_stopping', [40]),
                'loss': hp.choice('loss', ['PINBALL']),
                'loss_hypar': hp.uniform('loss_hypar', 0.48, 0.51),
                'val_loss': hp.choice('val_loss', [args.val_loss]),
                'l1_theta': hp.choice('l1_theta', [0, hp.loguniform('lambdal1', np.log(1e-5), np.log(1))]),
                # Data parameters
                'frequency': hp.choice('frequency', ['H']),
                'seasonality': hp.choice('seasonality', [24]),
                'include_var_dict': hp.choice('include_var_dict', [{'y': [-2, -3, -8],
                                                                    'Exogenous1': [-1, -2, -8],
                                                                    'Exogenous2': [-1, -2, -8],
                                                                    'week_day': [-1]}]),
                'random_seed': hp.quniform('random_seed', 10, 20, 1)}

    else:
        print(f'Experiment space {args.space} not available')

    return space

def parse_trials(trials):
    # Initialize
    trials_dict = {'tid': [], 'loss': [], 'trajectories': [], 'mc': []}
    for tidx in range(len(trials)):
        # Main
        trials_dict['tid']  += [trials.trials[tidx]['tid']]
        trials_dict['loss'] += [trials.trials[tidx]['result']['loss']]
        trials_dict['trajectories'] += [trials.trials[tidx]['result']['trajectories']]

        # Model Configs
        mc = trials.trials[tidx]['result']['mc']
        trials_dict['mc'] += [mc]

    trials_df = pd.DataFrame(trials_dict)
    return trials_df

def main(args):
    #---------------------------------------------- Directories ----------------------------------------------#
    output_dir = f'./results/{args.dataset}/{args.space}/'
    os.makedirs(output_dir, exist_ok = True)
    assert os.path.exists(output_dir), f'Output dir {output_dir} does not exist'

    hyperopt_file = output_dir + f'hyperopt_{args.experiment_id}.p'
    result_test_file = output_dir + f'result_test_{args.experiment_id}.p'

    #---------------------------------------------- Read  Data ----------------------------------------------#
    print('\n'+75*'-')
    print(28*'-', 'Preparing Dataset', 28*'-')
    print(75*'-'+'\n')

    #TEST_DATE = {'NP': '2016-12-27',
    #             'PJM':'2016-12-27',
    #             'BE':'2015-01-04',
    #             'FR': '2015-01-04',
    #             'DE':'2016-01-04'}
    #test_date = TEST_DATE[args.dataset]
    #Y_insample_df, Xt_insample_df, Y_outsample_df, Xt_outsample_df, _ = load_epf(directory='../data/',
    #                                                                             market=args.dataset,
    #                                                                             first_date_test=test_date,
    #                                                                             days_in_test=728)
    Y_df, Xt_df = EPF.load(directory='../data/', group=args.dataset)

    # To not modify original data
    Xt_scaled_df = Xt_df.copy()

    # Transform data with scale transformation
    offset = 365 * 24 * 2
    scaler = Scaler(normalizer='norm')
    Xt_scaled_df['Exogenous1'] = scaler.scale(x=Xt_scaled_df['Exogenous1'].values, offset=offset)

    scaler = Scaler(normalizer='norm')
    Xt_scaled_df['Exogenous2'] = scaler.scale(x=Xt_scaled_df['Exogenous2'].values, offset=offset)

    # train_mask: 1 to keep, 0 to mask
    train_outsample_mask = np.ones(len(Y_df))
    train_outsample_mask[-offset:] = 0

    ts_dataset = TimeSeriesDataset(Y_df=Y_df, S_df=None, X_df=Xt_scaled_df,
                                   ts_train_mask=train_outsample_mask)

    train_loader = TimeSeriesLoader(ts_dataset=ts_dataset,
                                    model='nbeats',
                                    offset=0, #offset,
                                    window_sampling_limit=365*4*24,
                                    input_size=7*24,
                                    output_size=24,
                                    idx_to_sample_freq=24,
                                    batch_size=256,
                                    is_train_loader=True,
                                    shuffle=True)

    val_loader = TimeSeriesLoader(ts_dataset=ts_dataset,
                                  model='nbeats',
                                  offset=0, #offset,
                                  window_sampling_limit=365*4*24,
                                  input_size=7*24,
                                  output_size=24,
                                  idx_to_sample_freq=24,
                                  batch_size=1024,
                                  is_train_loader=False, # Samples the opposite of train_outsample_mask
                                  shuffle=False)

    print(f'Dataset: {args.dataset}')
    #print("Xt_df.columns", Xt_df.columns)
    print(f'Train mask percentage: {np.round(np.sum(train_outsample_mask)/len(train_outsample_mask),2)}')
    print('X: time series features, of shape (#hours, #times,#features): \t' + str(Xt_df.shape))
    print('Y: target series (in X), of shape (#hours, #times): \t \t' + str(Y_df.shape))
    print(f'Train {sum(1-train_outsample_mask)} hours = {np.round(sum(1-train_outsample_mask)/(24*365),2)} years')
    print(f'Validation {sum(train_outsample_mask)} hours = {np.round(sum(train_outsample_mask)/(24*365),2)} years')
    # print('S: static features, of shape (#series,#features): \t \t' + str(S.shape))
    #Y_df.head()
    print('\n')

    #-------------------------------------- Hyperparameter Optimization --------------------------------------#

    if not os.path.isfile(hyperopt_file):
        print('\n'+75*'-')
        print(22*'-', 'Start Hyperparameter  tunning', 22*'-')
        print(75*'-'+'\n')

        space = get_experiment_space(args)

        trials = Trials()
        #fmin_objective = partial(run_val_nbeatsx, y_df=y_insample_df, X_t_df=X_t_insample_df, val_ds=365,
        #                         trials=trials, trials_file_name=hyperopt_file)
        fmin_objective = partial(run_val_nbeatsx, train_loader=train_loader, val_loader=val_loader,
                                 trials=trials, trials_file_name=hyperopt_file)
        fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=args.hyperopt_iters, trials=trials, verbose=True)

        # Save output
        with open(hyperopt_file, "wb") as f:
            pickle.dump(trials, f)

    print('\n'+75*'-')
    print(20*'-', 'Hyperparameter  tunning  finished', 20*'-')
    print(75*'-'+'\n')

    # Read and parse trials pickle
    trials = pickle.load(open(hyperopt_file, 'rb'))
    trials_df = parse_trials(trials)

    # Get best mc
    idx = trials_df.loss.idxmin()
    best_mc = trials_df.loc[idx]['mc']

    run_val_nbeatsx(best_mc, train_loader=train_loader, val_loader=val_loader,
                    trials=trials, trials_file_name=hyperopt_file, final_evaluation=True)

def parse_args():
    desc = "NBEATSx overfit"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, required=True, help='NP')
    parser.add_argument('--space', type=str, required=True, help='Experiment hyperparameter space')
    parser.add_argument('--hyperopt_iters', type=int, help='hyperopt_iters')
    parser.add_argument('--max_epochs', type=int, required=True, default=2000, help='max train epochs')
    parser.add_argument('--val_loss', type=str, required=False, default=None, help='validation loss')
    parser.add_argument('--experiment_id', default=None, required=False, type=str, help='string to identify experiment')
    return parser.parse_args()


# Cell
if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    main(args)

# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python nixtla/experiments/nbeats/hyperopt_epf.py --dataset 'NP' --space "nbeats_collapsed" --hyperopt_iters 200 --val_loss "SMAPE" --experiment_id "SMAPEval_20210110"
# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python src/overfit_nbeatsx.py --dataset 'NP' --space "nbeats_collapsed" --hyperopt_iters 200 --val_loss "SMAPE" --experiment_id "SMAPEval_20210110"