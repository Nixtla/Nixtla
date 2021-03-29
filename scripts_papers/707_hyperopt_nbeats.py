import os
import pickle
import glob
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from nixtla.losses.numpy import mae, mape, smape, rmse, pinball_loss
from nixtla.experiments.utils import *
from nixtla.data.tsdataset import TimeSeriesDataset
from nixtla.data.tsloader_general import TimeSeriesLoader

def evaluate_horizon(horizon, data, n_trials, feature):    
    nbeats_space= {# Architecture parameters
                'model':'nbeats',
                'input_size_multiplier': hp.choice('input_size_multiplier', [1, 2, 3, 4, 5]),
                'output_size': hp.choice('output_size', [horizon]),
                'shared_weights': hp.choice('shared_weights', [False]),
                'activation': hp.choice('activation', ['relu','selu']),
                'initialization':  hp.choice('initialization', ['glorot_normal','he_normal']),
                'stack_types': hp.choice('stack_types', [ ['trend', 'seasonality'] ]),
                'n_blocks': hp.choice('n_blocks', [ [1, 1], [3, 3] ]),
                'n_layers': hp.choice('n_layers', [ 6*[2] ]),
                'n_hidden': hp.choice('n_hidden', [ 256, 512 ]),
                'n_harmonics': hp.choice('n_harmonics', [1, 2]),
                'n_polynomials': hp.choice('n_polynomials', [2, 4]),
                'exogenous_n_channels': hp.quniform('exogenous_n_channels', 1, 10, 1),
                'x_s_n_hidden': hp.choice('x_s_n_hidden', [0]),
                # Regularization and optimization parameters
                'batch_normalization': hp.choice('batch_normalization', [False, True]),
                'dropout_prob_theta': hp.uniform('dropout_prob_theta', 0, 0.5),
                'dropout_prob_exogenous': hp.uniform('dropout_prob_exogenous', 0, 0.5),
                'learning_rate': hp.loguniform('learning_rate', np.log(5e-4), np.log(0.001)),
                'lr_decay': hp.uniform('lr_decay', 0.3, 0.5),
                'n_lr_decay_steps': hp.choice('n_lr_decay_steps', [3]),
                'weight_decay': hp.loguniform('weight_decay', np.log(5e-5), np.log(5e-3)),
                'n_iterations': hp.choice('n_iterations', [2_000]), #1_000
                'early_stopping': hp.choice('early_stopping', [5]),
                'eval_freq': hp.choice('eval_freq', [50]),
                'n_val_weeks': hp.choice('n_val_weeks', [52*2]),
                'loss': hp.choice('loss', ['MAE', 'MSE']),
                'loss_hypar': hp.choice('loss_hypar', [0.5]),                
                'val_loss': hp.choice('val_loss', ['MAE']),
                'l1_theta': hp.choice('l1_theta', [0]),
                # Data parameters
                'len_sample_chunks': hp.choice('len_sample_chunks', [None]),
                'normalizer_y': hp.choice('normalizer_y', [None]),
                'normalizer_x': hp.choice('normalizer_x', [None]),
                'window_sampling_limit': hp.choice('window_sampling_limit', [100_000]),
                'complete_inputs': hp.choice('complete_inputs', [True]),
                'complete_sample': hp.choice('complete_sample', [True]),                
                'frequency': hp.choice('frequency', ['H']),
                'seasonality': hp.choice('seasonality', [24]),      
                'idx_to_sample_freq': hp.choice('idx_to_sample_freq', [1]),
                'val_idx_to_sample_freq': hp.choice('val_idx_to_sample_freq', [1]),
                'batch_size': hp.choice('batch_size', [256]),
                'n_series_per_batch': hp.choice('n_series_per_batch', [1]),
                'random_seed': hp.quniform('random_seed', 10, 20, 1)}

    n_patients = data.unique_id.nunique()           
    Y_df = data[['unique_id','ds', feature]]
    Y_df = Y_df.sort_values(['unique_id','ds']).reset_index(drop=True)
    Y_df = Y_df.rename(columns={feature:'y'})
    Y_df['ds'] = np.tile(np.array(range(10000)), n_patients)
    Y_train_df = Y_df[Y_df['ds']<10000-horizon].reset_index(drop=True)
    Y_train_df['ds'] = pd.to_datetime(Y_train_df['ds'])
    Y_df['ds'] = pd.to_datetime(Y_df['ds'])

    trials = hyperopt_tunning(space=nbeats_space, hyperopt_iters=n_trials, loss_function=mae, Y_df=Y_train_df, X_df=None, S_df=None,
                            ds_in_test=horizon, shuffle_outsample=False)

    # Best mc
    mc = trials.trials[np.argmin(trials.losses())]['result']['mc']

    # Model
    final_nbeats = instantiate_nbeats(mc)

    # Datasets
    # Train and val
    train_ts_dataset, validation_ts_dataset, scaler_y = create_datasets(mc=mc, Y_df=Y_train_df, X_df=None, S_df=None, ds_in_test=horizon,                                                                                 shuffle_outsample=False)
    # Test
    test_mask_df = get_default_mask_df(Y_df=Y_df, ds_in_test=horizon, is_test=True)
    test_ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=None, S_df=None, mask_df=test_mask_df, verbose=True)

    # Loaders
    # Train and val
    train_ts_loader, val_ts_loader = instantiate_loaders(mc=mc, train_ts_dataset=train_ts_dataset, outsample_ts_dataset=validation_ts_dataset)
    # Test
    test_ts_loader = TimeSeriesLoader(ts_dataset=test_ts_dataset,
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
                                    shuffle=False)

    # Val loader not implemented during training for ESRNN and RNN
    final_nbeats.fit(train_ts_loader=train_ts_loader, val_ts_loader=val_ts_loader, verbose=True,
                    eval_freq=mc['eval_freq'])

    y_true_test, y_hat_test, mask_test = final_nbeats.predict(ts_loader=test_ts_loader, return_decomposition=False)

    fig, ax = plt.subplots(nrows=3, ncols=4, figsize = (15,10))
    for i in range(12):
        ax[i//4, i%4].plot(y_true_test[i,0,:])
        ax[i//4, i%4].plot(y_hat_test[i,0,:])
        ax[i//4, i%4].grid(True)
        ax[i//4, i%4].set_xlabel('Timestamp')
        ax[i//4, i%4].set_ylabel('ART')
        ax[i//4, i%4].set_title(f'Patient {i}')
    plt.tight_layout()
    plt.savefig(f'./results/{feature}_{horizon}.pdf')
    plt.cla()

    return y_true_test, y_hat_test


def main(args):
    data = pd.read_csv('./data/healthcare/data_waveforms_icu.csv')
    data.head()

    horizons = [15, 30, 60, 120, 240, 480, 960]
    mae_list = []
    rmse_list = []
    y_true_list = []
    y_hat_list = []
    for horizon in horizons:
        print(100*'-')
        print(100*'-')
        print('HORIZON: ', horizon)
        y_true, y_hat = evaluate_horizon(horizon=horizon, data=data, n_trials=args.hyperopt_iters, feature=args.feature)
        y_true_list.append(y_true)
        y_hat_list.append(y_hat)
        mae_list.append(mae(y_true, y_hat))
        rmse_list.append(rmse(y_true, y_hat))
        print(100*'-')
        print(100*'-')

    result = {'horizons': horizons, 'y_true':y_true_list, 'y_hat':y_hat_list, 'mae': mae_list, 'rmse': rmse_list}
    with open(f'./results/result_{args.feature}_{args.experiment_id}.p', "wb") as f:
        pickle.dump(result, f)

    # plt.plot(horizons, mae_list)
    # plt.xlabel('Forecasting Horizon')
    # plt.ylabel('MAE')
    # plt.grid()
    # plt.savefig('horizon_vs_mae.pdf')

def parse_args():
    desc = "707"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--feature', type=str, help='feature')
    parser.add_argument('--hyperopt_iters', type=int, help='hyperopt_iters')
    parser.add_argument('--experiment_id', default=None, required=False, type=str, help='string to identify experiment')
    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    main(args)

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate riemann
# PYTHONPATH=. python scripts_papers/707_hyperopt_nbeats.py --feature 'ART' --hyperopt_iters 2 --experiment_id "debug"
