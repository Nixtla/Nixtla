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
from nixtla.data.scalers import Scaler


def create_space(model, horizon, pooling):
    if pooling:
        print('Pooling activated')
        n_pooling_kernel = [ 2*[1], 2*[2], 2*[4], 2*[8], 2*[16], 2*[32] ]
    else:
        n_pooling_kernel = [ 2*[1] ]

    if model == 'nbeats_i':
        space= {# Architecture parameters
                    'model':'nbeats',
                    'input_size_multiplier': hp.choice('input_size_multiplier', [1, 2, 3, 4, 5]),
                    'output_size': hp.choice('output_size', [horizon]),
                    'shared_weights': hp.choice('shared_weights', [False]),
                    'activation': hp.choice('activation', ['relu']),
                    'initialization':  hp.choice('initialization', ['default']),
                    'stack_types': hp.choice('stack_types', [ ['trend', 'seasonality'] ]),
                    'n_blocks': hp.choice('n_blocks', [ [1, 1], [3, 3] ]),
                    'n_layers': hp.choice('n_layers', [ 10*[2] ]),
                    'n_pooling_kernel': hp.choice('n_pooling_kernel', n_pooling_kernel),
                    'n_freq_downsample': hp.choice('n_freq_downsample', [ [1, 1] ]),
                    'n_hidden': hp.choice('n_hidden', [ 32, 64, 128, 256, 512 ]),
                    'n_harmonics': hp.choice('n_harmonics', [1, 2]),
                    'n_polynomials': hp.choice('n_polynomials', [2, 4]),
                    'exogenous_n_channels': hp.choice('exogenous_n_channels', [0]),
                    'x_s_n_hidden': hp.choice('x_s_n_hidden', [0]),
                    # Regularization and optimization parameters
                    'batch_normalization': hp.choice('batch_normalization', [False]),
                    'dropout_prob_theta': hp.choice('dropout_prob_theta', [0]),
                    'dropout_prob_exogenous': hp.choice('dropout_prob_exogenous', [0]),
                    'learning_rate': hp.loguniform('learning_rate', np.log(5e-4), np.log(0.001)),
                    'lr_decay': hp.choice('lr_decay', [0.5]),
                    'n_lr_decay_steps': hp.choice('n_lr_decay_steps', [3]),
                    'weight_decay': hp.choice('weight_decay', [0]),
                    'n_iterations': hp.choice('n_iterations', [3_000]),
                    'early_stopping': hp.choice('early_stopping', [10]),
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
                    'batch_size': hp.choice('batch_size', [128, 256, 512]),
                    'n_series_per_batch': hp.choice('n_series_per_batch', [1]),
                    'random_seed': hp.quniform('random_seed', 1, 50, 1)}
    elif model == 'nbeats_g':
        space= {# Architecture parameters
                    'model':'nbeats',
                    'input_size_multiplier': hp.choice('input_size_multiplier', [1, 2, 3, 4, 5]),
                    'output_size': hp.choice('output_size', [horizon]),
                    'shared_weights': hp.choice('shared_weights', [False]),
                    'activation': hp.choice('activation', ['relu']),
                    'initialization':  hp.choice('initialization', ['default']),
                    'stack_types': hp.choice('stack_types', [ ['identity'] ]),
                    'n_blocks': hp.choice('n_blocks', [ [3], [5] ]),
                    'n_layers': hp.choice('n_layers', [ 10*[2] ]),
                    'n_pooling_kernel': hp.choice('n_pooling_kernel', n_pooling_kernel),
                    'n_freq_downsample': hp.choice('n_freq_downsample', [ [1] ]),
                    'n_hidden': hp.choice('n_hidden', [ 32, 64, 128, 256, 512 ]),
                    'n_harmonics': hp.choice('n_harmonics', [0]),
                    'n_polynomials': hp.choice('n_polynomials', [0]),
                    'exogenous_n_channels': hp.choice('exogenous_n_channels', [0]),
                    'x_s_n_hidden': hp.choice('x_s_n_hidden', [0]),
                    # Regularization and optimization parameters
                    'batch_normalization': hp.choice('batch_normalization', [False]),
                    'dropout_prob_theta': hp.choice('dropout_prob_theta', [0]),
                    'dropout_prob_exogenous': hp.choice('dropout_prob_exogenous', [0]),
                    'learning_rate': hp.loguniform('learning_rate', np.log(5e-4), np.log(0.001)),
                    'lr_decay': hp.choice('lr_decay', [0.5]),
                    'n_lr_decay_steps': hp.choice('n_lr_decay_steps', [3]),
                    'weight_decay': hp.choice('weight_decay', [0]),
                    'n_iterations': hp.choice('n_iterations', [3_000]),
                    'early_stopping': hp.choice('early_stopping', [10]),
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
                    'batch_size': hp.choice('batch_size', [128, 256, 512]),
                    'n_series_per_batch': hp.choice('n_series_per_batch', [1]),
                    'random_seed': hp.quniform('random_seed', 1, 50, 1)}
    elif model == 'MLP':
        space= {# Architecture parameters
                    'model':'nbeats',
                    'input_size_multiplier': hp.choice('input_size_multiplier', [1, 2, 3, 4, 5]),
                    'output_size': hp.choice('output_size', [horizon]),
                    'shared_weights': hp.choice('shared_weights', [False]),
                    'activation': hp.choice('activation', ['relu']),
                    'initialization':  hp.choice('initialization', ['default']),
                    'stack_types': hp.choice('stack_types', [ ['identity'] ]),
                    'n_blocks': hp.choice('n_blocks', [ [1] ]),
                    'n_layers': hp.choice('n_layers', [ 10*[2] ]),
                    'n_pooling_kernel': hp.choice('n_pooling_kernel', n_pooling_kernel),
                    'n_freq_downsample': hp.choice('n_freq_downsample', [ [1] ]),
                    'n_hidden': hp.choice('n_hidden', [ 32, 64, 128, 256, 512 ]),
                    'n_harmonics': hp.choice('n_harmonics', [0]),
                    'n_polynomials': hp.choice('n_polynomials', [0]),
                    'exogenous_n_channels': hp.choice('exogenous_n_channels', [0]),
                    'x_s_n_hidden': hp.choice('x_s_n_hidden', [0]),
                    # Regularization and optimization parameters
                    'batch_normalization': hp.choice('batch_normalization', [False]),
                    'dropout_prob_theta': hp.choice('dropout_prob_theta', [0]),
                    'dropout_prob_exogenous': hp.choice('dropout_prob_exogenous', [0]),
                    'learning_rate': hp.loguniform('learning_rate', np.log(5e-4), np.log(0.001)),
                    'lr_decay': hp.choice('lr_decay', [0.5]),
                    'n_lr_decay_steps': hp.choice('n_lr_decay_steps', [3]),
                    'weight_decay': hp.choice('weight_decay', [0]),
                    'n_iterations': hp.choice('n_iterations', [3_000]),
                    'early_stopping': hp.choice('early_stopping', [10]),
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
                    'batch_size': hp.choice('batch_size', [128, 256, 512]),
                    'n_series_per_batch': hp.choice('n_series_per_batch', [1]),
                    'random_seed': hp.quniform('random_seed', 1, 50, 1)}

    return space

def evaluate_horizon(model, horizon, len_validation, len_test, data, n_trials, feature, pooling):
    # ------------------------------------------------------- HYPERPARAMATER SPACE --------------------------------------------------
    nbeats_space = create_space(model=model, horizon=horizon, pooling=pooling)

    # ------------------------------------------------------- DATA PROCESSING -------------------------------------------------------
    if feature == 'BOTH':
        features = ['ART', 'PLETH']
    else:
        features = [feature]

    ts_per_patient = 10000     
    n_patients = data.unique_id.nunique()
    uniques = data.unique_id.unique()
    Y_df = data[['unique_id','ds'] + features].copy()
    Y_df = Y_df.sort_values(['unique_id','ds']).reset_index(drop=True)
    Y_df['ds'] = np.tile(np.array(range(ts_per_patient)), n_patients)

    # Scaling
    scaler_list = []
    for f in features:
        print(f'Scaling {f}...')
        scaled_y_list = []
        feature_scaler_list = []
        for uid in uniques:
            serie_data = Y_df[Y_df['unique_id']==uid]
            scaler_y = Scaler(normalizer='median')
            scaled_y = scaler_y.scale(x=serie_data[f].values, mask=np.ones(len(serie_data)))
            scaled_y_list.append(scaled_y)
            feature_scaler_list.append(scaler_y)
        Y_df[f] = np.hstack(scaled_y_list)
        scaler_list.append(feature_scaler_list)

    Y_train_df = Y_df[Y_df['ds']<ts_per_patient-len_test].reset_index(drop=True)
    Y_train_df['ds'] = pd.to_datetime(Y_train_df['ds'])
    Y_df['ds'] = pd.to_datetime(Y_df['ds'])

    # ------------------------------------------------------- HYPERPARAMATER TUNNING -------------------------------------------------------
    trials = hyperopt_tunning(space=nbeats_space, hyperopt_iters=n_trials, loss_function=mae, Y_df=Y_train_df, X_df=None, S_df=None,
                              ds_in_test=len_validation, shuffle_outsample=False)

    # Best mc
    mc = trials.trials[np.argmin(trials.losses())]['result']['mc']

    # ------------------------------------------------------- RUN IN TEST ---------------------------------------------------------------
    # Model
    final_nbeats = instantiate_nbeats(mc)

    # Datasets
    # Train and val
    train_ts_dataset, validation_ts_dataset, scaler_y = create_datasets(mc=mc, Y_df=Y_train_df, X_df=None, S_df=None,
                                                                        ds_in_test=horizon, shuffle_outsample=False)
    # Test
    test_mask_df = get_default_mask_df(Y_df=Y_df, ds_in_test=len_test, is_test=True)
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

    final_nbeats.fit(train_ts_loader=train_ts_loader, val_ts_loader=val_ts_loader, verbose=True,
                    eval_freq=mc['eval_freq'])

    y_true_test, y_hat_test, mask_test = final_nbeats.predict(ts_loader=test_ts_loader, return_decomposition=False)

    print('y_true_test.shape', y_true_test.shape)
    print('y_hat_test.shape', y_hat_test.shape)

    # De-scaling
    for f, scalers in enumerate(scaler_list):
        for p, scaler in enumerate(scalers): # len(y_true) has number of patients    
            y_true_test[p, :, f, :] = scaler.inv_scale(y_true_test[p, :, f, :])
            y_hat_test[p, :, f, :] = scaler.inv_scale(y_hat_test[p, :, f, :])

    result = {'horizon': horizon, 'trials': trials, 'y_true':y_true_test, 'y_hat':y_hat_test}
    with open(f'./results/result_{model}_{horizon}_{feature}_{pooling}_{args.experiment_id}.p', "wb") as f:
        pickle.dump(result, f)


def main(args):
    # Read data
    data = pd.read_csv('./data/healthcare/data_waveforms_icu.csv')

    # Filter patients with exploding PLETH    
    aux = data[['unique_id','PLETH']].groupby('unique_id').min().reset_index()
    aux = aux[aux['PLETH']>0]
    filter_patients = aux.unique_id.unique()
    data = data[data['unique_id'].isin(filter_patients)].reset_index(drop=True)

    #horizons = [30, 60, 120, 240, 480, 960, 1200]
    horizons = [30, 60, 120, 240]
    #horizons = [480, 960]

    len_validation = 5*250
    len_test = 5*250
    for horizon in horizons:
        print(100*'-')
        print(100*'-')
        print('HORIZON: ', horizon)
        evaluate_horizon(model=args.model, horizon=horizon, len_validation=len_validation, len_test=len_test,
                         data=data, n_trials=args.hyperopt_iters, feature=args.feature, pooling=args.pooling)
  
        print(100*'-')
        print(100*'-')

    # result = {'horizons': horizons, 'y_true':y_true_list, 'y_hat':y_hat_list, 'mae': mae_list, 'rmse': rmse_list}
    # with open(f'./results/result_{args.feature}_{args.pooling}_{args.experiment_id}.p', "wb") as f:
    #     pickle.dump(result, f)

def parse_args():
    desc = "707"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, help='model')
    parser.add_argument('--feature', type=str, help='feature')
    parser.add_argument('--horizon', type=int, help='horizon')
    parser.add_argument('--pooling', type=int, help='pooling')
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
# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts_papers/707_hyperopt_nbeats.py --model 'nbeats_g' --feature 'ART' --horizon 1500 --pooling 1 --hyperopt_iters 50 --experiment_id "20210505"

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate riemann
# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts_papers/707_hyperopt_nbeats.py --model 'nbeats_g' --feature 'PLETH' --horizon 1500 --pooling 1 --hyperopt_iters 50 --experiment_id "20210505"

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate riemann
# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts_papers/707_hyperopt_nbeats.py --model 'nbeats_g' --feature 'ART' --horizon 1500 --pooling 0 --hyperopt_iters 50 --experiment_id "20210505"

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate riemann
# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts_papers/707_hyperopt_nbeats.py --model 'nbeats_g' --feature 'PLETH' --horizon 1500 --pooling 0 --hyperopt_iters 50 --experiment_id "20210505"

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate riemann
# CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts_papers/707_hyperopt_nbeats.py --model 'nbeats_g' --feature 'BOTH' --horizon 1500 --pooling 1 --hyperopt_iters 50 --experiment_id "20210505"

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate riemann
# CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts_papers/707_hyperopt_nbeats.py --model 'nbeats_g' --feature 'BOTH' --horizon 1500 --pooling 0 --hyperopt_iters 50 --experiment_id "20210505"






# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate riemann
# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts_papers/707_hyperopt_nbeats.py --model 'nbeats_g' --feature 'ART' --horizon 1500 --pooling 1 --hyperopt_iters 50 --experiment_id "20210505_2"

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate riemann
# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts_papers/707_hyperopt_nbeats.py --model 'nbeats_g' --feature 'PLETH' --horizon 1500 --pooling 1 --hyperopt_iters 50 --experiment_id "20210505_2"

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate riemann
# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts_papers/707_hyperopt_nbeats.py --model 'nbeats_g' --feature 'ART' --horizon 1500 --pooling 0 --hyperopt_iters 50 --experiment_id "20210505_2"

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate riemann
# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts_papers/707_hyperopt_nbeats.py --model 'nbeats_g' --feature 'PLETH' --horizon 1500 --pooling 0 --hyperopt_iters 50 --experiment_id "20210505_2"

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate riemann
# CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts_papers/707_hyperopt_nbeats.py --model 'nbeats_g' --feature 'BOTH' --horizon 1500 --pooling 1 --hyperopt_iters 50 --experiment_id "20210505_2"

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate riemann
# CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts_papers/707_hyperopt_nbeats.py --model 'nbeats_g' --feature 'BOTH' --horizon 1500 --pooling 0 --hyperopt_iters 50 --experiment_id "20210505_2"
