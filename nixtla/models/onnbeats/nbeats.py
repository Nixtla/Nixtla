# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/models_onnbeats__nbeats.ipynb (unless otherwise specified).

__all__ = ['accuracy_logits', 'Nbeats']

# Cell
import os
import time
import numpy as np
import pandas as pd
import random
from collections import defaultdict
import copy

import torch as t
from torch import optim
import torch.nn.functional as F
from pathlib import Path
from functools import partial

from .nbeats_model import NBeats, NBeatsBlock, IdentityBasis
from .nbeats_model import TrendBasis, SeasonalityBasis, ExogenousFutureBasis, ExogenousBasisInterpretable
from ...losses.pytorch import MAPELoss, MASELoss, SMAPELoss, MSELoss, MAELoss, RMSELoss
from ...losses.numpy import mae, mse, mape, smape, rmse#, accuracy

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
    metric_protections(y, y_hat, weights)

    y_hat = ((1/(1 + np.exp(-y_hat))) > thr) * 1
    accuracy = np.average(y_hat==y, weights=weights) * 100
    return (100-accuracy)

# Cell
class Nbeats(object):
    """
    Future documentation
    """
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    IDENTITY_BLOCK = 'identity'

    def __init__(self,
                 input_size_multiplier,
                 output_size,
                 shared_weights,
                 stack_types,
                 n_blocks,
                 n_layers,
                 n_hidden,
                 n_harmonics,
                 n_polynomials,
                 exogenous_n_channels,
                 f_cols,
                 theta_with_exogenous,
                 batch_normalization,
                 dropout_prob,
                 x_s_n_hidden,
                 learning_rate,
                 lr_decay,
                 n_lr_decay_steps,
                 l1_lambda_x,
                 weight_decay,
                 n_iterations,
                 early_stopping,
                 loss,
                 val_loss,
                 frequency,
                 seasonality,
                 random_seed,
                 device=None):
        super(Nbeats, self).__init__()

        self.input_size = int(input_size_multiplier*output_size)
        self.output_size = output_size
        self.shared_weights = shared_weights
        self.stack_types = stack_types
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_harmonics = n_harmonics
        self.n_polynomials = n_polynomials
        self.exogenous_n_channels = exogenous_n_channels
        self.f_cols = f_cols
        self.theta_with_exogenous = theta_with_exogenous
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob
        self.x_s_n_hidden = x_s_n_hidden
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.n_lr_decay_steps = n_lr_decay_steps
        self.l1_lambda_x = l1_lambda_x
        self.weight_decay = weight_decay
        self.n_iterations = n_iterations
        self.early_stopping = early_stopping
        self.loss = loss
        self.val_loss = val_loss
        self.frequency = frequency
        self.seasonality = seasonality
        self.random_seed = random_seed
        if device is None:
            device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.device = device

        self._is_instantiated = False

    def create_stack(self):
        # Declare parameter dimensions
        if self.theta_with_exogenous:
            x_t_n_inputs = self.input_size + self.n_x_t*self.output_size
        else:
            x_t_n_inputs = self.input_size # y_lags

        #------------------------ Model Definition ------------------------#
        block_list = []
        self.blocks_regularizer = []
        for i in range(len(self.stack_types)):
            #print(f'| --  Stack {self.stack_types[i]} (#{i})')
            for block_id in range(self.n_blocks[i]):
                # Batch norm only on first block
                if (len(block_list)==0) and (self.batch_normalization):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                # Dummy of regularizer in block. Override with 1 if exogenous_block
                #self.blocks_regularizer += [0]

                # Shared weights
                if self.shared_weights and block_id>0:
                    nbeats_block = block_list[-1]

                else:
                    if self.stack_types[i] == 'seasonality':
                        nbeats_block = NBeatsBlock(x_t_n_inputs = x_t_n_inputs,
                                                   x_s_n_inputs = self.n_x_s,
                                                   x_s_n_hidden= self.x_s_n_hidden,
                                                   theta_n_dim=4 * int(
                                                        np.ceil(self.n_harmonics / 2 * self.output_size) - (self.n_harmonics - 1)),
                                                   basis=SeasonalityBasis(harmonics=self.n_harmonics,
                                                                                backcast_size=self.input_size,
                                                                                forecast_size=self.output_size),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   theta_with_exogenous=self.theta_with_exogenous,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob)
                    elif self.stack_types[i] == 'trend':
                        nbeats_block = NBeatsBlock(x_t_n_inputs = x_t_n_inputs,
                                                   x_s_n_inputs = self.n_x_s,
                                                   x_s_n_hidden= self.x_s_n_hidden,
                                                   theta_n_dim=2 * (self.n_polynomials + 1),
                                                   basis=TrendBasis(degree_of_polynomial=self.n_polynomials,
                                                                            backcast_size=self.input_size,
                                                                            forecast_size=self.output_size),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   theta_with_exogenous=self.theta_with_exogenous,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob)
                    elif self.stack_types[i] == 'identity':
                        nbeats_block = NBeatsBlock(x_t_n_inputs = x_t_n_inputs,
                                                   x_s_n_inputs = self.n_x_s,
                                                   x_s_n_hidden= self.x_s_n_hidden,
                                                   theta_n_dim=self.input_size + self.output_size,
                                                   basis=IdentityBasis(backcast_size=self.input_size,
                                                                       forecast_size=self.output_size),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   theta_with_exogenous=self.theta_with_exogenous,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob)
                    elif self.stack_types[i] == 'exogenous':
                        nbeats_block = NBeatsBlock(x_t_n_inputs = x_t_n_inputs,
                                                   x_s_n_inputs = self.n_x_s,
                                                   x_s_n_hidden= self.x_s_n_hidden,
                                                   theta_n_dim=2*self.n_x_t,
                                                   basis=ExogenousBasisInterpretable(),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   theta_with_exogenous=self.theta_with_exogenous,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob)
                    elif self.stack_types[i] == 'exogenous_g_a':
                        assert len(self.f_cols)>0, 'If ExogenousFutureBasis, provide x_f_cols hyperparameter'
                        nbeats_block = NBeatsBlock(x_t_n_inputs = x_t_n_inputs,
                                                   x_s_n_inputs = self.n_x_s,
                                                   x_s_n_hidden= self.x_s_n_hidden,
                                                   theta_n_dim=2*(self.exogenous_n_channels),
                                                   basis=ExogenousFutureBasis(out_features=self.exogenous_n_channels,
                                                                              f_idxs=self.f_idxs),
                                                   n_layers=self.n_layers[i],
                                                   theta_n_hidden=self.n_hidden[i],
                                                   theta_with_exogenous=self.theta_with_exogenous,
                                                   batch_normalization=batch_normalization_block,
                                                   dropout_prob=self.dropout_prob)
                #        self.blocks_regularizer[-1] = 1
                #print(f'     | -- {nbeats_block}')
                block_list.append(nbeats_block)
        return block_list

    def __loss_fn(self, loss_name: str):
        def loss(x, freq, forecast, target, mask):
            if loss_name == 'MAPE':
                return MAPELoss(y=target, y_hat=forecast, mask=mask) + self.l1_regularization()
            elif loss_name == 'MASE':
                return MASELoss(y=target, y_hat=forecast, y_insample=x, seasonality=freq, mask=mask) + self.l1_regularization()
            elif loss_name == 'SMAPE':
                return SMAPELoss(y=target, y_hat=forecast, mask=mask) + self.l1_regularization()
            elif loss_name == 'MSE':
                return MSELoss(y=target, y_hat=forecast, mask=mask) + self.l1_regularization()
            elif loss_name == 'RMSE':
                return RMSELoss(y=target, y_hat=forecast, mask=mask) + self.l1_regularization()
            elif loss_name == 'MAE':
                return MAELoss(y=target, y_hat=forecast, mask=mask) + self.l1_regularization()
            elif loss_name == 'BCE':
                return F.binary_cross_entropy_with_logits(input=forecast, target=target) + \
                       self.loss_l1_conv_layers() + self.loss_l1_theta()
            else:
                raise Exception(f'Unknown loss function: {loss_name}')
        return loss

    def __val_loss_fn(self, loss_name: str):
        #TODO: mase not implemented
        def loss(forecast, target, weights):
            if loss_name == 'MAPE':
                return mape(y=target, y_hat=forecast, weights=weights) #TODO: faltan weights
            elif loss_name == 'SMAPE':
                return smape(y=target, y_hat=forecast, weights=weights) #TODO: faltan weights
            elif loss_name == 'MSE':
                return mse(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'RMSE':
                return rmse(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'MAE':
                return mae(y=target, y_hat=forecast, weights=weights)
            elif loss_name == 'ACCURACY':
                return accuracy_logits(y=target, y_hat=forecast, weights=weights)
            else:
                raise Exception(f'Unknown loss function: {loss_name}')
        return loss

    def l1_regularization(self):
        # l1_loss = 0
        # for i, indicator in enumerate(self.blocks_regularizer):
        #     if indicator:
        #         l1_loss +=  self.l1_lambda*t.sum(t.abs(self.model.blocks[i].basis.weight))
        # return l1_loss
        return self.l1_lambda_x * t.sum(t.abs(self.model.l1_weight))

    def to_tensor(self, x: np.ndarray) -> t.Tensor:
        tensor = t.as_tensor(x, dtype=t.float32).to(self.device)
        return tensor

    def evaluate_performance(self, ts_loader, validation_loss_fn):
        #TODO: mas opciones que mae
        self.model.eval()

        losses = []
        with t.no_grad():
            for batch in iter(ts_loader):
                insample_y     = self.to_tensor(batch['insample_y'])
                insample_x     = self.to_tensor(batch['insample_x'])
                insample_mask  = self.to_tensor(batch['insample_mask'])
                outsample_x    = self.to_tensor(batch['outsample_x'])
                outsample_y    = self.to_tensor(batch['outsample_y'])
                outsample_mask = self.to_tensor(batch['outsample_mask'])
                s_matrix       = self.to_tensor(batch['s_matrix'])

                forecast = self.model(insample_y=insample_y, insample_x_t=insample_x,
                                    insample_mask=insample_mask, outsample_x_t=outsample_x, x_s=s_matrix)
                batch_loss = validation_loss_fn(target=outsample_y.cpu().data.numpy(),
                                                forecast=forecast.cpu().data.numpy(),
                                                weights=outsample_mask.cpu().data.numpy())
                losses.append(batch_loss)
        loss = np.mean(losses)
        self.model.train()
        return loss

    def fit(self, train_ts_loader, val_ts_loader=None, n_iterations=None, verbose=True, eval_steps=1):
        # Asserts
        assert train_ts_loader.t_cols[0] == 'y', f'First variable must be y not {train_ts_loader.t_cols[0]}'
        assert train_ts_loader.t_cols[1] == 'ejecutado', f'First exogenous variable must be ejecutado not {train_ts_loader.t_cols[1]}'
        assert (self.input_size)==train_ts_loader.input_size, \
            f'model input_size {self.input_size} data input_size {train_ts_loader.input_size}'

        # Random Seeds (model initialization)
        t.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed) #TODO: interaccion rara con window_sampling de validacion

        # Attributes of ts_dataset
        self.n_x_t, self.n_x_s = train_ts_loader.get_n_variables()
        self.t_cols = train_ts_loader.t_cols
        self.f_idxs = train_ts_loader.ts_dataset.get_f_idxs(self.f_cols)

        # Instantiate model
        if not self._is_instantiated:
            block_list = self.create_stack()
            self.model = NBeats(blocks=t.nn.ModuleList(block_list), in_features=self.n_x_t).to(self.device)
            self._is_instantiated = True

        # Overwrite n_iterations and train datasets
        if n_iterations is None:
            n_iterations = self.n_iterations

        lr_decay_steps = n_iterations // self.n_lr_decay_steps
        if lr_decay_steps == 0:
            lr_decay_steps = 1

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_steps, gamma=self.lr_decay)

        training_loss_fn = self.__loss_fn(self.loss)
        validation_loss_fn = self.__val_loss_fn(self.loss) #Uses numpy losses

        if verbose and (n_iterations > 0):
            print('='*30+' Start fitting '+'='*30)
            print(f'Number of exogenous variables: {self.n_x_t}')
            print(f'Number of static variables: {self.n_x_s} , with dim_hidden: {self.x_s_n_hidden}')
            print(f'Number of iterations: {n_iterations}')
            print(f'Number of blocks: {len(self.model.blocks)}')

        #self.loss_dict = {} # Restart self.loss_dict
        start = time.time()
        self.trajectories = {'iteration':[],'train_loss':[], 'val_loss':[]}
        self.final_insample_loss = None
        self.final_outsample_loss = None

        # Training Loop
        best_val_loss = np.inf
        early_stopping_counter = 0
        best_state_dict = copy.deepcopy(self.model.state_dict())
        break_flag = False
        iteration = 0
        epoch = 0
        while (iteration < n_iterations) and (not break_flag):
            epoch +=1
            for batch in iter(train_ts_loader):
                iteration += 1
                if (iteration > n_iterations) or (break_flag):
                    continue
                self.model.train()

                insample_y     = self.to_tensor(batch['insample_y'])
                insample_x     = self.to_tensor(batch['insample_x'])
                insample_mask  = self.to_tensor(batch['insample_mask'])
                outsample_x    = self.to_tensor(batch['outsample_x'])
                outsample_y    = self.to_tensor(batch['outsample_y'])
                outsample_mask = self.to_tensor(batch['outsample_mask'])
                s_matrix       = self.to_tensor(batch['s_matrix'])

                optimizer.zero_grad()
                forecast = self.model(insample_y=insample_y, insample_x_t=insample_x,
                                    insample_mask=insample_mask, outsample_x_t=outsample_x, x_s=s_matrix)

                training_loss = training_loss_fn(x=insample_y, freq=self.seasonality, forecast=forecast,
                                                target=outsample_y, mask=outsample_mask)

                if np.isnan(float(training_loss)):
                    break

                training_loss.backward()
                t.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                lr_scheduler.step()
                if (iteration % eval_steps == 0):
                    display_string = 'Iteration: {}, Time: {:03.3f}, Insample {}: {:.5f}'.format(iteration,
                                                                                    time.time()-start,
                                                                                    self.loss,
                                                                                    training_loss.cpu().data.numpy())
                    self.trajectories['iteration'].append(iteration)
                    self.trajectories['train_loss'].append(training_loss.cpu().data.numpy())

                    if val_ts_loader is not None:
                        loss = self.evaluate_performance(val_ts_loader, validation_loss_fn=validation_loss_fn)
                        display_string += ", Outsample {}: {:.5f}".format(self.loss, loss)
                        self.trajectories['val_loss'].append(loss)

                        if self.early_stopping:
                            if loss < best_val_loss:
                                # Save current model if improves outsample loss
                                best_state_dict = copy.deepcopy(self.model.state_dict())
                                best_insample_loss = training_loss.cpu().data.numpy()
                                early_stopping_counter = 0
                                best_val_loss = loss
                            else:
                                early_stopping_counter += 1
                            if early_stopping_counter >= self.early_stopping:
                                break_flag = True

                    print(display_string)

                    self.model.train()

                if break_flag:
                    print(10*'-',' Stopped training by early stopping', 10*'-')
                    self.model.load_state_dict(best_state_dict)
                    break

        #End of fitting
        if n_iterations >0:
            self.final_insample_loss = training_loss.cpu().data.numpy() if not break_flag else best_insample_loss #This is batch!
            string = 'Iteration: {}, Time: {:03.3f}, Insample {}: {:.5f}'.format(iteration,
                                                                            time.time()-start,
                                                                            self.loss,
                                                                            self.final_insample_loss)
            if val_ts_loader is not None:
                self.final_outsample_loss = self.evaluate_performance(val_ts_loader, validation_loss_fn=validation_loss_fn)
                string += ", Outsample {}: {:.5f}".format(self.loss, self.final_outsample_loss)
            print(string)
            print('='*30+'End fitting '+'='*30)

    def predict(self, ts_loader, y_insample_tensor=None, X_test=None, eval_mode=False):
        self.model.eval()
        assert not ts_loader.shuffle, 'ts_loader must have shuffle as False.'

        forecasts = []
        outsample_ys = []
        outsample_masks = []
        with t.no_grad():
            counter = 0
            for batch in iter(ts_loader):
                if y_insample_tensor is None:
                    insample_y     = batch['insample_y']
                else:
                    insample_y     = y_insample_tensor[counter*ts_loader.batch_size:(counter+1)*ts_loader.batch_size, :]
                insample_y         = self.to_tensor(insample_y)
                insample_x         = self.to_tensor(batch['insample_x'])
                insample_mask      = self.to_tensor(batch['insample_mask'])
                outsample_x        = self.to_tensor(batch['outsample_x'])
                s_matrix           = self.to_tensor(batch['s_matrix'])

                forecast = self.model(insample_y=insample_y, insample_x_t=insample_x,
                                      insample_mask=insample_mask, outsample_x_t=outsample_x, x_s=s_matrix)
                forecasts.append(forecast.cpu().data.numpy())
                outsample_ys.append(batch['outsample_y'])
                outsample_masks.append(batch['outsample_mask'])
                counter += 1

        forecasts = np.vstack(forecasts)
        outsample_ys = np.vstack(outsample_ys)
        outsample_masks = np.vstack(outsample_masks)

        self.model.train()
        if eval_mode:
            return outsample_ys, forecasts, outsample_masks

        # Pandas wrangling
        frequency = ts_loader.get_frequency()
        unique_ids = ts_loader.get_meta_data_col('unique_id')
        last_ds = ts_loader.get_meta_data_col('last_ds') #TODO: ajustar of offset

        # Predictions for panel
        Y_hat_panel = pd.DataFrame(columns=['unique_id', 'ds'])
        for i, unique_id in enumerate(unique_ids):
            Y_hat_id = pd.DataFrame([unique_id]*self.output_size, columns=["unique_id"])
            ds = pd.date_range(start=last_ds[i], periods=self.output_size+1, freq=frequency)
            Y_hat_id["ds"] = ds[1:]
            Y_hat_panel = Y_hat_panel.append(Y_hat_id, sort=False).reset_index(drop=True)

        Y_hat_panel['y_hat'] = forecasts.flatten()

        if X_test is not None:
            Y_hat_panel = X_test.merge(Y_hat_panel, on=['unique_id', 'ds'], how='left')

        return Y_hat_panel


    def save(self, model_dir, model_id):

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_file = os.path.join(model_dir, f"model_{model_id}.model")
        print('Saving model to:\n {}'.format(model_file)+'\n')
        t.save({'model_state_dict': self.model.state_dict()}, model_file)

    def load(self, model_dir, model_id):

        model_file = os.path.join(model_dir, f"model_{model_id}.model")
        path = Path(model_file)

        assert path.is_file(), 'No model_*.model file found in this path!'

        print('Loading model from:\n {}'.format(model_file)+'\n')

        checkpoint = t.load(model_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)