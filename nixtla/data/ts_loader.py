# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/loader_class.ipynb (unless otherwise specified).

__all__ = ['TimeSeriesLoader']

# Cell
import numpy as np
import pandas as pd
import random
import torch as t
import copy
from fastcore.foundation import patch
from .ts_dataset import TimeSeriesDataset
from collections import defaultdict

# Cell
# TODO: pensar variable shuffle para compatibilidad con dataloader de pytorch y keras
#.      por el momento tenemos solo validacion boostrapeada, no existe modo no shuffle
#.      para evaluacion no estocástica, nuestra validación está hackeada.
class TimeSeriesLoader(object):
    def __init__(self,
                 ts_dataset:TimeSeriesDataset,
                 model:str,
                 offset:int,
                 window_sampling_limit: int,
                 input_size: int,
                 output_size: int,
                 idx_to_sample_freq: int, # TODO: Usada en hack ENORME para window frequency sampling
                 batch_size: int,
                 is_train_loader: bool):
        """
        """
        # Dataloader attributes
        self.model = model
        self.window_sampling_limit = window_sampling_limit
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.idx_to_sample_freq = idx_to_sample_freq
        self.offset = offset
        self.ts_dataset = ts_dataset
        self.t_cols = self.ts_dataset.t_cols
        self.is_train_loader = is_train_loader # Boolean variable for train and validation mask

        # Create rolling window matrix in advanced for faster access to data and broadcasted s_matrix
        self._create_train_data()
        self._is_train = True # Boolean variable for train and eval mode for dataloader (random vs ordered batches)

        #TODO: mejorar estos prints
        # print('X: time series features, of shape (#series,#times,#features): \t' + str(X.shape))
        # print('Y: target series (in X), of shape (#series,#times): \t \t' + str(Y.shape))
        # print('S: static features, of shape (#series,#features): \t \t' + str(S.shape))

    def _update_sampling_windows_idxs(self):
        # Only sample during training windows with at least one active output mask
        sampling_idx = t.sum(self.ts_windows[:, self.t_cols.index('outsample_mask'), -self.output_size:], axis=1)
        sampling_idx = t.nonzero(sampling_idx > 0)
        sampling_idx = list(sampling_idx.flatten().numpy())
        # TODO: pensar como resolver el hack de +1,
        #.      el +1 está diseñado para addressear el shift que tenemos que garantiza que el primer train tenga
        #.      por lo menos un input en la train_mask, además este código necesita la condición de que la serie más larga empieza
        #.      en el ds del que se va a querer samplear con la frecuencia particular. Hay dos hacks ENORMES.
        sampling_idx = [idx for idx in sampling_idx if (idx+1) % self.idx_to_sample_freq==0] # TODO: Esta linea muy malvada aumenta .6 segundos
        return sampling_idx

    def _create_windows_tensor(self):
        """
        Comment here
        TODO: Cuando creemos el otro dataloader, si es compatible lo hacemos funcion transform en utils
        """
        # Memory efficiency is gained from keeping across dataloaders common ts_tensor in dataset
        # Filter function is used to define train tensor and validation tensor with the offset
        # Default ts_idxs=ts_idxs sends all the data
        tensor, right_padding, train_mask = self.ts_dataset.get_filtered_ts_tensor(offset=self.offset, output_size=self.output_size,
                                                                                   window_sampling_limit=self.window_sampling_limit)
        tensor = t.Tensor(tensor)

        # Outsample mask checks existance of values in ts, train_mask mask is used to filter out validation
        # is_train_loader inverts the train_mask in case the dataloader is in validation mode
        mask = train_mask if self.is_train_loader else (1 - train_mask)
        tensor[:, self.t_cols.index('outsample_mask'), :] = tensor[:, self.t_cols.index('outsample_mask'), :] * mask

        padder = t.nn.ConstantPad1d(padding=(self.input_size-1, right_padding), value=0)
        tensor = padder(tensor)

        # Last output_size outsample_mask and y to 0
        tensor[:, self.t_cols.index('y'), -self.output_size:] = 0 # overkill to ensure no validation leakage
        tensor[:, self.t_cols.index('outsample_mask'), -self.output_size:] = 0

        # Creating rolling windows
        windows = tensor.unfold(dimension=-1, size=self.input_size + self.output_size, step=1)
        windows = windows.permute(2,0,1,3)
        windows = windows.reshape(-1, self.ts_dataset.n_channels, self.input_size + self.output_size)
        return windows

    def __len__(self):
        return len(self.len_series)

    def __iter__(self):
        #TODO: revisar como se hace el -1 de batch_size en un dataloader de torch. Otra opcion es simplemente batch_size grande,
        # tambien se puede arregar con epoca
        while True:
            if self._is_train:
                if self.batch_size > 0:
                    sampled_ts_indices = np.random.choice(self.windows_sampling_idx, size=self.batch_size, replace=True)
                else:
                    sampled_ts_indices = self.windows_sampling_idx
            else:
                # Get last n_series windows, dataset is ordered because of unfold
                sampled_ts_indices = list(range(self.n_windows-self.ts_dataset.n_series, self.n_windows))

            batch = self.__get_item__(sampled_ts_indices)

            yield batch

    def __get_item__(self, index):
        if self.model == 'nbeats':
            return self._nbeats_batch(index)
        elif self.model == 'esrnn':
            assert 1<0, 'hacer esrnn'
        else:
            assert 1<0, 'error'

    def _nbeats_batch(self, index):
        # Access precomputed rolling window matrix (RAM intensive)
        windows = self.ts_windows[index]
        s_matrix = self.s_matrix[index]

        insample_y = windows[:, self.t_cols.index('y'), :self.input_size]
        insample_x = windows[:, (self.t_cols.index('y')+1):self.t_cols.index('insample_mask'), :self.input_size]
        insample_mask = windows[:, self.t_cols.index('insample_mask'), :self.input_size]

        outsample_y = windows[:, self.t_cols.index('y'), self.input_size:]
        outsample_x = windows[:, (self.t_cols.index('y')+1):self.t_cols.index('insample_mask'), self.input_size:]
        outsample_mask = windows[:, self.t_cols.index('outsample_mask'), self.input_size:]

        batch = {'s_matrix': s_matrix,
                 'insample_y': insample_y, 'insample_x':insample_x, 'insample_mask':insample_mask,
                 'outsample_y': outsample_y, 'outsample_x':outsample_x, 'outsample_mask':outsample_mask}
        return batch

    def _create_train_data(self):
        """
        """
        #print('Creating windows matrix ...')
        # Create rolling window matrix for fast information retrieval
        self.ts_windows = self._create_windows_tensor()
        self.n_windows = len(self.ts_windows)
        # Broadcast s_matrix: This works because unfold in windows_tensor, padded windows, unshuffled data.
        self.s_matrix = self.ts_dataset.s_matrix.repeat(int(self.n_windows/self.ts_dataset.n_series), 1)
        self.windows_sampling_idx = self._update_sampling_windows_idxs()

    def update_offset(self, offset):
        if offset == self.offset:
            return # Avoid extra computation
        self.offset = offset
        self._create_train_data()

    def get_meta_data_col(self, col):
        return self.ts_dataset.get_meta_data_col(col)

    def get_n_variables(self):
        return self.ts_dataset.n_x, self.ts_dataset.n_s

    def get_n_series(self):
        return self.ts_dataset.n_series

    def get_max_len(self):
        return self.ts_dataset.max_len

    def get_n_channels(self):
        return self.ts_dataset.n_channels

    def get_X_cols(self):
        return self.ts_dataset.X_cols

    def get_frequency(self):
        return self.ts_dataset.frequency

    def train(self):
        self._is_train = True

    def eval(self):
        self._is_train = False