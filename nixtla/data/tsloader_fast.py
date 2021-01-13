# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/data__tsloader_fast.ipynb (unless otherwise specified).

__all__ = ['TimeSeriesLoader']

# Cell
import numpy as np
import pandas as pd
import random
import torch as t
import copy
from fastcore.foundation import patch
from .tsdataset import TimeSeriesDataset
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
                 is_train_loader: bool,
                 shuffle:bool,
                 random_seed: int):
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
        self.shuffle = shuffle # Boolean to shuffle data, useful for validation
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        assert offset==0, 'sample_mask and offset interaction not implemented'
        assert window_sampling_limit==self.ts_dataset.max_len, \
            'sample_mask and window_samplig_limit interaction not implemented'

        # Create rolling window matrix in advanced for faster access to data and broadcasted s_matrix
        self._create_sample_data()

    def _update_sampling_windows_idxs(self):
        # Only sample during available windows with at least one active output mask and input mask
        #n_windows, n_channels, max_len
        available_condition = t.sum(self.ts_windows[:, self.t_cols.index('available_mask'), :self.input_size], axis=1)
        sample_condition = t.sum(self.ts_windows[:, self.t_cols.index('sample_mask'), -self.output_size:], axis=1)

        #sampling_idx = t.nonzero(available_condition * sample_condition > 0)
        sampling_idx = t.nonzero(sample_condition)

        sampling_idx = list(sampling_idx.flatten().numpy())
        assert len(sampling_idx)>0, 'Check the data and masks as sample_idxs are empty'
        return sampling_idx

    def _create_windows_tensor(self):
        """
        Comment here
        TODO: Cuando creemos el otro dataloader, si es compatible lo hacemos funcion transform en utils
        """
        # Memory efficiency is gained from keeping across dataloaders common ts_tensor in dataset
        # Filter function is used to define train tensor and validation tensor with the offset
        # Default ts_idxs=ts_idxs sends all the data
        tensor, right_padding = self.ts_dataset.get_filtered_ts_tensor(offset=self.offset, output_size=self.output_size,
                                                                       window_sampling_limit=self.window_sampling_limit)
        tensor = t.Tensor(tensor)

        # Outsample mask checks existance of values in ts, train_mask mask is used to filter out validation
        # is_train_loader inverts the train_mask in case the dataloader is in validation mode
        # ###########
        # ###########
        # ###########
        # markets = ['BE', 'FR', 'NP', 'PJM']
        # for idx, market in enumerate(markets):
        #     print("\n")
        #     available_mask = tensor[idx, self.ts_dataset.t_cols.index('available_mask'), :]
        #     sample_mask = tensor[idx, self.ts_dataset.t_cols.index('sample_mask'), :]
        #     train_mask = available_mask * sample_mask
        #     n_hours = len(available_mask)
        #     print("available_mask.shape", available_mask.shape)

        #     print(f'LOADER {market} Available Mask {t.sum(available_mask/n_hours)}')
        #     print(f'LOADER {market} Sample Mask {t.sum(sample_mask/n_hours)}')
        #     print(f'LOADER {market} Train Mask {t.sum(train_mask/n_hours)}')
        #     ###########
        #     ###########
        #     ###########

        if self.is_train_loader:
            tensor[:, self.t_cols.index('sample_mask'), :] = \
                (tensor[:, self.t_cols.index('available_mask'), :] * tensor[:, self.t_cols.index('sample_mask'), :])
        else:
            tensor[:, self.t_cols.index('sample_mask'), :] = (1-tensor[:, self.t_cols.index('sample_mask'), :])

        padder = t.nn.ConstantPad1d(padding=(self.input_size, right_padding), value=0)
        tensor = padder(tensor)

        # Creating rolling windows and 'flattens' them
        windows = tensor.unfold(dimension=-1, size=self.input_size + self.output_size, step=self.idx_to_sample_freq)
        # n_serie, n_channel, n_time, window_size -> n_serie, n_time, n_channel, window_size
        #print(f'n_serie, n_channel, n_time, window_size = {windows.shape}')
        windows = windows.permute(0,2,1,3)
        #print(f'n_serie, n_time, n_channel, window_size = {windows.shape}')
        windows = windows.reshape(-1, self.ts_dataset.n_channels, self.input_size + self.output_size)

        # Broadcast s_matrix: This works because unfold in windows_tensor, orders: time, serie
        s_matrix = self.ts_dataset.s_matrix.repeat(repeats=int(len(windows)/self.ts_dataset.n_series), axis=0)

        return windows, s_matrix, tensor

    def __len__(self):
        return len(self.len_series)

    def __iter__(self):
        if self.shuffle:
            sample_idxs = np.random.choice(a=self.windows_sampling_idx,
                                           size=len(self.windows_sampling_idx), replace=False)
        else:
            sample_idxs = self.windows_sampling_idx

        n_batches = int(np.ceil(len(sample_idxs) / self.batch_size)) # Must be multiple of batch_size for paralel gpu

        for idx in range(n_batches):
            ws_idxs = sample_idxs[(idx * self.batch_size) : (idx + 1) * self.batch_size]
            batch = self.__get_item__(index=ws_idxs)
            yield batch

    def __get_item__(self, index):
        if self.model == 'nbeats':
            return self._windows_batch(index)
        elif self.model == 'esrnn':
            #return self._full_series_batch(index)
            assert 1<0, 'Hierarchical sampling not implemented'
        else:
            assert 1<0, 'error'

    def _windows_batch(self, index):
        """ NBEATS, TCN models """

        # Access precomputed rolling window matrix (RAM intensive)
        windows = self.ts_windows[index]
        s_matrix = self.s_matrix[index]

        insample_y = windows[:, self.t_cols.index('y'), :self.input_size]
        insample_x = windows[:, (self.t_cols.index('y')+1):self.t_cols.index('available_mask'), :self.input_size]
        available_mask = windows[:, self.t_cols.index('available_mask'), :self.input_size]

        outsample_y = windows[:, self.t_cols.index('y'), self.input_size:]
        outsample_x = windows[:, (self.t_cols.index('y')+1):self.t_cols.index('available_mask'), self.input_size:]
        sample_mask = windows[:, self.t_cols.index('sample_mask'), self.input_size:]

        batch = {'s_matrix': s_matrix,
                 'insample_y': insample_y, 'insample_x':insample_x, 'insample_mask':available_mask,
                 'outsample_y': outsample_y, 'outsample_x':outsample_x, 'outsample_mask':sample_mask}
        return batch

    # def _full_series_batch(self, index):
    #     """ ESRNN, RNN models """

    #     print("[index]", index)
    #     print("self.ts_tensor.shape", self.ts_tensor.shape)

    #     ts_tensor = self.ts_tensor[index]

    #     # Trim batch to shorter time series to avoid zero padding
    #     insample_y = ts_tensor[:, self.t_cols.index('y'), :]
    #     batch_len_series = np.array(self.ts_dataset.len_series)[index]
    #     min_batch_len = np.min(batch_len_series)
    #     insample_y = insample_y[:, -min_batch_len:]

    #     insample_x = ts_tensor[:, self.t_cols.index('y')+1:self.t_cols.index('insample_mask'), :]
    #     insample_x = insample_x[:, -min_batch_len:]

    #     s_matrix = self.ts_dataset.s_matrix[index]

    #     batch = {'insample_y': insample_y, 'idxs': index, 'insample_x': insample_x, 's_matrix': s_matrix}

    #     return batch

    def _create_sample_data(self):
        """
        """
        # print('Creating windows matrix ...')
        # Create rolling window matrix for fast information retrieval
        self.ts_windows, self.s_matrix, _ = self._create_windows_tensor()
        self.n_windows = len(self.ts_windows)
        self.windows_sampling_idx = self._update_sampling_windows_idxs()

        #expected_windows = self.ts_dataset.n_trn if self.is_train_loader else self.ts_dataset.n_prd
        #assert expected_windows == (len(self.windows_sampling_idx) * self.idx_to_sample_freq), \
        #    f'Check predict windows {self.ts_dataset.n_trn} sample windows {len(self.windows_sampling_idx)}'
        assert (self.ts_dataset.n_prd % self.ts_dataset.n_series == 0), 'Predictions tensor is unbalanced'
        assert (self.ts_dataset.n_prd % self.idx_to_sample_freq == 0), 'Predictions tensor is unbalanced'

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