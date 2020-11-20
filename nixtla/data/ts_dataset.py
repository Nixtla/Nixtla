# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/dataset_class.ipynb (unless otherwise specified).

__all__ = ['TimeSeriesDataset']

# Cell
import numpy as np
import pandas as pd
import random
import torch as t

from fastcore.foundation import patch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# Cell
class TimeSeriesDataset(Dataset):
    def __init__(self,
                 y_df,
                 X_t_df = None,
                 X_s_df = None):
        """
        """
        assert type(y_df) == pd.core.frame.DataFrame
        assert all([(col in y_df) for col in ['unique_id', 'ds', 'y']])
        if X_t_df is not None:
            assert type(X_t_df) == pd.core.frame.DataFrame
            assert all([(col in X_t_df) for col in ['unique_id', 'ds']])

        print('Processing dataframes ...')
        ts_data, x_s, self.meta_data, self.t_cols = self._df_to_lists(y_df=y_df, X_s_df=X_s_df, X_t_df=X_t_df)

        # Attributes
        self.n_series = len(ts_data)
        self.max_len = max([len(ts['y']) for ts in ts_data])
        self.n_channels = len(ts_data[0].values())
        self.frequency = pd.infer_freq(y_df.head()['ds']) #TODO: improve, can die with head

        self.n_x_t, self.n_s_t = 0, 0
        if X_t_df is not None:
            self.n_x_t = X_t_df.shape[1]-2 # 2 for unique_id and ds
        if X_s_df is not None:
            self.n_s_t = X_s_df.shape[1]-1 # 1 for unique_id

        print('Creating ts tensor ...')
        self.ts_tensor, self.x_s, self.len_series = self._create_tensor(ts_data, x_s)

    def _df_to_lists(self, y_df, X_s_df, X_t_df):
        """
        """
        unique_ids = y_df['unique_id'].unique()

        if X_t_df is not None:
            X_t_vars = [col for col in X_t_df.columns if col not in ['unique_id','ds']]
        else:
            X_t_vars = []

        if X_s_df is not None:
            X_s_vars = [col for col in X_s_df.columns if col not in ['unique_id']]
        else:
            X_s_vars = []

        ts_data = []
        x_s = []
        meta_data = []
        for i, u_id in enumerate(unique_ids):
            top_row = np.asscalar(y_df['unique_id'].searchsorted(u_id, 'left'))
            bottom_row = np.asscalar(y_df['unique_id'].searchsorted(u_id, 'right'))
            serie = y_df[top_row:bottom_row]['y'].values
            last_ds_i = y_df[top_row:bottom_row]['ds'].max()

            # Y values
            ts_data_i = {'y': serie}
            # X_t values
            for X_t_var in X_t_vars:
                serie =  X_t_df[top_row:bottom_row][X_t_var].values
                ts_data_i[X_t_var] = serie
            ts_data.append(ts_data_i)

            # Static data
            s_data_i = defaultdict(list)
            for X_s_var in X_s_vars:
                s_data_i[X_s_var] = X_s_df.loc[X_s_df['unique_id']==u_id, X_s_var].values
            x_s.append(s_data_i)

            # Metadata
            meta_data_i = {'unique_id': u_id,
                           'last_ds': last_ds_i}
            meta_data.append(meta_data_i)

        t_cols = ['y'] + X_t_vars + ['insample_mask', 'outsample_mask']

        return ts_data, x_s, meta_data, t_cols

    def _create_tensor(self, ts_data, x_s):
        """
        ts_tensor: n_series x n_channels x max_len
        """
        ts_tensor = np.zeros((self.n_series, self.n_channels + 2, self.max_len)) # +2 for the masks
        static_tensor = np.zeros((self.n_series, len(x_s[0])))

        #TODO: usar tcols.get_loc()
        len_series = []
        for idx in range(self.n_series):
            ts_idx = np.array(list(ts_data[idx].values()))
            ts_tensor[idx, :self.t_cols.index('insample_mask'), -ts_idx.shape[1]:] = ts_idx
            ts_tensor[idx, self.t_cols.index('insample_mask'), -ts_idx.shape[1]:] = 1 # TODO: pensar si sacar esto al loader
            ts_tensor[idx, self.t_cols.index('outsample_mask'), -ts_idx.shape[1]:] = 1 # TODO: pensar si sacar esto al loader
            static_tensor[idx, :] = list(x_s[idx].values())
            len_series.append(ts_idx.shape[1])

        # TODO: mover a loader, se puede mantener un numpy
        ts_tensor = t.Tensor(ts_tensor)
        static_tensor = t.Tensor(static_tensor)

        return ts_tensor, static_tensor, np.array(len_series)

    def get_meta_data_var(self, var):
        """
        """
        var_values = [x[var] for x in self.meta_data]
        return var_values

    def get_x_s(self):
        return self.x_s

    def get_filtered_tensor(self, offset, output_size, window_sampling_limit):
        """
        Esto te da todo lo que tenga el tensor, el futuro incluido esto orque se usa exogenoas del futuro
        La mascara se hace despues
        """
        last_outsample_ds = self.max_len - offset + output_size
        first_ds = max(last_outsample_ds - window_sampling_limit - output_size, 0)
        filtered_tensor = self.ts_tensor[:, :, first_ds:last_outsample_ds]
        right_padding = max(last_outsample_ds - self.max_len, 0) #To padd with zeros if there is "nothing" to the right
        return filtered_tensor, right_padding