# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/datasets_m4.ipynb (unless otherwise specified).

__all__ = ['SOURCE_URL', 'Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly', 'Other', 'M4Info', 'M4']

# Cell
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .utils import download_file, Info, TimeSeriesDataclass
from ..ts_dataset import TimeSeriesDataset

# Cell
SOURCE_URL = 'https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/'

# Cell
@dataclass
class Yearly:
    seasonality: int = 1
    horizon: int = 6
    freq: str = 'Y'
    name: str = 'Yearly'
    n_ts: int = 23_000

@dataclass
class Quarterly:
    seasonality: int = 4
    horizon: int = 8
    freq: str = 'Q'
    name: str = 'Quarterly'
    n_ts: int = 24_000

@dataclass
class Monthly:
    seasonality: int = 12
    horizon: int = 18
    freq: str = 'M'
    name: str = 'Monthly'
    n_ts: int = 48_000

@dataclass
class Weekly:
    seasonality: int = 52
    horizon: int = 13
    freq: str = 'W'
    name: str = 'Weekly'
    n_ts: int = 359

@dataclass
class Daily:
    seasonality: int = 7
    horizon: int = 14
    freq: str = 'D'
    name: str = 'Daily'
    n_ts: int = 4_227

@dataclass
class Hourly:
    seasonality: int = 24
    horizon: int = 48
    freq: str = 'H'
    name: str = 'Hourly'
    n_ts: int = 414


@dataclass
class Other:
    seasonality: int = 1
    horizon: int = 8
    freq: str = 'D'
    name: str = 'Other'
    n_ts: int = 5_000
    included_groups: Tuple = ('Weekly', 'Daily', 'Hourly')

# Cell
M4Info = Info(groups=('Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly', 'Other'),
              class_groups=(Yearly, Quarterly, Monthly, Weekly, Daily, Hourly, Other))

# Cell
@dataclass
class M4(TimeSeriesDataclass):

    @staticmethod
    def load(directory: str,
             group: str,
             return_tensor: bool = True) -> Union[TimeSeriesDataset, TimeSeriesDataclass]:
        """
        Downloads and loads M4 data.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        group: str
            Group name.
            Allowed groups: 'Yearly', 'Quarterly', 'Monthly',
                            'Weekly', 'Daily', 'Hourly'.
        return_tensor: bool
            Wheter return TimeSeriesDataset (tensors, True) or
            TimeSeriesDataclass (dataframes)

        Notes
        -----
        [1] Returns train+test sets.
        """
        if group == 'Other':
            #Special case.
            included_dfs = [M4.load(directory, gr, False).Y \
                            for gr in M4Info['Other'].included_groups]
            df = pd.concat(included_dfs)
        else:
            path = Path(directory) / 'm4' / 'datasets'

            M4.download(directory)

            class_group = M4Info[group]

            def read_and_melt(file):
                df = pd.read_csv(file)
                df.columns = ['unique_id'] + list(range(1, df.shape[1]))
                df = pd.melt(df, id_vars=['unique_id'], var_name='ds', value_name='y')
                df = df.dropna()

                return df

            df_train = read_and_melt(path / f'{group}-train.csv')
            df_test = read_and_melt(path / f'{group}-test.csv')

            len_train = df_train.groupby('unique_id').agg({'ds': 'max'}).reset_index()
            len_train.columns = ['unique_id', 'len_serie']
            df_test = df_test.merge(len_train, on=['unique_id'])
            df_test['ds'] = df_test['ds'] + df_test['len_serie']
            df_test.drop('len_serie', axis=1, inplace=True)

            df = pd.concat([df_train, df_test])
            df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

#         print(df)

#         freq = pd.tseries.frequencies.to_offset(class_group.freq)

#         if group == 'Other':
#             df['year'] = 1970

#         df['ds'] = df.groupby('unique_id')['year'] \
#                      .transform(lambda df: pd.date_range(f'{_return_year(df)}-01-01',
#                                                          periods=df.shape[0],
#                                                          freq=freq))

#         df = df.filter(items=['unique_id', 'ds', 'y'])

        if return_tensor:
            #S['category'] = S['category'].astype('category').cat.codes
            return TimeSeriesDataset(Y_df=df, S_df=None, X_df=None)
        else:
            return TimeSeriesDataclass(Y=df, S=None, X=None, group=group)

    @staticmethod
    def download(directory: Path) -> None:
        """Download M4 Dataset."""
        path = Path(directory) / 'm4' / 'datasets'
        if not path.exists():
            for group in M4Info.groups:
                download_file(path, f'{SOURCE_URL}/Train/{group}-train.csv')
                download_file(path, f'{SOURCE_URL}/Test/{group}-test.csv')
            download_file(path, f'{SOURCE_URL}/M4-info.csv')