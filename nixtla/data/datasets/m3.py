# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/datasets_m3.ipynb (unless otherwise specified).

__all__ = ['SOURCE_URL', 'Yearly', 'Quarterly', 'Monthly', 'Other', 'M3Info', 'M3']

# Cell
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .utils import download_file, Info, TimeSeriesDataclass
from ..ts_dataset import TimeSeriesDataset

# Cell
SOURCE_URL = 'https://forecasters.org/data/m3comp/M3C.xls'

# Cell
@dataclass
class Yearly:
    seasonality: int = 1
    horizon: int = 6
    freq: str = 'Y'
    sheet_name: str = 'M3Year'
    name: str = 'Yearly'

@dataclass
class Quarterly:
    seasonality: int = 4
    horizon: int = 8
    freq: str = 'Q'
    sheet_name: str = 'M3Quart'
    name: str = 'Quarterly'

@dataclass
class Monthly:
    seasonality: int = 12
    horizon: int = 18
    freq: str = 'M'
    sheet_name: str = 'M3Month'
    name: str = 'Monthly'

@dataclass
class Other:
    seasonality: int = 1
    horizon: int = 8
    freq: str = 'D'
    sheet_name: str = 'M3Other'
    name: str = 'Other'

# Cell
M3Info = Info(groups=('Yearly', 'Quarterly', 'Monthly', 'Other'),
              class_groups=(Yearly, Quarterly, Monthly, Other))

# Internal Cell
def _return_year(ts):
    year = ts.iloc[0]
    year = year if year != 0 else 1970

    return year

# Cell
@dataclass
class M3(TimeSeriesDataclass):

    @staticmethod
    def load(directory: str,
             group: str,
             training: bool = True,
             return_tensor: bool = True) -> Union[TimeSeriesDataset, TimeSeriesDataclass]:
        """
        Downloads and loads M3 data.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        group: str
            Group name.
            Allowed groups: 'Yearly', 'Quarterly', 'Monthly', 'Other'.
        training: bool
            Wheter return training or testing data. Default True.
        return_tensor: bool
            Wheter return TimeSeriesDataset (tensors, True) or
            TimeSeriesDataclass (dataframes)
        """
        path = Path(directory) / 'm3' / 'datasets'

        M3.download(directory)

        class_group = M3Info.get_group(group)

        df = pd.read_excel(path / 'M3C.xls', sheet_name=class_group.sheet_name)

        df = df.rename(columns={'Series': 'unique_id',
                                'Category': 'category',
                                'Starting Year': 'year',
                                'Starting Month': 'month'})

        df['unique_id'] = [class_group.name[0] + str(i + 1) for i in range(len(df))]
        S = df.filter(items=['unique_id', 'category'])

        id_vars = list(df.columns[:6])
        df = pd.melt(df, id_vars=id_vars, var_name='ds', value_name='y')
        df = df.dropna().sort_values(['unique_id', 'ds']).reset_index(drop=True)

        freq = pd.tseries.frequencies.to_offset(class_group.freq)

        if group == 'Other':
            df['year'] = 1970

        df['ds'] = df.groupby('unique_id')['year'] \
                     .transform(lambda df: pd.date_range(f'{_return_year(df)}-01-01',
                                                         periods=df.shape[0],
                                                         freq=freq))

        df = df.filter(items=['unique_id', 'ds', 'y'])

        if training:
            df = df.groupby('unique_id').apply(lambda df: df.head(-class_group.horizon)).reset_index(drop=True)
        else:
            df = df.groupby('unique_id').tail(class_group.horizon)
            df['ds'] = df.groupby('unique_id').cumcount() + 1

        if return_tensor:
            S['category'] = S['category'].astype('category').cat.codes
            return TimeSeriesDataset(y_df=df, X_s_df=S, X_t_df=None)
        else:
            return TimeSeriesDataclass(Y=df, S=S, X=None, idx_categorical_static=[0], group=group)

    @staticmethod
    def download(directory: Path) -> None:
        """Download M3 Dataset."""
        path = Path(directory) / 'm3' / 'datasets'
        if not path.exists():
            download_file(path, SOURCE_URL)