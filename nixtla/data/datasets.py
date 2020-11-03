# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/datasets.ipynb (unless otherwise specified).

__all__ = ['SOURCE_URL', 'logger', 'download_file', 'TimeSeriesDataset', 'Yearly', 'Quarterly', 'Monthly',
           'TourismInfo', 'Tourism']

# Cell
import logging
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

SOURCE_URL = 'https://robjhyndman.com/data/27-3-Athanasopoulos1.zip'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cell
def download_file(directory: Union[str, Path], source_url: str, decompress: bool = False) -> None:
    """Download data from source_ulr inside directory.

    Parameters
    ----------
    directory: str, Path
        Custom directory where data will be downloaded.
    source_url: str
        URL where data is hosted.
    decompress: bool
        Wheter decompress downloaded file. Default False.
    """
    if isinstance(directory, str):
        directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    filename = source_url.split('/')[-1]
    filepath = directory / filename

    # Streaming, so we can iterate over the response.
    r = requests.get(source_url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte

    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filepath, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        logger.error('ERROR, something went wrong downloading data')

    size = filepath.stat().st_size
    logger.info(f'Successfully downloaded {filename}, {size}, bytes.')

    if decompress:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(directory)

        logger.info(f'Successfully decompressed {filepath}')

# Cell
@dataclass
class TimeSeriesDataset:
    """
    Args:
        S (pd.DataFrame): DataFrame of static features of shape
            (n_time_series, n_features).
        X (pd.DataFrame): DataFrame of exogenous variables of shape
            (sum n_periods_i for i=1..n_time_series, n_exogenous).
        Y (pd.DataFrame): DataFrame of target variable of shape
            (sum n_periods_i for i=1..n_time_series, 1).
        idx_categorical_static (list, optional): List of categorical indexes
            of S.
        groups (dict, optional): Dictionary of groups.
            Example: {'Yearly': ['Y1', 'Y2']}
    """
    S: pd.DataFrame
    X: pd.DataFrame
    Y: pd.DataFrame
    idx_categorical_static: Optional[List] = None
    groups: Optional[Dict[str, np.array]] = None

# Cell
@dataclass
class Yearly:
    seasonality: int = 1
    horizon: int = 4
    freq: str = 'D'
    rows: int = 2
    name: str = 'Yearly'

@dataclass
class Quarterly:
    seasonality: int = 4
    horizon: int = 8
    freq: str = 'Q'
    rows: int = 3
    name: str = 'Quarterly'

@dataclass
class Monthly:
    seasonality: int = 12
    horizon: int = 24
    freq: str = 'M'
    rows: int = 3
    name: str = 'Monthly'

@dataclass
class TourismInfo:
    groups: Tuple = (Yearly, Quarterly, Monthly)

# Cell
class Tourism(TimeSeriesDataset):

    @staticmethod
    def load(directory: str, training: bool = True, download: bool = True) -> 'Tourism':
        """
        Downloads and loads Tourism data.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        training: bool
            Wheter return training or testing data. Default True.
        download: bool
            Wheter download dataset first.
        """
        path = Path(directory) / 'tourism' / 'datasets'

        if download:
            Tourism.download(directory)

        data = []
        groups = {}

        for group in TourismInfo.groups:
            if training:
                file = path / f'{group.name.lower()}_in.csv'
            else:
                file = path / f'{group.name.lower()}_oos.csv'

            df = pd.read_csv(file)
            groups[group.name] = df.columns.values

            dfs = []
            for col in df.columns:
                df_col = df[col]
                length, year = df_col[:2].astype(int)
                skip_rows = group.rows

                df_col = df_col[skip_rows:length + skip_rows]
                df_col = df_col.rename('y').to_frame()
                df_col['unique_id'] = col
                freq = pd.tseries.frequencies.to_offset(group.name[0])
                df_col['ds'] = pd.date_range(f'{year}-01-01', periods=length, freq=freq)

                dfs.append(df_col)

            df = pd.concat(dfs)

            data.append(df)

        data = pd.concat(data).reset_index(drop=True)[['unique_id', 'ds', 'y']]

        return Tourism(Y=data, S=None, X=None, groups=groups)

    @staticmethod
    def download(directory: str) -> None:
        """Download Tourism Dataset."""
        path = Path(directory) / 'tourism' / 'datasets'
        if not path.exists():
            download_file(path, SOURCE_URL, decompress=True)

    def get_group(self, group: str) -> 'Tourism':
        """Filters group data.

        Parameters
        ----------
        group: str
            Group name.
        """
        assert group in self.groups, \
            f'Please provide a valid group: {", ".join(self.groups.keys())}'

        ids = self.groups[group]

        y = self.Y.query('unique_id in @ids')

        return Tourism(Y=y, S=None, X=None, groups={group: ids})