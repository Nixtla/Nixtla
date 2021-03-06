# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/data_datasets__favorita.ipynb (unless otherwise specified).

__all__ = ['logger', 'check_nans', 'Favorita']

# Cell
import os
import re
import gc
import time
import logging

import shutil
from py7zr import unpack_7zarchive

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from .utils import (
    download_file,
    Info,
    TimeSeriesDataclass,
    create_calendar_variables,
    create_us_holiday_distance_variables,
)
from ..tsdataset import TimeSeriesDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)

# Cell
def check_nans(df):
    """ For data wrangling logs """
    n_rows = len(df)

    check_df = {'col': [], 'dtype': [], 'nan_prc': []}
    for col in df.columns:
        check_df['col'].append(col)
        check_df['dtype'].append(df[col].dtype)
        check_df['nan_prc'].append(df[col].isna().sum()/n_rows)

    check_df = pd.DataFrame(check_df)
    print("\n")
    print(f"dataframe n_rows {n_rows}")
    print(check_df)
    print("\n")

# Cell
class Favorita:

    # original data available from Kaggle directly
    # pip install kaggle --upgrade
    # kaggle competitions download -c favorita-grocery-sales-forecasting
    source_url = 'https://www.dropbox.com/s/fe4y1hnphb4ykpy/favorita-grocery-sales-forecasting.zip?dl=1'
    files = ['holidays_events.csv.7z', 'items.csv.7z', 'oil.csv.7z', 'sample_submission.csv.7z',
             'stores.csv.7z', 'test.csv.7z', 'train.csv.7z', 'transactions.csv.7z']

    @staticmethod
    def unzip(path):
        # Unzip Load, Price, Solar and Wind data
        # shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)
        for file in Favorita.files:
            filepath = f'{path}/{file}'
            #Archive(filepath).extractall(path)
            shutil.unpack_archive(filepath, path)
            logger.info(f'Successfully decompressed {filepath}')

    @staticmethod
    def download(directory: str) -> None:
        """Downloads Favorita Competition Dataset."""
        path = f'{directory}/favorita'
        if not os.path.exists(path):
            download_file(directory=path,
                          source_url=Favorita.source_url,
                          decompress=True)
            Favorita.unzip(path)

    @staticmethod
    def read_raw_data(directory):
        path = f'{directory}/favorita'

        #------------------------------ Read Data ------------------------------#
        # We avoid memory-intensive task of infering dtypes
        dtypes_dict = {'id': 'int32',
                       'date': 'str',
                       'item_nbr': 'int32',
                       'store_nbr': 'int8', # there are only 54 stores
                       'unit_sales': 'float32', # values beyond are f32 outliers
                       'onpromotion': 'boolean'}

        # We read once from csv then from feather (much faster)
        if not os.path.exists(f'{path}/train.feather'):
            train_df = pd.read_csv(f'{path}/train.csv',
                                   dtype=dtypes_dict,
                                   parse_dates=['date'])
            del train_df['id']
            train_df.reset_index(drop=True, inplace=True)
            train_df.to_feather(f'{path}/train.feather')
            print("saved train.csv to train.feather for fast access")

        items_df = pd.read_csv(f'{path}/items.csv')
        stores_df = pd.read_csv(f'{path}/stores.csv')

        # Test is avoided because y_true is unavailable
        train_df = pd.read_feather(f'{path}/train.feather')
        oil_df = pd.read_csv(f'{path}/oil.csv', parse_dates=['date'])
        holidays_df = pd.read_csv(f'{path}/holidays_events.csv', parse_dates=['date'])
        transactions_df = pd.read_csv(f'{path}/transactions.csv', parse_dates=['date'])

        return train_df, oil_df, items_df, stores_df, holidays_df, transactions_df

    @staticmethod
    def load(directory):

        # To prioritize comparability with benchmark models
        # We replicate the data processing from:
        # https://github.com/google-research/google-research/blob/master/tft/script_download_data.py

        Favorita.download(directory)
        temporal, oil, items, stores, holidays, transactions \
                                                = Favorita.read_raw_data(directory)

        #------------------------------ S Data Wrangling -----------------------------#
        start = time.time()

        # Transform the static variable strings to int for later fast one_hot
        encoder = LabelEncoder()
        items['family'] = encoder.fit_transform(items['family'].values)
        items['class']  = encoder.fit_transform(items['class'].values)
        items['perishable'] = encoder.fit_transform(items['perishable'].values)

        #stores['city']  = encoder.fit_transform(stores['city'].values)
        #stores['state'] = encoder.fit_transform(stores['state'].values)
        stores['type']  = encoder.fit_transform(stores['type'].values)
        stores['cluster']  = encoder.fit_transform(stores['cluster'].values)
        stores_statecity = stores[['store_nbr', 'city', 'state']]

        # Declare static variables dataframe and unique_id
        S_df = temporal[['item_nbr', 'store_nbr']].copy()
        S_df.drop_duplicates(inplace=True)

        S_df = S_df.merge(items,  on='item_nbr',  how='left')
        S_df = S_df.merge(stores, on='store_nbr', how='left')
        S_df['traj_id'] = S_df.index

        # Create one_hot variables,
        store_nbr_orig = S_df['store_nbr'].values # used to merge with holidays
        S_df = pd.get_dummies(data=S_df,
                              columns=['family', 'class', 'perishable',
                                       'store_nbr', 'city', 'state', 'type', 'cluster'])
        S_df['store_nbr'] = store_nbr_orig

        # # Avoid item_nbr dummies because length would be 4096
        # # Substitute item_nbr/store_nbr the mean log(unit_sales) per item_nbr/store_nbr
        # y_item = temporal.groupby('item_nbr', as_index=False).y.mean() \
        #                                             .rename(columns={'y':'y_item'})
        # y_store = temporal.groupby('store_nbr', as_index=False).y.mean() \
        #                                             .rename(columns={'y':'y_store'})
        # y_itemstore = temporal.groupby(['item_nbr', 'store_nbr'], as_index=False).y.mean() \
        #                                                 .rename(columns={'y':'y_itemstore'})

        # S_df = S_df.merge(y_item, on='item_nbr', how='left')
        # S_df = S_df.merge(y_store, on='store_nbr', how='left')
        # S_df = S_df.merge(y_itemstore, on=['item_nbr', 'store_nbr'], how='left')

        # S_df['y_item'].fillna(S_df['y_store'], inplace=True)
        # S_df['y_itemstore'].fillna(S_df['y_item'], inplace=True)

        del items, stores
        gc.collect()

        print(f'S wrangle time {time.time()-start :.2f} segs \n')
        check_nans(S_df)



        # Avoid computation if temporal data is already created
        temporal_path = f'{directory}/favorita/temporal.feather'
        if not os.path.exists(temporal_path):
            # if promotion is unknown it is set to False
            temporal['onpromotion'] = temporal['onpromotion'].fillna(False)
            temporal['open'] = 1 # flag where sales data is known
            check_nans(temporal)

            #------------------------- Filter/Balance Temporal Data ----------------------#
            # Extract only a subset of data to save/process for efficiency (according to GR)
            start = time.time()
            start_date = datetime(2015, 1, 1)
            end_date = datetime(2016, 6, 1)
            if start_date is not None: temporal = temporal[(temporal['date'] >= start_date)]
            if end_date is not None: temporal = temporal[(temporal['date'] < end_date)]

            # Add trajectory identifier
            temporal = temporal.merge(S_df[['item_nbr', 'store_nbr', 'traj_id']],
                                      on=['item_nbr', 'store_nbr'], how='left')

            # Remove all IDs with negative returns
            start = time.time()
            min_returns = temporal['unit_sales'].groupby(temporal['traj_id']).min()
            valid_ids = set(min_returns[min_returns >= 0].index)
            new_temporal = temporal[temporal['traj_id'].isin(valid_ids)].copy()

            del temporal, S_df['store_nbr']
            gc.collect()
            temporal = new_temporal

            print(f'Filter and Removing returns data {time.time()-start :.2f} segs')
            check_nans(temporal)

            # Resampling
            print("Resampling")
            start = time.time()
            resampled_dfs = []
            for traj_id, raw_sub_df in temporal.groupby('traj_id'):

                #print('Resampling', traj_id)
                sub_df = raw_sub_df.set_index('date', drop=True).copy()

                #sub_df = sub_df.resample('1d').last()
                #sub_df['date'] = sub_df.index
                #sub_df[['store_nbr', 'item_nbr', 'onpromotion']] \
                #            = sub_df[['store_nbr', 'item_nbr', 'onpromotion']].fillna(method='ffill')
                #sub_df['open'] = sub_df['open'].fillna(0)    # flag where sales data is unknown

                index = pd.date_range(sub_df.index[0], sub_df.index[-1], freq="D")

                ffill_vars = ['store_nbr', 'item_nbr', 'traj_id', 'onpromotion']
                sub_df1 = sub_df[ffill_vars].reindex(index, method='ffill')
                sub_df1['date'] = sub_df1.index

                zfill_vars = ['unit_sales', 'open']
                sub_df2 = sub_df[zfill_vars].reindex(index, fill_value=0)

                sub_df = pd.concat([sub_df1, sub_df2], axis=1)
                resampled_dfs.append(sub_df.reset_index(drop=True))

            new_temporal = pd.concat(resampled_dfs, axis=0)
            del temporal
            gc.collect()
            temporal = new_temporal
            temporal['log_sales'] = np.log1p(temporal['unit_sales'])

            check_nans(temporal)
            print(f'Resampling to regular grid {time.time()-start} segs')


            #--------------------------- Temporal Data Augmentation ---------------------------#
            dates = temporal['date'].unique()

            # Transactions variable
            start = time.time()
            temporal = temporal.merge(transactions,
                                      on=['date', 'store_nbr'],
                                      #left_on=['date', 'store_nbr'],
                                      #right_on=['date', 'store_nbr'],
                                      how='left')
            temporal['transactions'] = temporal['transactions'].fillna(-1)
            print(f'Transactions variable {time.time()-start :.2f} segs')

            # Oil variable
            start = time.time()
            oil = oil[oil['date'].isin(dates)].fillna(method='ffill')
            oil.rename(columns={"dcoilwtico": "oil"}, inplace=True)
            temporal = temporal.merge(oil, on=['date'], how='left')
            temporal['oil'] = temporal['oil'].fillna(-1)
            print(f'Oil variables {time.time()-start :.2f} segs')

            # Calendar variables
            start = time.time()
            calendar = pd.DataFrame({'date': dates})
            calendar['day_of_week'] = calendar['date'].dt.dayofweek
            calendar['day_of_month'] = calendar['date'].dt.day
            calendar['month'] = calendar['date'].dt.month
            temporal = temporal.merge(calendar, on=['date'], how='left')
            print(f'Calendar variables {time.time()-start :.2f} segs')

            # Holiday variables
            start = time.time()
            holidays = holidays[holidays['transferred']==False].copy()
            holidays.rename(columns={'type': 'holiday_type'}, inplace=True)

            national_holidays = holidays[holidays['locale']=='National']
            local_holidays    = holidays[holidays['locale']=='Local']
            regional_holidays = holidays[holidays['locale']=='Regional']

            temporal['national_hol'] = temporal.merge(national_holidays,
                                                      left_on=['date'],
                                                      right_on=['date'],
                                                      how='left')['description'].fillna('')

            temporal = temporal.merge(stores_statecity, on=['store_nbr'], how='left')
            temporal['regional_hol'] = temporal.merge(regional_holidays,
                                                      left_on=['state', 'date'],
                                                      right_on=['locale_name', 'date'],
                                                      how='left')['description'].fillna('')
            temporal['local_hol'] = temporal.merge(local_holidays,
                                                   left_on=['city', 'date'],
                                                   right_on=['locale_name', 'date'],
                                                   how='left')['description'].fillna('')
            del temporal['state'], temporal['city']

            print(f'Holiday variables {time.time()-start :.2f} segs')
            check_nans(temporal)

            print('Saving processed file to {}'.format(temporal_path))
            temporal.to_feather(temporal_path)

        else:
            temporal = pd.read_feather(temporal_path)

        #------------------------------ X, Y Data Wrangling -----------------------------#

        Y_df = temporal[['traj_id', 'date', 'log_sales']]
        X_df = temporal[['traj_id', 'date',
                         'onpromotion', 'open', 'transactions', 'oil',
                         'day_of_week', 'day_of_month', 'month']]
        # 'national_hol', 'regional_hol', 'local_hol'
        del temporal
        gc.collect()

        Y_df.rename(columns={'traj_id': 'unique_id', 'date': 'ds'}, inplace=True)
        X_df.rename(columns={'traj_id': 'unique_id', 'date': 'ds'}, inplace=True)

        return S_df, Y_df, X_df