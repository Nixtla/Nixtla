# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/data__utils.ipynb (unless otherwise specified).

__all__ = ['create_synthetic_tsdata']

# Cell
from typing import Tuple

import numpy as np
import pandas as pd

# Cell
def create_synthetic_tsdata(n_ts: int = 64,
                            sort: bool = False) -> Tuple[pd.DataFrame,
                                                         pd.DataFrame,
                                                         pd.DataFrame]:
    """Creates synthetic time serie data."""
    uids = np.array([f'uid_{i + 1}' for i in range(n_ts)])
    dss = pd.date_range(end='2020-12-31', periods=n_ts)

    df = []
    for idx in range(n_ts):
        ts = pd.DataFrame({'unique_id': np.repeat(uids[idx], idx + 1),
                           'ds': dss[-(idx + 1):],
                           'y': 1 + np.arange(idx + 1)})
        df.append(ts)

    df = pd.concat(df)
    df['day_of_week'] = df['ds'].dt.day_of_week
    df['future_1'] = df['y'] + 1
    df['id_ts'] = df['unique_id'].astype('category').cat.codes
    if sort:
        df = df.sort_values(['unique_id', 'ds'])

    Y_df = df.filter(items=['unique_id', 'ds', 'y'])
    X_df = df.filter(items=['unique_id', 'ds', 'day_of_week', 'future_1'])
    S_df = df.filter(items=['unique_id', 'id_ts']).drop_duplicates()

    return Y_df, X_df, S_df