import os

import pandas as pd
from nixtla.data.datasets import Tourism

if __name__ == '__main__':
    os.environ['AWS_PROFILE'] = 'pos'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-2'
    
    tourism = Tourism.load('data')
    s3_prefix = 's3://nixtla-datasets/tourism'
    for group in ('Yearly', 'Quarterly', 'Monthly'):
        data = tourism.get_group(group).Y
        data = data.set_index(['unique_id', 'ds'])
        data.to_parquet(f'{s3_prefix}/{group}/data.parquet')
