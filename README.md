# Nixtla
> Machine learning for time series forecasting.


salu2

## Install

`pip install nixtla`

## How to use

Import a dataset.

```
from nixtla.data.datasets import Tourism
from nixtla.data.dataloaders import NBeatsDataLoader, uids_ts_from_df

tourism_dataset = Tourism.load('data')
tourism_yearly = tourism_dataset.get_group('Yearly')
tourism_yearly.Y.head()
```

    100%|██████████| 356k/356k [00:00<00:00, 747kiB/s] 
    INFO:nixtla.data.datasets:Successfully downloaded 27-3-Athanasopoulos1.zip, 356334, bytes.
    INFO:nixtla.data.datasets:Successfully decompressed data/tourism/datasets/27-3-Athanasopoulos1.zip





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unique_id</th>
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Y1</td>
      <td>1</td>
      <td>25092.2284</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Y1</td>
      <td>2</td>
      <td>24271.5134</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Y1</td>
      <td>3</td>
      <td>25828.9883</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Y1</td>
      <td>4</td>
      <td>27697.5047</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Y1</td>
      <td>5</td>
      <td>27956.2276</td>
    </tr>
  </tbody>
</table>
</div>



Split dataframe into sequence of series

```
unique_ids, time_series = uids_ts_from_df(tourism_yearly.Y, id_cols='unique_id')
unique_ids[:2], time_series[:2]
```




    (('Y1', 'Y2'),
     (array([25092.2284, 24271.5134, 25828.9883, 27697.5047, 27956.2276,
             29924.4321, 30216.8321, 32613.4968, 36053.1674, 38472.7532,
             38420.894 ]),
      array([ 887896.51,  887068.98,  971549.04, 1064206.39, 1195560.94,
             1351933.55, 1372823.36, 1532533.61, 1587760.62, 1617737.85,
             1499631.11])))



Create a train dataloader

```
train_dl = NBeatsDataLoader(time_series, input_size=4, output_size=2, batch_size=4, shuffle=False)
next(iter(train_dl))
```




    (tensor([[    0.,     0.,     0.,     0.],
             [    0.,     0.,     0.,     0.],
             [    0.,     0.,     0.,     0.],
             [    0.,     0.,     0., 18441.]]),
     tensor([[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 1.]]),
     tensor([[    0.,     0.],
             [    0.,     0.],
             [    0.,     0.],
             [21934., 23739.]]),
     tensor([[0., 0.],
             [0., 0.],
             [0., 0.],
             [1., 1.]]))


