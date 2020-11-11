# Nixtla



[nikstla] (noun, nahuatl) Period of time
> Machine learning for time series forecasting.

## Install

`pip install nixtla`

## How to use

Import a dataset.

```python
from nixtla.data.datasets import Tourism
from nixtla.data.dataloaders import NBeatsDataLoader, uids_ts_from_df

tourism_yearly = Tourism.load(directory='data', group='Yearly', return_tensor=False)
tourism_yearly.Y.head()
```




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
      <td>1979-12-31</td>
      <td>25092.2284</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Y1</td>
      <td>1980-12-31</td>
      <td>24271.5134</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Y1</td>
      <td>1981-12-31</td>
      <td>25828.9883</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Y1</td>
      <td>1982-12-31</td>
      <td>27697.5047</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Y1</td>
      <td>1983-12-31</td>
      <td>27956.2276</td>
    </tr>
  </tbody>
</table>
</div>



Split dataframe into sequence of series

```python
unique_ids, time_series = uids_ts_from_df(tourism_yearly.Y, id_cols='unique_id')
unique_ids[:2], time_series[:2]
```




    (('Y1', 'Y10'),
     (array([25092.2284, 24271.5134, 25828.9883, 27697.5047, 27956.2276,
             29924.4321, 30216.8321, 32613.4968, 36053.1674, 38472.7532,
             38420.894 ]),
      array([ 17913.,  27161.,  31997.,  38761.,  52844.,  86381.,  89549.,
              98628., 104690., 104505.,  92500.,  97887., 112147., 118189.,
             143865., 158712., 162135., 170896., 183523., 229567., 236363.,
             249698., 295018., 376746., 326418., 214556., 259858., 278460.,
             293105., 329604., 234260.])))



Create a train dataloader

```python
train_dl = NBeatsDataLoader(time_series, input_size=4, output_size=2, batch_size=4, shuffle=False)
next(iter(train_dl))
```




    (tensor([[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]]),
     tensor([[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]]),
     tensor([[0., 0.],
             [0., 0.],
             [0., 0.],
             [0., 0.]]),
     tensor([[0., 0.],
             [0., 0.],
             [0., 0.],
             [0., 0.]]))


