# Nixtla



[nikstla] (noun, nahuatl) Period of time
> Machine learning for time series forecasting.

## Install

`pip install nixtla`

## How to use

Import a dataset and NBEATS model

```python
from nixtla.data.datasets import EPF
from nixtla.data.ts_loader import TimeSeriesLoader
from nixtla.models.nbeats import Nbeats

pjm = EPF.load(directory='data', group='PJM')
```

    Processing dataframes ...
    Creating ts tensor ...


Loader parameters

```python
window_sampling_limit = 365
input_size_multiplier = 3
output_size = 24 * 4
offset = 30 * output_size
```

```python
ts_loader = TimeSeriesLoader(ts_dataset=pjm,
                             offset=offset,
                             window_sampling_limit=window_sampling_limit,
                             input_size=input_size_multiplier * output_size,
                             output_size=output_size,
                             idx_to_sample_freq=1,
                             batch_size=512,
                             model='nbeats')
```

    Creating windows matrix ...


```python
model = Nbeats(input_size_multiplier=input_size_multiplier,
               output_size=output_size,
               shared_weights=False,
               stack_types=['identity'],
               n_blocks=[1],
               n_layers=[4],
               n_hidden=[256],
               exogenous_in_mlp=False,
               learning_rate=0.001,
               lr_decay=1.0,
               n_lr_decay_steps=3,
               n_iterations=10,
               early_stopping=None,
               loss='MAPE',
               random_seed=1)
```

```python
model.fit(ts_loader, eval_steps=2)
```

    ============================== Start fitting ==============================
    Number of exogenous variables: 9
    Number of static variables: 0 , with dim_hidden: 1
    Number of iterations: 10
    Number of blocks: 1
    Step: 0, Time: 0.726, Insample MAPE: 0.39416
    Step: 2, Time: 0.786, Insample MAPE: 0.37484
    Step: 4, Time: 0.862, Insample MAPE: 0.37288
    Step: 6, Time: 0.921, Insample MAPE: 0.34576
    Step: 8, Time: 0.993, Insample MAPE: 0.29966


```python
y_hat = model.predict(ts_loader)
```

```python
y_hat
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
      <th>y_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PJM</td>
      <td>2016-12-27 23:00:00</td>
      <td>21.116310</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PJM</td>
      <td>2016-12-28 23:00:00</td>
      <td>18.795292</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PJM</td>
      <td>2016-12-29 23:00:00</td>
      <td>20.518282</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PJM</td>
      <td>2016-12-30 23:00:00</td>
      <td>19.262295</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PJM</td>
      <td>2016-12-31 23:00:00</td>
      <td>16.154509</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>91</th>
      <td>PJM</td>
      <td>2017-03-28 23:00:00</td>
      <td>21.821650</td>
    </tr>
    <tr>
      <th>92</th>
      <td>PJM</td>
      <td>2017-03-29 23:00:00</td>
      <td>23.728720</td>
    </tr>
    <tr>
      <th>93</th>
      <td>PJM</td>
      <td>2017-03-30 23:00:00</td>
      <td>23.458321</td>
    </tr>
    <tr>
      <th>94</th>
      <td>PJM</td>
      <td>2017-03-31 23:00:00</td>
      <td>26.678900</td>
    </tr>
    <tr>
      <th>95</th>
      <td>PJM</td>
      <td>2017-04-01 23:00:00</td>
      <td>24.330669</td>
    </tr>
  </tbody>
</table>
<p>96 rows × 3 columns</p>
</div>


