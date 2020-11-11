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

