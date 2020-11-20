# Nixtla



[nikstla] (noun, nahuatl) Period of time
> Machine learning for time series forecasting.

## Install

`pip install nixtla`

## How to use

Import a dataset and NBEATS model

```python
from IPython.display import Markdown
from nixtla.data.datasets import EPF
from nixtla.data.ts_loader import TimeSeriesLoader
from nixtla.models import Nbeats

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
    Step: 0, Time: 0.028, Insample MAPE: 0.39416
    Step: 2, Time: 0.098, Insample MAPE: 0.37484
    Step: 4, Time: 0.168, Insample MAPE: 0.37288
    Step: 6, Time: 0.238, Insample MAPE: 0.34576
    Step: 8, Time: 0.308, Insample MAPE: 0.29966


```python
y_hat = model.predict(ts_loader)
```

```python
Markdown(y_hat.head().to_markdown(index=False))
```




| unique_id   | ds                  |   y_hat |
|:------------|:--------------------|--------:|
| PJM         | 2016-12-27 23:00:00 | 21.1163 |
| PJM         | 2016-12-28 23:00:00 | 18.7953 |
| PJM         | 2016-12-29 23:00:00 | 20.5183 |
| PJM         | 2016-12-30 23:00:00 | 19.2623 |
| PJM         | 2016-12-31 23:00:00 | 16.1545 |



# ESRNN

```python
from nixtla.models import ESRNN

pjm = EPF.load_groups(directory='data', groups=['NP', 'PJM'], return_tensor=False)
pjm_test = EPF.load_groups(directory='data', groups=['NP', 'PJM'], training=False, return_tensor=False)
```

```python
esrnn_model = ESRNN(max_epochs=2, input_size=48, 
                    batch_size=2,
                    output_size=728, seasonality=[24])
```

```python
X = pjm.Y[['unique_id', 'ds']]
X['x'] = 1
```

```python
esrnn_model.fit(X, pjm.Y)
```

    Infered frequency: H
    =============== Training ESRNN  ===============
    
    ========= Epoch 0 finished =========
    Training time: 5.12275
    Training loss (50 prc): 0.27288
    ========= Epoch 1 finished =========
    Training time: 5.25982
    Training loss (50 prc): 0.27159
    Train finished! 
    


```python
Markdown(esrnn_model.predict(pjm_test.Y).head().to_markdown(index=False))
```




| unique_id   | ds                  |     y |   y_hat |
|:------------|:--------------------|------:|--------:|
| NP          | 2016-12-27 00:00:00 | 24.08 | 28.6131 |
| NP          | 2016-12-27 01:00:00 | 22.52 | 23.1586 |
| NP          | 2016-12-27 02:00:00 | 20.13 | 19.901  |
| NP          | 2016-12-27 03:00:00 | 19.86 | 24.0669 |
| NP          | 2016-12-27 04:00:00 | 20.09 | 25.3265 |


