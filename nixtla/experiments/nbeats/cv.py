# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/experiments_nbeats__cv.ipynb (unless otherwise specified).

__all__ = ['CrossValidationNbeats']

# Cell
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Iterable, Union

import pandas as pd
from tqdm import tqdm

from ...models.nbeats import Nbeats

# Cell
class CrossValidationNbeats:

    def __init__(self, directory: Union[Path, str],
                 grid: Dict[str, Iterable],
                 ensemble_grid: Dict[str, Iterable],
                 loader: Callable,
                 gpu_id: int):
        self.directory = Path(str(directory))
        self.grid = grid
        self.ensemble_grid = ensemble_grid
        self.loader = loader
        self.gpu_id = gpu_id

        self.params = _parameter_grid(grid)
        self.ensemble_params = _parameter_grid(ensemble_grid)

        self.directory.mkdir(exist_ok=True, parents=True)

    def fit(self, ts_dataset):
        ensemble_dir = [f'{param}={value}' for param, value in self.ensemble_grid.items()]
        ensemble_dir = self.directory / '_'.join(ensemble_dir)
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        for _, row_params in self.params.iterrows():
            hparams_grid = row_params.to_dict()

            forecast_file = [f'{param}={value}' for param, value in hparams_grid.items()]
            forecast_file = '_'.join(forecast_file) + '.p'
            forecast_file = ensemble_dir / forecast_file

            forecasts = []
            for idx_ensemble, row_ensemble in tqdm(self.ensemble_params.iterrows()):
                hparams_ensemble = row_ensemble.to_dict()
                hparams = {**hparams_grid, **hparams_ensemble}

                ts_loader = self.loader(ts_dataset=ts_dataset,
                                        offset=hparams['offset'],
                                        window_sampling_limit=hparams['window_sampling_limit_multiplier'] * hparams['output_size'],
                                        input_size=hparams['input_size_multiplier'] * hparams['output_size'],
                                        output_size=hparams['output_size'],
                                        idx_to_sample_freq=1,
                                        batch_size=hparams['batch_size'],
                                        model='nbeats',
                                        train_loader=True)

                model = Nbeats(input_size_multiplier=hparams['input_size_multiplier'],
                               output_size=hparams['output_size'],
                               shared_weights=hparams['shared_weights'],
                               stack_types=hparams['stack_types'],
                               n_blocks=hparams['n_blocks'],
                               n_layers=hparams['n_layers'],
                               n_hidden=hparams['n_hidden'],
                               n_harmonics=hparams['n_harmonics'],
                               n_polynomials=hparams['n_polynomials'],
                               learning_rate=hparams['learning_rate'],
                               lr_decay=hparams['lr_decay'],
                               n_lr_decay_steps=hparams['n_lr_decay_steps'],
                               n_iterations=hparams['n_iterations'],
                               loss=hparams['loss'],
                               frequency=hparams['frequency'],
                               seasonality=hparams['seasonality'],
                               random_seed=hparams['random_seed'])
                model.fit(ts_loader, eval_steps=1000, verbose=False)

                y_hat = model.predict(ts_loader)
                y_hat.rename({'y_hat': f'y_hat_{idx_ensemble}'}, axis=1, inplace=True)

                forecasts.append(y_hat.set_index(['unique_id', 'ds']))

            forecasts = pd.concat(forecasts, 1)
            forecasts['y_hat'] = forecasts.median(1)
            forecasts = forecasts.reset_index()

            return forecasts


# Internal Cell
def _parameter_grid(grid):
    specs_list = list(product(*list(grid.values())))
    model_specs_df = pd.DataFrame(specs_list, columns=list(grid.keys()))

    return model_specs_df