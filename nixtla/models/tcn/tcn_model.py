# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/models_tcn__tcn_model.ipynb (unless otherwise specified).

__all__ = ['TCNModule']

# Cell
import torch as t
import torch.nn as nn
from ..component import TemporalConvNet
import numpy as np

# Cell
# TODO: rename
class TCNModule(nn.Module):
    def __init__(self, output_size, num_inputs, num_channels, num_static, kernel_size, dropout):
        super(TCNModule, self).__init__()
        self.output_size = output_size
        self.tcn = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels,
                                    kernel_size=kernel_size, dropout=dropout)
        n_x_t = num_inputs - 1
        self.linear = nn.Linear(num_channels[-1] + num_static + n_x_t*output_size, output_size)

    def forward(self, insample_y, insample_x, outsample_x, s_matrix):
        #TODO: linear convolucion de kernel 1, sin FC, version metrica solo en la ultima

        insample_y = insample_y.unsqueeze(1)
        insample_y = t.cat([insample_y, insample_x], dim=1)
        forecast = self.tcn(insample_y)
        forecast = forecast[:, :, -1:]

        forecast = forecast.reshape(len(forecast),-1)

        # Context
        # (bs, n_x_t, t) -> (bs,n_x_t*t)
        outsample_x = outsample_x.reshape(len(outsample_x), -1)
        forecast_context = t.cat([forecast, outsample_x], dim=1)
        forecast_context = t.cat([forecast_context, s_matrix], dim=1)

        forecast = self.linear(forecast_context)

        return forecast