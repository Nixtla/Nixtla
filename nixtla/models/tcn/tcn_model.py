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
    def __init__(self, output_size, num_inputs, num_channels, kernel_size, dropout):
        super(TCNModule, self).__init__()
        self.output_size = output_size
        self.tcn = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels,
                                    kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(output_size, output_size)

    def forward(self, insample_y, insample_x, insample_mask, outsample_y):
        #TODO: insample_mask
        # size = insample_y.size()
        # noise = t.autograd.Variable(insample_y.data.new(size).normal_(0, 0.2))
        insample_y = insample_y #+ noise
        outsample_y = outsample_y
        insample_y = insample_y.unsqueeze(1)
        normalizer = insample_y[:,:,[-1]]

        # Normalize
        insample_y = insample_y/normalizer
        #insample_y = t.log(insample_y)
        insample_y = t.cat([insample_y, insample_x], dim=1)
        forecast = self.tcn(insample_y)
        forecast = forecast[:, :, -self.output_size:]
        #forecast = self.linear(forecast)
        outsample_y = outsample_y/normalizer
        #outsample_y = t.log(outsample_y)

        # print('insample_y', insample_y)
        # print('outsample_y', outsample_y)
        # print('insample_y', insample_y)
        # print('outsample_y', outsample_y)
        # print('forecast', forecast)

        return forecast, outsample_y

    def predict(self, insample_y, insample_x):
        #TODO: insample_mask
        insample_y = insample_y.unsqueeze(1)
        normalizer = insample_y[:,:,[-1]]

        # Normalize
        insample_y = insample_y/normalizer
        #insample_y = t.log(insample_y/normalizer)
        insample_y = t.cat([insample_y, insample_x], dim=1)
        forecast = self.tcn(insample_y)
        forecast = forecast[:, :, -self.output_size:]
        #forecast = t.exp(forecast)
        forecast = forecast * normalizer
        forecast = forecast.squeeze(1)
        #forecast = self.linear(forecast)

        return forecast