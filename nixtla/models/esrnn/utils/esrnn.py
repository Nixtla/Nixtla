# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/models_esrnn_utils__esrnn.ipynb (unless otherwise specified).

__all__ = []

# Cell
import torch
import torch.nn as nn
from .drnn import DRNN
import numpy as np

# Cell
class _ES(nn.Module):
    def __init__(self, n_series, input_size, output_size, n_s, seasonality, noise_std, device):
        super(_ES, self).__init__()

        self.n_series = n_series
        self.input_size = input_size
        self.output_size = output_size
        self.n_s = n_s
        self.seasonality = seasonality
        assert len(self.seasonality) in [0, 1, 2]

        self.noise_std = noise_std

        self.device = device


    def gaussian_noise(self, input_data, std=0.2):
        size = input_data.size()
        noise = torch.autograd.Variable(input_data.data.new(size).normal_(0, std))
        return input_data + noise

    #@jit.script_method
    def compute_levels_seasons(self, y, idxs):
        pass

    def normalize(self, y, level, seasonalities):
        pass

    def predict(self, trend, levels, seasonalities):
        pass

    def forward(self, y, s_matrix, idxs):
        # parse attributes
        input_size = self.input_size
        output_size = self.output_size
        n_s = self.n_s
        noise_std = self.noise_std
        seasonality = self.seasonality
        batch_size = len(idxs)

        n_series, n_time = y.shape

        # Explicacion: windows_end es el idx delultimo inicio de windows. Como se necesitan windows completas con input
        # + output, se pierden (input_size+output_size-1) windows del len total de la serie.
        windows_end = n_time - input_size - output_size + 1
        windows_range = range(windows_end)
        n_windows = len(windows_range)

        assert n_windows>0

        # Initialize windows, levels and seasonalities
        levels, seasonalities = self.compute_levels_seasons(y=y, idxs=idxs)
        windows_y_insample = torch.zeros((n_windows, batch_size, input_size + n_s),
                                          device=self.device)
        windows_y_outsample = torch.zeros((n_windows, batch_size, output_size),
                                           device=self.device)

        for i, window in enumerate(windows_range):
            # Windows yhat
            y_insample_start = window
            y_insample_end = input_size + window

            # Y_hat deseasonalization and normalization
            window_y_insample = self.normalize(y=y[:, y_insample_start:y_insample_end],
                                               level=levels[:, [y_insample_end-1]],
                                               seasonalities=seasonalities,
                                               start=y_insample_start, end=y_insample_end) #TODO: improve this inputs

            if self.training:
                window_y_insample = self.gaussian_noise(window_y_insample, std=noise_std)

            # Concatenate S_matrix
            if n_s > 0:
                window_y_insample = torch.cat((window_y_insample, s_matrix), 1)

            windows_y_insample[i, :, :] += window_y_insample

            # Windows_y_outsample
            y_outsample_start = y_insample_end
            y_outsample_end = y_outsample_start + output_size
            window_y_outsample = y[:, y_outsample_start:y_outsample_end]
            # If training, normalize outsample for loss on normalized data
            if self.training:
                # Y deseasonalization and normalization
                window_y_outsample = self.normalize(y=window_y_outsample,
                                                    level=levels[:, [y_outsample_start]],
                                                    seasonalities=seasonalities,
                                                    start=y_outsample_start, end=y_outsample_end) #TODO: improve this inputs
            windows_y_outsample[i, :, :] += window_y_outsample

        return windows_y_insample, windows_y_outsample, levels, seasonalities

class _ESM(_ES):
    def __init__(self, n_series, input_size, output_size, n_s, seasonality, noise_std, device):
        super(_ESM, self).__init__(n_series, input_size, output_size, n_s, seasonality, noise_std, device)
        # Level and Seasonality Smoothing parameters
        # 1 level, S seasonalities, S init_seas
        embeds_size = 1 + len(self.seasonality) + sum(self.seasonality)
        init_embeds = torch.ones((self.n_series, embeds_size)) * 0.5
        self.embeds = nn.Embedding(self.n_series, embeds_size)
        self.embeds.weight.data.copy_(init_embeds)
        self.seasonality = torch.LongTensor(self.seasonality)

    #@jit.script_method
    def compute_levels_seasons(self, y, idxs):
        """
        Computes levels and seasons
        """
        # Lookup parameters per serie
        #seasonality = self.seasonality
        embeds = self.embeds(idxs)
        lev_sms = torch.sigmoid(embeds[:, 0])

        # Initialize seasonalities
        seas_prod = torch.ones(len(y[:,0])).to(y.device)
        seasonalities1 = []
        seasonalities2 = []
        seas_sms1 = torch.ones(1).to(y.device)
        seas_sms2 = torch.ones(1).to(y.device)

        if len(self.seasonality)>0:
            seas_sms1 = torch.sigmoid(embeds[:, 1])
            init_seas1 = torch.exp(embeds[:, 2:(2+self.seasonality[0])]).unbind(1)
            assert len(init_seas1) == self.seasonality[0]

            for i in range(len(init_seas1)):
                seasonalities1 += [init_seas1[i]]
            seasonalities1 += [init_seas1[0]]
            seas_prod = seas_prod * init_seas1[0]

        if len(self.seasonality)==2:
            seas_sms2 = torch.sigmoid(embeds[:, 2+self.seasonality[0]])
            init_seas2 = torch.exp(embeds[:, 3+self.seasonality[0]:]).unbind(1)
            assert len(init_seas2) == self.seasonality[1]

            for i in range(len(init_seas2)):
                seasonalities2 += [init_seas2[i]]
            seasonalities2 += [init_seas2[0]]
            seas_prod = seas_prod * init_seas2[0]

        # Initialize levels
        levels = []
        levels += [y[:,0]/seas_prod]

        # Recursive seasonalities and levels
        ys = y.unbind(1)
        n_time = len(ys)
        for t in range(1, n_time):
            seas_prod_t = torch.ones(len(y[:,t])).to(y.device)
            if len(self.seasonality)>0:
                seas_prod_t = seas_prod_t * seasonalities1[t]
            if len(self.seasonality)==2:
                seas_prod_t = seas_prod_t * seasonalities2[t]

            newlev = lev_sms * (ys[t] / seas_prod_t) + (1-lev_sms) * levels[t-1]
            levels += [newlev]

            if len(self.seasonality)==1:
                newseason1 = seas_sms1 * (ys[t] / newlev) + (1-seas_sms1) * seasonalities1[t]
                seasonalities1 += [newseason1]

            if len(self.seasonality)==2:
                newseason1 = seas_sms1 * (ys[t] / (newlev * seasonalities2[t])) + \
                                         (1-seas_sms1) * seasonalities1[t]
                seasonalities1 += [newseason1]
                newseason2 = seas_sms2 * (ys[t] / (newlev * seasonalities1[t])) + \
                                         (1-seas_sms2) * seasonalities2[t]
                seasonalities2 += [newseason2]

        levels = torch.stack(levels).transpose(1,0)

        seasonalities = []

        if len(self.seasonality)>0:
            seasonalities += [torch.stack(seasonalities1).transpose(1,0)]

        if len(self.seasonality)==2:
            seasonalities += [torch.stack(seasonalities2).transpose(1,0)]

        return levels, seasonalities

    def normalize(self, y, level, seasonalities, start, end):
        # Deseasonalization and normalization
        y_n = y / level
        for s in range(len(self.seasonality)):
            y_n /= seasonalities[s][:, start:end]
        y_n = torch.log(y_n)
        return y_n

    def predict(self, trends, levels, seasonalities):

        # First trend uses last value of first y_insample of length self.input_size.
        # Last self.output_size levels are not used (leakeage!!!)
        levels = levels[:, (self.input_size-1):-self.output_size]
        levels = levels.unsqueeze(2)

        # Seasonalities are unfolded, because each element of trends must be multiplied
        # by the corresponding seasonality.
        for i in range(len(seasonalities)):
            seasonalities[i] = seasonalities[i][:, self.input_size : -self.seasonality[i]]
            seasonalities[i] = seasonalities[i].unfold(dimension=-1, size=self.output_size, step=1)

        # Denormalize
        trends = torch.exp(trends)
        # Deseasonalization and normalization (inverse)
        y_hat = trends * levels
        for s in range(len(self.seasonality)):
            y_hat *= seasonalities[s]

        return y_hat

class _RNN(nn.Module):
    def __init__(self, input_size, output_size, n_s, cell_type, dilations, state_hsize, add_nl_layer):
        super(_RNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.n_s = n_s

        self.cell_type = cell_type
        self.dilations = dilations
        self.state_hsize = state_hsize
        self.add_nl_layer = add_nl_layer
        self.layers = len(dilations)

        layers = []
        for grp_num in range(len(self.dilations)):
            if grp_num == 0:
                input_size = self.input_size + self.n_s
            else:
                input_size = self.state_hsize
            layer = DRNN(input_size,
                         self.state_hsize,
                         n_layers=len(self.dilations[grp_num]),
                         dilations=self.dilations[grp_num],
                         cell_type=self.cell_type)
            layers.append(layer)

        self.rnn_stack = nn.Sequential(*layers)

        if self.add_nl_layer:
            self.MLPW  = nn.Linear(self.state_hsize, self.state_hsize)

        self.adapterW  = nn.Linear(self.state_hsize, self.output_size)

    def forward(self, input_data):
        for layer_num in range(len(self.rnn_stack)):
            residual = input_data
            output, _ = self.rnn_stack[layer_num](input_data)
            if layer_num > 0:
                output += residual
            input_data = output

        if self.add_nl_layer:
            input_data = self.MLPW(input_data)
            input_data = torch.tanh(input_data)

        input_data = self.adapterW(input_data)
        return input_data


class _ESRNN(nn.Module):
    def __init__(self, n_series, input_size, output_size, n_s, seasonality,
                 noise_std, cell_type, dilations, state_hsize, add_nl_layer, device):
        super(_ESRNN, self).__init__()

        self.es = _ESM(n_series=n_series, input_size=input_size, output_size=output_size,
                       n_s=n_s, seasonality=seasonality, noise_std=noise_std,
                       device=device).to(device)
        self.rnn = _RNN(input_size=input_size, output_size=output_size, n_s=n_s,
                        cell_type=cell_type, dilations=dilations, state_hsize=state_hsize,
                        add_nl_layer=add_nl_layer).to(device)

    def forward(self, insample_y, s_matrix, idxs):
        # ES Forward
        windows_y_insample, windows_y_outsample, levels, seasonalities = self.es(y=insample_y,
                                                                         s_matrix=s_matrix, idxs=idxs)
        # RNN Forward
        windows_y_hat = self.rnn(windows_y_insample)

        return windows_y_outsample, windows_y_hat, levels

    def predict(self, insample_y, s_matrix, idxs):
        # ES Forward
        windows_y_insample, windows_y_outsample, levels, seasonalities = self.es(y=insample_y,
                                                                 s_matrix=s_matrix, idxs=idxs)

        # RNN Forward
        trends = self.rnn(windows_y_insample)

        # (n_windows, batch_size, input_size) -> (batch_size, n_windows, input_size)
        trends = trends.permute(1,0,2)
        windows_y_outsample = windows_y_outsample.permute(1,0,2)

        y_hat = self.es.predict(trends, levels, seasonalities)

        return windows_y_outsample, y_hat