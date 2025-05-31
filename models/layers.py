import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GRUEncoder(nn.Module):
    def __init__(self, in_size, out_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=in_size, hidden_size=out_dim,
                          batch_first=True, bidirectional=False)
        self.out_dim = out_dim

    def forward(self, x):
        _, hidden = self.gru(x)
        return hidden.squeeze(0)


class FullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim, linear_sizes):
        super().__init__()
        layers = []
        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i - 1]
            layers += [nn.Linear(input_size, hidden), nn.ReLU()]
        layers += [nn.Linear(linear_sizes[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self._xavier_init()

    def forward(self, x):
        return self.clf(x)

    def _xavier_init(self):
        for m in self.clf:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)