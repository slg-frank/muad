import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATv2Conv, GlobalAttentionPooling
import math


class GraphModel1(nn.Module):
    def __init__(self, in_dim, graph_hiddens=[64], device='cpu',
                 attn_head=4, activation=0.2, **kwargs):
        super().__init__()
        layers = []
        for i, hidden in enumerate(graph_hiddens):
            in_feats = graph_hiddens[i - 1] if i > 0 else in_dim
            dropout = kwargs.get("attn_drop", 0)
            layers.append(GATv2Conv(
                in_feats, out_feats=hidden, num_heads=attn_head,
                attn_drop=dropout, negative_slope=activation,
                allow_zero_in_degree=True
            ))
        self.net = nn.Sequential(*layers).to(device)
        self.out_dim = graph_hiddens[-1]
        self.pooling = GlobalAttentionPooling(nn.Linear(self.out_dim, 1))
        self.maxpool = nn.MaxPool1d(attn_head)

    def forward(self, graph, x):
        out = x
        for layer in self.net:
            out = layer(graph, out)
            out = self.maxpool(out.permute(0, 2, 1)).permute(0, 2, 1).squeeze()
        return self.pooling(graph, out)


class SimpleAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))
        self._glorot_init()

    def forward(self, x):
        attention_scores = torch.matmul(x, self.attention_weights) + self.bias
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return x * attention_weights

    def _glorot_init(self):
        stdv = math.sqrt(6.0 / (self.attention_weights.size(-2) + self.attention_weights.size(-1)))
        self.attention_weights.data.uniform_(-stdv, stdv)