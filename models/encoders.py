import torch.nn as nn
from .graph_model import GraphModel1
from .layers import GRUEncoder


class MetricEncoder(nn.Module):
    def __init__(self, device, out_dim=64, **kwargs):
        super().__init__()
        self.metric_model = GRUEncoder(in_size=7, out_dim=64).to(device)
        self.status_model = GraphModel1(in_dim=out_dim, device=device, **kwargs)

    def forward(self, graph):
        metric_embedding = self.metric_model(graph.ndata["metrics"])
        return {'metric_embedding1': self.status_model(graph, metric_embedding)}


class TraceEncoder(nn.Module):
    def __init__(self, device, out_dim=64, **kwargs):
        super().__init__()
        self.trace_model = GRUEncoder(in_size=1, out_dim=64).to(device)
        self.status_model = GraphModel1(in_dim=out_dim, device=device, **kwargs)

    def forward(self, graph):
        trace_embedding = self.trace_model(graph.ndata["traces"])
        return {"trace_embedding1": self.status_model(graph, trace_embedding)}


class LogEncoder(nn.Module):
    def __init__(self, device, out_dim=64, **kwargs):
        super().__init__()
        self.log_embedder = nn.Linear(15, out_dim).to(device)
        self.status_model = GraphModel1(in_dim=out_dim, device=device, **kwargs)

    def forward(self, graph):
        log_emb = self.log_embedder(graph.ndata["logs"])
        return {'log_embedding': self.status_model(graph, log_emb)}