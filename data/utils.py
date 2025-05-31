import os
import json
import pickle

import dgl
import torch
import numpy as np
from torch.utils.data import DataLoader


def read_json(filepath):
    if os.path.exists(filepath):
        assert filepath.endswith('.json')
        with open(filepath, 'r') as f:
            return json.loads(f.read())
    else:
        raise FileNotFoundError(f"File path {filepath} not exists!")


def collate(data):
    graphs, labels = map(list, zip(*data))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def load_data(data_path, metadata_path):
    with open(data_path, "rb") as fr:
        data = pickle.load(fr)

    metadata = read_json(metadata_path)
    node_num, edges_tup = metadata["node_num"], tuple(metadata["edges"])
    return data, node_num, edges_tup


def create_dataloader(dataset, batch_size=50, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        pin_memory=True
    )