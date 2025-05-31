import torch
from torch.utils.data import Dataset
import dgl
import pickle


class ChunkDataset(Dataset):
    def __init__(self, chunks, chunk_ids, node_num, edges, labels=None):
        self.data = []
        self.idx2id = {}
        for idx, chunk_id in enumerate(chunk_ids):
            self.idx2id[idx] = chunk_id
            chunk = chunks[chunk_id]
            graph = dgl.graph(edges, num_nodes=node_num)
            graph.ndata["metrics"] = torch.FloatTensor(chunk["metrics"])
            graph.ndata["traces"] = torch.FloatTensor(chunk["traces"])
            graph.ndata["logs"] = torch.FloatTensor(chunk["logs"])

            if labels is not None:
                self.data.append((graph, labels[idx]))
            else:
                self.data.append((graph, chunk["culprit"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_chunk_id(self, idx):
        return self.idx2id[idx]


class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.data = dataset1.data + dataset2.data
        self.idx2id = {**dataset1.idx2id, **dataset2.idx2id}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_chunk_id(self, idx):
        return self.idx2id[idx]