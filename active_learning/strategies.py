import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
from models import MainModel

def entropy_based_selection(model, dataloader, device, n_samples=5):
    model.eval()
    uncertainties = []
    indices = []

    with torch.no_grad():
        for batch_idx, (graph, labels) in enumerate(dataloader):
            graph = graph.to(device)
            entropy_batch = []

            for _ in range(n_samples):
                res = model(graph, labels)
                probs = F.softmax(res["MMlogit"], dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                entropy_batch.append(entropy)

            avg_entropy = torch.stack(entropy_batch).mean(dim=0)
            uncertainties.extend(avg_entropy.cpu().numpy())
            indices.extend(range(batch_idx * len(labels), (batch_idx + 1) * len(labels)))

    # 选择最不确定的样本
    sorted_indices = np.argsort(uncertainties)[::-1]
    return [indices[i] for i in sorted_indices]


def confidence_based_selection(model, dataloader, device, confidence_threshold=0.9):
    model.eval()
    high_confidence_indices = []
    predicted_labels = []

    with torch.no_grad():
        for batch_idx, (graph, labels) in enumerate(dataloader):
            graph = graph.to(device)
            res = model(graph, labels)
            confidences = res["TCPConfidence_sig"].squeeze()

            high_conf_mask = confidences > confidence_threshold
            batch_indices = torch.where(high_conf_mask)[0].cpu().numpy()
            global_indices = [batch_idx * len(labels) + i for i in batch_indices]

            high_confidence_indices.extend(global_indices)
            predicted_labels.extend(res["MMlogit"].argmax(dim=1)[batch_indices].cpu().numpy())

    return high_confidence_indices, predicted_labels


def hybrid_selection(model, dataloader, device, n_select=200, confidence_threshold=0.9):

    uncertain_indices = entropy_based_selection(model, dataloader, device)
    selected_uncertain = uncertain_indices[:n_select]


    high_conf_indices, pseudo_labels = confidence_based_selection(
        model, dataloader, device, confidence_threshold)


    final_uncertain = [idx for idx in selected_uncertain if idx not in high_conf_indices]

    return final_uncertain, high_conf_indices, pseudo_labels