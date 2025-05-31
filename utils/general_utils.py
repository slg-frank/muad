import os
import json
import random
import numpy as np
import torch
import hashlib

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def dump_params(params, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    params_path = os.path.join(result_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)

def dump_scores(scores, result_dir, epoch=None):
    with open(os.path.join(result_dir, "scores.txt"), "a") as f:
        if epoch:
            f.write(f"Epoch {epoch}: ")
        f.write(", ".join([f"{k}:{v:.4f}" for k, v in scores.items()]) + "\n")

def get_device(use_gpu):
    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

def split_data(data, train_ratio=0.1):
    keys = list(data.keys())
    n_train = int(len(keys) * train_ratio)
    train_keys = random.sample(keys, n_train)
    val_keys = [k for k in keys if k not in train_keys]
    return train_keys, val_keys