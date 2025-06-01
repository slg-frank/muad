import os
import json
import logging
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import DataLoader

from data.dataset import ChunkDataset, CombinedDataset
from data.utils import load_data, create_dataloader

from training.base_model import BaseModel
from active_learning.strategies import hybrid_selection
from utils.general_utils import seed_everything, dump_params, dump_scores, split_data
from utils.logging_utils import setup_logging


def main(config_path):
    # Load configuration
    with open(config_path) as f:
        params = json.load(f)

    # Set random seed
    seed_everything(params["random_seed"])

    # Initialize logging
    logger = setup_logging(params["result_dir"])
    logger.info("Starting MUAD training")

    # Load data
    train_data, node_num, edges = load_data(
        "Dataset/chunk/chunk_train.pkl",
        "Dataset/chunk/metadata.json"
    )
    test_data, _, _ = load_data(
        "Dataset/chunk/chunk_test.pkl",
        "Dataset/chunk/metadata.json"
    )

    # Split initial train and validation sets
    train_keys, val_keys = split_data(train_data, train_ratio=0.1)

    # Create datasets
    train_dataset = ChunkDataset(train_data, train_keys, node_num, edges)
    val_dataset = ChunkDataset(train_data, val_keys, node_num, edges)
    test_dataset = ChunkDataset(test_data, list(test_data.keys()), node_num, edges)

    # Create data loaders
    train_loader = create_dataloader(train_dataset)
    val_loader = create_dataloader(val_dataset, shuffle=False)
    test_loader = create_dataloader(test_dataset, shuffle=False)

    # Initialize device
    device = torch.device("cuda" if params["gpu"] and torch.cuda.is_available() else "cpu")

    # Main training loop
    for iteration in range(params["max_iter"]):
        logger.info(f"Starting active learning iteration {iteration + 1}")

        # Initialize model
        model = BaseModel(
            device,
            lr=params["lr"],
            epochs=params["epochs"],
            patience=params["patience"],
            result_dir=params["result_dir"],
            hash_id=f"iter_{iteration}",
            hidden_dim = [64, 64]


        )

        # Train model
        best_f1, best_epoch = model.fit(train_loader, test_loader, params["evaluation_epoch"])

        # Evaluate final model
        test_results = model.evaluate(test_loader)
        dump_scores(test_results, params["result_dir"], iteration)

        # Active learning sample selection
        uncertain_indices, high_conf_indices, pseudo_labels = hybrid_selection(
            model.model, val_loader, device, n_select=200
        )

        # Update datasets
        # Add uncertain samples to training set
        new_train_keys = [val_keys[i] for i in uncertain_indices]
        train_keys.extend(new_train_keys)

        # Add high-confidence samples to training set (with pseudo labels)
        if high_conf_indices:
            high_conf_keys = [val_keys[i] for i in high_conf_indices]
            pseudo_dataset = ChunkDataset(train_data, high_conf_keys, node_num, edges, pseudo_labels)
            train_dataset = CombinedDataset(train_dataset, pseudo_dataset)
            train_keys.extend(high_conf_keys)

        # Remove selected samples from validation set
        for idx in sorted(uncertain_indices + high_conf_indices, reverse=True):
            del val_keys[idx]

        logger.info(f"Iteration {iteration + 1} complete. Train size: {len(train_keys)}, Val size: {len(val_keys)}")

        # Check stopping condition
        if test_results["Pre"] >= params["precicison"]:
            logger.info(f"Precision {test_results['Pre']:.4f} reached threshold. Stopping early.")
            break

    logger.info("Training completed")


if __name__ == "__main__":
    main("configs/params.json")