import os

import torch
import logging
import copy
from torch import nn, optim
from tqdm import tqdm
from models.main_model import MainModel


class BaseModel(nn.Module):
    def __init__(self, device, lr=1e-3, epochs=125, patience=5,
                 result_dir='./', hash_id=None, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.device = device
        self.model_save_dir = os.path.join(result_dir, hash_id)
        self.model = MainModel(device,**kwargs).to(device)

    def evaluate(self, test_loader, datatype="Test"):
        self.model.eval()
        TP, FP, FN, TN = 0, 0, 0, 0

        with torch.no_grad():
            for graph, labels in test_loader:
                graph = graph.to(self.device)
                res = self.model(graph, labels)
                predictions = res["y_pred"]

                for i, pred in enumerate(predictions):
                    label = labels[i].item()
                    if label == 0:
                        TN += 1 if pred == 0 else 0
                        FP += 1 if pred != 0 else 0
                    else:
                        TP += 1 if pred != 0 else 0
                        FN += 1 if pred == 0 else 0

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        logging.info(
            f"{datatype} -- F1: {f1:.4f}, Rec: {recall:.4f}, Pre: {precision:.4f}")

        return {
            "F1": f1,
            "Rec": recall,
            "Pre": precision
        }

    def fit(self, train_loader, test_loader=None, evaluation_epoch=5):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        best_f1, best_state, best_epoch = -1, None, 0
        prev_loss, worse_count = float("inf"), 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0

            for graph, labels in train_loader:
                graph = graph.to(self.device)
                optimizer.zero_grad()
                res = self.model(graph, labels)
                loss = res["loss"]
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.5f}")


            if avg_loss > prev_loss:
                worse_count += 1
                if worse_count >= self.patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                worse_count = 0
            prev_loss = avg_loss


            if epoch % evaluation_epoch == 0 and test_loader:
                test_results = self.evaluate(test_loader)
                if test_results["F1"] > best_f1:
                    best_f1 = test_results["F1"]
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch

        if best_state:
            self.model.load_state_dict(best_state)
            logging.info(f"Best F1 {best_f1:.4f} at epoch {best_epoch}")

        return best_f1, best_epoch