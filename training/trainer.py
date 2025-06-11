import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import time
import numpy as np


class Trainer:
    def __init__(self, model, dataloaders, config):
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Set up loss function
        self.criterion = self._get_loss_function()

        # Set up optimizer
        self.optimizer = self._get_optimizer()

    def _get_loss_function(self):
        loss_name = self.config.get("training", {}).get("loss_function", "bce")
        if loss_name == "focal_loss":
            from models.components.losses import FocalLoss

            return FocalLoss()
        else:
            return nn.BCEWithLogitsLoss()

    def _get_optimizer(self):
        lr = self.config.get("training", {}).get("learning_rate", 0.001)
        optimizer_name = self.config.get("training", {}).get("optimizer", "Adam")

        if optimizer_name == "Adam":
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == "SGD":
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            return optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        epochs = self.config.get("training", {}).get("epochs", 50)
        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for inputs, labels in tqdm(self.dataloaders["train"]):
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in self.dataloaders["val"]:
                    inputs, labels = (
                        inputs.to(self.device),
                        labels.to(self.device).float(),
                    )
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels.unsqueeze(1))
                    val_loss += loss.item() * inputs.size(0)

            # Calculate epoch losses
            train_loss = train_loss / len(self.dataloaders["train"].dataset)
            val_loss = val_loss / len(self.dataloaders["val"].dataset)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model()

    def _save_model(self):
        save_path = self.config.get("logging", {}).get(
            "model_save_path", "./saved_models"
        )
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth"))
