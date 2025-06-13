import os
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataloader import get_dataloaders
from train import Trainer
from models.complex_resnet import complex_resnet18
from torchvision import models
import yaml
import time
import pandas as pd
import wandb


def load_config(config_path=None):
    """Load configuration from YAML file"""
    if config_path is None:
        # Get the path to the default config
        config_dir = os.path.join(os.path.dirname(__file__), "config")
        config_path = os.path.join(config_dir, "default.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config["default_config"]


def main():
    # Load config from YAML
    config = load_config()

    # Extract variables for use in script
    learning_rate = config.get("training", {}).get("learning_rate", 0.001)
    epochs = config.get("training", {}).get("epochs", 5)
    batch_size = config.get("training", {}).get("batch_size", 32)
    save_dir = config.get("logging", {}).get("model_save_path", "./saved_models")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Create run name
    run_name = f"{config.get('run_name', 'nill')}_{time.strftime('%Y%m%d_%H%M%S')}"

    # Initialize wandb with the FULL config from YAML
    wandb.login(key=config.get("wandb_api_key", ""))
    wandb.init(
        entity="CVNN_Based_Networks",
        project=config.get("model", {}).get("architecture", "AF_Detection"),
        name=run_name,
        config=config,
        settings={"console": "wrap"},  # Modern way to capture terminal output
    )

    # Get data loaders (complex data for complex model)
    dataloaders = get_dataloaders(dataset_type="gadf")

    # Create model directly
    model = models.AlexNet(num_classes=1)

    # Log model architecture to wandb
    wandb.watch(model, log="all")

    # Set up simple loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set up trainer
    config["use_wandb"] = True
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        config=config,
        criterion=criterion,
        optimizer=optimizer,
    )
    # Calculate training time
    start_time = time.time()
    print("\nStarting training...")
    history = trainer.train()
    print(f"\nTraining complete. Model saved to {save_dir}")
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    # Save training history to CSV
    history_df = pd.DataFrame(history)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    history_file = os.path.join(save_dir, f"training_history_{timestamp}.csv")
    history_df.to_csv(history_file, index=False)
    print(f"Training history saved to {history_file}")

    # Log final metrics to wandb
    best_epoch_idx = history_df["val_loss"].idxmin()
    best_metrics = history_df.iloc[best_epoch_idx]
    wandb.log(
        {
            "best_val_loss": best_metrics["val_loss"],
            "best_val_acc": best_metrics["val_acc"],
            "best_f1": best_metrics["f1"],
            "best_epoch": best_metrics["Epoch"],
        }
    )

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
