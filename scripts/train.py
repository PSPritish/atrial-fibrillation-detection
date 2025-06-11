import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from models.model_factory import create_model
from data.processed.transformations import get_transformations
from experiments.experiment_manager import ExperimentManager
from utils.callbacks import EarlyStopping
from utils.io_utils import save_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train Atrial Fibrillation Detection Model')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    args = parse_args()
    config = load_config(args.config)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    train_transform, val_transform = get_transformations(config['data'])
    train_dataset = config['data']['train_dataset'](transform=train_transform)
    val_dataset = config['data']['val_dataset'](transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Initialize model
    model = create_model(config['model']).to(device)

    # Define loss function and optimizer
    criterion = config['training']['loss_function']()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Initialize experiment manager and early stopping
    experiment_manager = ExperimentManager(config['experiment'])
    early_stopping = EarlyStopping(patience=config['training']['patience'], verbose=True)

    # Training loop
    for epoch in range(config['training']['num_epochs']):
        model.train()
        for batch in train_loader:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        val_loss, val_accuracy = experiment_manager.validate(model, val_loader, criterion)
        print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Check for early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Save the model
    save_model(model, config['paths']['saved_model_path'])

if __name__ == '__main__':
    main()