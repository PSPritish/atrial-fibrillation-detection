import argparse
import os
import yaml
import torch
from models.model_factory import create_model
from data.processed.transformations import load_data
from experiments.metrics import calculate_metrics

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(config['model']).to(device)
    model.load_state_dict(torch.load(config['model']['checkpoint_path']))
    
    data_loader = load_data(config['data'])
    
    preds, labels = evaluate_model(model, data_loader, device)
    
    metrics = calculate_metrics(labels, preds)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    
    main(args.config)