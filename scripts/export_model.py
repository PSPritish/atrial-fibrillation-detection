import os
import argparse
import torch
from models.model_factory import create_model

def export_model(model, export_path):
    torch.save(model.state_dict(), export_path)
    print(f"Model exported to {export_path}")

def main(config_path, model_name, export_dir):
    # Load model configuration
    model = create_model(model_name)
    
    # Ensure export directory exists
    os.makedirs(export_dir, exist_ok=True)
    
    # Define export path
    export_path = os.path.join(export_dir, f"{model_name}.pth")
    
    # Export the model
    export_model(model, export_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained model for deployment.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration file.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to export.")
    parser.add_argument("--export_dir", type=str, default="saved_models", help="Directory to save the exported model.")
    
    args = parser.parse_args()
    
    main(args.config, args.model_name, args.export_dir)