import torch
from models.resnet import AFResNet
from data.dataloader import get_dataloaders, load_config
from training.trainer import Trainer
import argparse


def main():
    parser = argparse.ArgumentParser(description="AF Detection Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--dataset",
        type=str,
        default="combined",
        choices=["combined", "gasf", "gadf", "complex"],
        help="Dataset type to use",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get dataloaders
    dataloaders = get_dataloaders(dataset_type=args.dataset, config_path=args.config)

    # Create model
    model = AFResNet(config)

    # Create trainer and train
    trainer = Trainer(model, dataloaders, config)
    trainer.train()


if __name__ == "__main__":
    main()
