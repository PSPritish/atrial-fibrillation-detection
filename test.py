import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import os
from torchvision import models  # Add this import
from data.dataloader import get_dataloaders, load_config
from models.complex_resnet import complex_resnet18


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test atrial fibrillation detection model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./saved_models/best_model.pth",
        help="Path to the saved model weights",
    )
    parser.add_argument(
        "--complex", action="store_true", help="Use complex-valued model"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize confusion matrix"
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save results to file"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help="Model architecture (resnet, alexnet)",
    )
    return parser.parse_args()


def evaluate_model(model, test_loader, device):
    """Evaluate model on test data"""
    model.eval()

    all_preds = []
    all_labels = []
    test_loss = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device).float()

            # Forward pass
            outputs = model(inputs)

            # If complex model, take magnitude
            if torch.is_complex(outputs):
                outputs = outputs.abs()

            # Compute loss
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item() * inputs.size(0)

            # Store predictions and labels
            preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    test_loss /= len(test_loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_preds == all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    # Print results
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Normal", "AF"]))

    return {
        "loss": test_loss,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
    }


def visualize_results(results):
    """Visualize confusion matrix"""
    plt.figure(figsize=(8, 6))
    cm = results["confusion_matrix"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "AF"],
        yticklabels=["Normal", "AF"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Accuracy: {results['accuracy']:.4f})")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()


def main():
    args = parse_args()

    # Load configuration
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine architecture to use
    architecture = args.architecture or config.get("model", {}).get(
        "architecture", "resnet"
    )

    # Load data
    if args.complex:
        dataloaders = get_dataloaders(dataset_type="complex")
    else:
        dataloaders = get_dataloaders(dataset_type="gadf")  # Use gadf as default

    test_loader = dataloaders["test"]

    # Create model based on architecture
    if architecture.lower() == "alexnet":
        model = models.AlexNet(num_classes=1)
        print("Using AlexNet model")
    elif args.complex:
        model = complex_resnet18(config)
        print("Using complex ResNet model")
    else:
        model = AFResNet(config)
        print("Using standard ResNet model")

    # Load model weights
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    print(f"Loaded model from {args.model_path}")

    # Evaluate model
    results = evaluate_model(model, test_loader, device)

    # Visualize results
    if args.visualize:
        visualize_results(results)

    # Save results
    if args.save_results:
        np.save(
            "test_results.npy",
            {
                "predictions": results["predictions"],
                "labels": results["labels"],
                "accuracy": results["accuracy"],
                "loss": results["loss"],
            },
        )
        print("Results saved to test_results.npy")


if __name__ == "__main__":
    main()
