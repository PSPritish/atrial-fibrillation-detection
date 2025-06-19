import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Replace with your actual model and dataset classes
from data.dataset import ComplexDataset
from data.transforms import get_transforms
from models.custom_resnet import resnet18


def load_model(model_path, device):
    model = resnet18()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate(model, test_loader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            if outputs.shape[-1] == 1:
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs > 0.5).long()
            else:
                preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def main():
    model_path = "checkpoints/best_model.pth"
    test_data_root = "path/to/test_data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    test_transforms = get_transforms(mode="test")
    test_dataset = ComplexDataset(root=test_data_root, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load and evaluate model
    model = load_model(model_path, device)
    preds, labels = evaluate(model, test_loader, device)

    # Metrics
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")

    plot_confusion_matrix(labels, preds)


if __name__ == "__main__":
    main()
