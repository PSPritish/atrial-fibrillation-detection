import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
import gc
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from models.custom_resnet import resnet18
from models.complex_resnet import complex_resnet18
from models.hybrid_resnet import hybrid_resnet18
from models.hybrid_resnet_RO import hybrid_resnet_RO_18
from models.DualStreamPhaseMagNet import dual_stream_phase_mag_resnet_18
from data.dataset import GADFDataset, GASFDataset, ComplexDataset
from data.transforms import get_transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path="config/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)["default_config"]


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_dataloader(dataset_class, transform, batch_size=32):
    test_dataset = dataset_class(mode="test", transforms=transform)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    ), len(test_dataset)


def test_model(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            if torch.is_complex(out):
                out = out.abs()
            probs = torch.sigmoid(out).squeeze()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_probs), np.array(all_labels)


def calculate_metrics(true_labels, avg_probs):
    preds = (avg_probs > 0.5).astype(float)
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds)
    rec = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    fpr, tpr, _ = roc_curve(true_labels, avg_probs)
    roc_auc = auc(fpr, tpr)
    pr_prec, pr_rec, _ = precision_recall_curve(true_labels, avg_probs)
    pr_auc = average_precision_score(true_labels, avg_probs)
    cm = confusion_matrix(true_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": specificity,
        "npv": npv,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }


def load_model(model_fn, model_path, config):
    model = model_fn(config)
    model = nn.DataParallel(model).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def main():
    clear_memory()
    config = load_config()
    transforms = get_transforms()

    # -------------------- Ensemble Models -------------------------
    models_info = [
        {
            "name": "ResNet-18 GADF",
            "model_fn": resnet18,
            "dataset": GADFDataset,
            "path": "saved_models/ResNet18_GADF/best_model.pth",
        },
        {
            "name": "Complex ResNet-18 GADF",
            "model_fn": complex_resnet18,
            "dataset": GADFDataset,
            "path": "saved_models/Complex_ResNet18_GADF/best_model.pth",
        },
        {
            "name": "Hybrid ResNet18 GASF+iGADF",
            "model_fn": hybrid_resnet18,
            "dataset": ComplexDataset,
            "path": "saved_models/Hybrid_ResNet18_GASF_iGADF/best_model.pth",
        },
        {
            "name": "DualStreamPhaseMagNet",
            "model_fn": dual_stream_phase_mag_resnet_18,
            "dataset": ComplexDataset,
            "path": "saved_models/DualStreamPhaseMagNet/best_model.pth",
        },
    ]

    all_probs = []
    all_labels = None

    for m in models_info:
        print(f"\n→ Testing: {m['name']}")
        model = load_model(m["model_fn"], m["path"], config)
        loader, size = get_dataloader(m["dataset"], transforms)
        probs, labels = test_model(model, loader)
        all_probs.append(probs)
        if all_labels is None:
            all_labels = labels

    # -------------- Soft Voting ----------------
    avg_probs = np.mean(np.vstack(all_probs), axis=0)
    metrics = calculate_metrics(all_labels, avg_probs)

    # ------------------ Report ------------------
    print("\n=== Ensemble Model Performance ===")
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"F1 Score      : {metrics['f1']:.4f}")
    print(f"Specificity   : {metrics['specificity']:.4f}")
    print(f"NPV           : {metrics['npv']:.4f}")
    print(f"ROC AUC       : {metrics['roc_auc']:.4f}")
    print(f"PR AUC        : {metrics['pr_auc']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    print("==================================================")


if __name__ == "__main__":
    main()
