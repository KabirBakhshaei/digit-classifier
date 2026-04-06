# src/evaluate.py
# ─────────────────────────────────────────────────────────────
# Evaluates the trained model on the MNIST test set.
# Reads settings from config.yaml.
#
# Outputs saved to results/:
#   - evaluation_results.json   → all metrics (use these in your report)
#   - confusion_matrix.png      → heatmap of predictions vs true labels
#   - assets/confusion_matrix.png (copy for report figures)
#
# Console output:
#   - Test accuracy
#   - Per-class precision, recall, F1
#
# Run:  cd src && python evaluate.py
# ─────────────────────────────────────────────────────────────

import os
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from model import DigitClassifier
from utils import load_model, get_device

DIGIT_LABELS = [str(i) for i in range(10)]


def load_config(path="../config.yaml"):
    """Load configuration from config.yaml."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_test_loader(cfg):
    """Load MNIST test set with standard normalisation (no augmentation)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST(
        root=cfg["paths"]["data_dir"],
        train=False, download=True, transform=transform
    )
    return DataLoader(test_set, batch_size=128, shuffle=False)


def get_predictions(model, loader, device):
    """Run inference over the full test set. Returns true labels and predictions."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(labels, preds, save_path):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=DIGIT_LABELS, yticklabels=DIGIT_LABELS)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label",      fontsize=12)
    plt.title("Confusion Matrix — MNIST Digit Classifier", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main():
    cfg    = load_config()
    device = get_device()

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["assets_dir"],  exist_ok=True)

    # Load model
    m_cfg = cfg["model"]
    model = DigitClassifier(
        input_size  = m_cfg["input_size"],
        hidden1     = m_cfg["hidden1"],
        hidden2     = m_cfg["hidden2"],
        output_size = m_cfg["output_size"],
        dropout     = m_cfg["dropout"]
    ).to(device)
    model = load_model(model, cfg["paths"]["model_path"], device)

    test_loader = get_test_loader(cfg)
    labels, preds = get_predictions(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────
    accuracy = accuracy_score(labels, preds) * 100
    report   = classification_report(
        labels, preds,
        target_names=DIGIT_LABELS,
        output_dict=True
    )

    print(f"\n{'='*55}")
    print(f"  Test Accuracy : {accuracy:.2f}%")
    print(f"{'='*55}")
    print(classification_report(labels, preds, target_names=DIGIT_LABELS))

    # ── Save results to JSON ──────────────────────────────────
    # These numbers are what you copy into your report tables.
    results = {
        "test_accuracy_percent": round(accuracy, 4),
        "per_class_metrics": {
            digit: {
                "precision": round(report[digit]["precision"], 4),
                "recall":    round(report[digit]["recall"],    4),
                "f1_score":  round(report[digit]["f1-score"],  4),
                "support":   int(report[digit]["support"])
            }
            for digit in DIGIT_LABELS
        },
        "macro_avg": {
            "precision": round(report["macro avg"]["precision"], 4),
            "recall":    round(report["macro avg"]["recall"],    4),
            "f1_score":  round(report["macro avg"]["f1-score"],  4),
        },
        "weighted_avg": {
            "precision": round(report["weighted avg"]["precision"], 4),
            "recall":    round(report["weighted avg"]["recall"],    4),
            "f1_score":  round(report["weighted avg"]["f1-score"],  4),
        },
        "total_test_samples": len(labels),
        "model_parameters": model.count_parameters()
    }

    results_path = os.path.join(cfg["paths"]["results_dir"],
                                "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Evaluation results saved to {results_path}")
    print("   → Use these numbers to fill in your report tables.\n")

    # ── Confusion Matrix ──────────────────────────────────────
    cm_path = os.path.join(cfg["paths"]["results_dir"], "confusion_matrix.png")
    plot_confusion_matrix(labels, preds, cm_path)

    # Also copy to assets/ for LaTeX \includegraphics
    assets_cm = os.path.join(cfg["paths"]["assets_dir"], "confusion_matrix.png")
    plot_confusion_matrix(labels, preds, assets_cm)


if __name__ == "__main__":
    main()

