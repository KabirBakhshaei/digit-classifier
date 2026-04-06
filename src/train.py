# src/train.py
# ─────────────────────────────────────────────────────────────
# Training script for the DigitClassifier on MNIST.
# Reads all settings from config.yaml (no hardcoded values).
#
# Features:
#   - Xavier weight initialisation (in model.py)
#   - Early stopping to avoid wasted epochs
#   - LR scheduler (ReduceLROnPlateau)
#   - Per-epoch CSV log saved to results/training_log.csv
#   - Training/validation loss and accuracy plots saved to results/
#   - Best model saved to model.pth
#
# Run:  cd src && python train.py
# ─────────────────────────────────────────────────────────────

import os
import sys
import csv
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import DigitClassifier
from utils import save_model, get_device


def load_config(path="../config.yaml"):
    """Load hyperparameters and paths from config.yaml."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Early Stopping ────────────────────────────────────────────
class EarlyStopping:
    """
    Monitors a chosen metric (val_loss or val_acc) and stops
    training when no improvement is seen for `patience` epochs.
    Also saves the best model weights automatically.
    """
    def __init__(self, patience=7, min_delta=0.0001,
                 monitor="val_loss", model_path="model.pth"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.monitor    = monitor
        self.model_path = model_path
        self.best       = None
        self.counter    = 0
        self.stop       = False

    def step(self, metric, model):
        """
        Call once per epoch. Returns True if training should stop.
        Saves the model whenever a new best is found.
        """
        # For val_loss: lower is better. For val_acc: higher is better.
        improved = (
            self.best is None or
            (self.monitor == "val_loss"  and metric < self.best - self.min_delta) or
            (self.monitor == "val_acc"   and metric > self.best + self.min_delta)
        )

        if improved:
            self.best    = metric
            self.counter = 0
            save_model(model, self.model_path)
            print(f"  ✔ New best {self.monitor}: {metric:.4f} — model saved.")
        else:
            self.counter += 1
            print(f"  ✘ No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.stop = True

        return self.stop


# ── Data Loading ─────────────────────────────────────────────
def get_data_loaders(cfg):
    """
    Load MNIST with augmentation for training and plain normalisation
    for validation/test. Split settings come from config.yaml.
    """
    aug   = cfg["training"]["augmentation"]
    paths = cfg["paths"]

    # Training: augmentation + normalisation
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=aug["random_rotation"]),
        transforms.RandomAffine(degrees=0,
                                translate=(aug["random_translate"],
                                           aug["random_translate"])),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Validation / test: only normalisation (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.MNIST(root=paths["data_dir"], train=True,
                                download=True, transform=train_transform)
    test_set   = datasets.MNIST(root=paths["data_dir"], train=False,
                                download=True, transform=eval_transform)

    val_size   = int(len(full_train) * cfg["training"]["val_split"])
    train_size = len(full_train) - val_size
    train_set, val_set = torch.utils.data.random_split(
        full_train, [train_size, val_size]
    )

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=bs, shuffle=False)

    print(f"Dataset  →  Train: {train_size} | Val: {val_size} | Test: {len(test_set)}")
    return train_loader, val_loader, test_loader


# ── One Epoch: Train ─────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one full pass over training data. Returns avg loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                      # clear previous gradients
        outputs = model(images)                    # forward pass
        loss    = criterion(outputs, labels)       # compute loss
        loss.backward()                            # backpropagation
        optimizer.step()                           # update weights

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


# ── One Epoch: Validate ───────────────────────────────────────
def validate(model, loader, criterion, device):
    """Evaluate on validation set. Returns avg loss and accuracy."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


# ── Plotting ─────────────────────────────────────────────────
def save_training_plots(history, results_dir):
    """
    Save two plots to results/:
      - training_loss_curve.png   (train vs val loss)
      - training_acc_curve.png    (train vs val accuracy)
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss",    color="steelblue")
    plt.plot(epochs, history["val_loss"],   label="Val Loss",      color="orange", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_loss_curve.png"), dpi=150)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy", color="steelblue")
    plt.plot(epochs, history["val_acc"],   label="Val Accuracy",   color="orange", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_acc_curve.png"), dpi=150)
    plt.close()

    print(f"Training plots saved to {results_dir}/")


# ── Main ─────────────────────────────────────────────────────
def main():
    cfg    = load_config()
    device = get_device()

    # Create output directories if they don't exist
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["assets_dir"],  exist_ok=True)

    train_loader, val_loader, _ = get_data_loaders(cfg)

    # Build model from config
    m_cfg = cfg["model"]
    model = DigitClassifier(
        input_size  = m_cfg["input_size"],
        hidden1     = m_cfg["hidden1"],
        hidden2     = m_cfg["hidden2"],
        output_size = m_cfg["output_size"],
        dropout     = m_cfg["dropout"]
    ).to(device)

    print(f"\nModel parameters: {model.count_parameters():,}")

    t_cfg     = cfg["training"]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=t_cfg["learning_rate"])

    # LR scheduler
    s_cfg     = t_cfg["scheduler"]
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=s_cfg["patience"],
        factor=s_cfg["factor"],
        min_lr=s_cfg["min_lr"]
    )

    # Early stopping
    es_cfg = t_cfg["early_stopping"]
    early_stopping = EarlyStopping(
        patience   = es_cfg["patience"],
        min_delta  = es_cfg["min_delta"],
        monitor    = es_cfg["monitor"],
        model_path = cfg["paths"]["model_path"]
    )

    # ── Training Loop ─────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  []
    }

    # CSV log file — one row per epoch
    csv_path = os.path.join(cfg["paths"]["results_dir"], "training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow(["epoch", "train_loss", "val_loss",
                     "train_acc", "val_acc", "lr"])

    print("\n── Starting Training ──────────────────────────────────")
    for epoch in range(1, t_cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(
            model, val_loader, criterion, device)

        # Step the LR scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log to console
        print(f"Epoch {epoch:03d}/{t_cfg['epochs']} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

        # Save to history and CSV
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                         f"{train_acc:.4f}", f"{val_acc:.4f}", current_lr])

        # Early stopping check (also saves best model inside)
        monitor_val = val_loss if es_cfg["monitor"] == "val_loss" else val_acc
        if early_stopping.step(monitor_val, model):
            print(f"\n⚡ Early stopping triggered at epoch {epoch}.")
            break

    csv_file.close()

    # Save training summary as JSON
    summary = {
        "epochs_trained":    len(history["train_loss"]),
        "best_val_loss":     min(history["val_loss"]),
        "best_val_acc":      max(history["val_acc"]),
        "final_train_loss":  history["train_loss"][-1],
        "final_train_acc":   history["train_acc"][-1],
        "model_parameters":  model.count_parameters(),
        "config":            cfg
    }
    summary_path = os.path.join(cfg["paths"]["results_dir"], "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    # Save plots
    save_training_plots(history, cfg["paths"]["results_dir"])

    print(f"\n✅ Training complete!")
    print(f"   Best val loss : {summary['best_val_loss']:.4f}")
    print(f"   Best val acc  : {summary['best_val_acc']:.2f}%")
    print(f"   Epochs trained: {summary['epochs_trained']}")
    print(f"   CSV log       : {csv_path}")
    print(f"   Summary JSON  : {summary_path}")
    print(f"   Model saved   : {cfg['paths']['model_path']}")


if __name__ == "__main__":
    main()

