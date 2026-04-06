# src/utils.py
# ─────────────────────────────────────────────────────────────
# Utility functions shared across training, evaluation, and app.
# Includes model save/load, image preprocessing, and device setup.
# ─────────────────────────────────────────────────────────────

import torch
import numpy as np
from torchvision import transforms
from PIL import Image


def save_model(model, path="model.pth"):
    """Save model weights to disk."""
    torch.save(model.state_dict(), path)
    print(f"  Model saved → {path}")


def load_model(model, path="model.pth", device="cpu"):
    """
    Load model weights from disk into an existing model instance.
    Sets the model to eval mode after loading.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from {path}")
    return model


def get_device():
    """
    Return the best available device.
    Uses GPU (CUDA) if available, otherwise falls back to CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def is_canvas_empty(gray_array, threshold=10):
    """
    Check if the canvas has any actual drawing.
    Since Gradio 6 sends full white canvas with dark strokes,
    we detect drawing by checking if any pixels are significantly dark.
    """
    dark_pixels = np.sum(gray_array < 200)
    return dark_pixels < threshold


def preprocess_canvas_image(image_array):
    """
    Preprocess a raw canvas image from Gradio 6 into the format
    expected by the model: grayscale, 28x28, normalised.

    Gradio 6 behaviour:
      - Background is white (255), drawn strokes are dark (~0)
      - Alpha channel is always 255 (fully opaque) — cannot be used
      - We must use RGB channels to detect and extract the drawing

    MNIST format:
      - Background is black (0), digit is white (255)
      - So we invert the grayscale image after extraction

    Args:
        image_array: numpy array (280 x 280 x 4, RGBA uint8) from Gradio

    Returns:
        torch.Tensor of shape (1, 1, 28, 28), or None if canvas is empty
    """
    if image_array is None:
        return None

    # Convert to grayscale using RGB channels (ignore alpha)
    rgb   = image_array[:, :, :3].astype(np.float32)
    gray  = np.mean(rgb, axis=2)  # shape (280, 280), values 0–255

    # Check if anything was drawn
    if is_canvas_empty(gray):
        return None

    # Invert: white background (255) → black (0), dark strokes → white (255)
    # This matches MNIST format: white digit on black background
    inverted = (255.0 - gray).astype(np.uint8)

    # Convert to PIL and resize to 28x28
    image = Image.fromarray(inverted, mode="L")
    image = image.resize((28, 28), Image.LANCZOS)

    # Apply the same normalisation used during MNIST training
    transform = transforms.Compose([
        transforms.ToTensor(),                       # [0,255] → [0.0,1.0]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])

    # Add batch dimension: (1, 28, 28) → (1, 1, 28, 28)
    tensor = transform(image).unsqueeze(0)
    return tensor

