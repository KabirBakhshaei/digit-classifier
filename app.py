# app.py
# ------------------------------------------------------------
# Gradio-based web interface for real-time digit classification.
# Draw a digit (0-9) on the canvas and the model predicts it
# instantly, showing confidence scores for all 10 classes.
#
# Run with:  python app.py
# Make sure you have trained the model first: python src/train.py
# ------------------------------------------------------------

import sys
import os
import torch
import numpy as np
import gradio as gr

# Add src/ to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model import DigitClassifier
from utils import load_model, get_device, preprocess_canvas_image

# ── Config ───────────────────────────────────────────────────
MODEL_PATH = "src\model.pth"
# ─────────────────────────────────────────────────────────────

# Load the trained model once at startup
device = get_device()
model  = DigitClassifier().to(device)
model  = load_model(model, MODEL_PATH, device)


def predict_digit(image):
    if image is None:
        return {}

    img_array = image["composite"]
    tensor = preprocess_canvas_image(img_array)
    if tensor is None:
        return {}

    tensor = tensor.to(device)
    model.eval()
    with torch.no_grad():
        output        = model(tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)

    probs = probabilities.squeeze().cpu().numpy()

    # Low confidence check — if max probability < 0.6 show warning
    if float(np.max(probs)) < 0.6:
        return {"Low confidence — please redraw": 1.0}

    return {str(i): float(probs[i]) for i in range(10)}


# ── Build the Gradio Interface ────────────────────────────────
with gr.Blocks(title="Digit Classifier", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # ✏️ Handwritten Digit Classifier
        **Draw a digit (0–9) on the canvas below and the model will predict it in real time.**

        *Project 2 – Neural Networks & Deep Learning | Sant'Anna School of Advanced Studies*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            # Drawing canvas where the user draws with the mouse
            sketch = gr.Sketchpad(
                label="Draw a digit here",
                canvas_size=(280, 280),
                brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=20),
            )
            with gr.Row():
                clear_btn   = gr.Button("🗑️ Clear",   variant="secondary")
                submit_btn  = gr.Button("🔍 Predict", variant="primary")

        with gr.Column(scale=1):
            # Shows confidence scores for each digit class
            label_output = gr.Label(
                label="Prediction Confidence",
                num_top_classes=10
            )

    # Trigger prediction on button click or when drawing changes
    submit_btn.click(fn=predict_digit, inputs=sketch, outputs=label_output)
    sketch.change(fn=predict_digit,    inputs=sketch, outputs=label_output)
    clear_btn.click(fn=lambda: None,   inputs=None,   outputs=label_output)

    gr.Markdown(
        """
        ---
        **How it works:** The drawing is converted to a 28×28 grayscale image,
        normalized, and passed through a 3-layer fully connected neural network
        trained on the MNIST dataset (~99% test accuracy).
        """
    )

if __name__ == "__main__":
    demo.launch(share=False)
