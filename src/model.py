# src/model.py
# ─────────────────────────────────────────────────────────────
# Defines the 3-layer fully connected neural network for digit
# classification. Architecture:
#   Input (784) -> Hidden1 (512) -> Hidden2 (256) -> Output (10)
#
# Key design choices:
#   - ReLU activations to avoid vanishing gradients
#   - Dropout regularisation to prevent overfitting
#   - Xavier (Glorot) weight initialisation for stable training
#   - Softmax applied externally (nn.CrossEntropyLoss handles it)
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn


class DigitClassifier(nn.Module):
    """
    A 3-layer fully connected neural network for classifying
    handwritten digits (0-9) from 28x28 grayscale MNIST images.
    """

    def __init__(self, input_size=784, hidden1=512, hidden2=256,
                 output_size=10, dropout=0.3):
        """
        Args:
            input_size  : number of input features (784 = 28x28)
            hidden1     : number of neurons in first hidden layer
            hidden2     : number of neurons in second hidden layer
            output_size : number of output classes (10 digits)
            dropout     : dropout probability applied after each hidden layer
        """
        super(DigitClassifier, self).__init__()

        # ── Layers ────────────────────────────────────────────
        self.fc1 = nn.Linear(input_size, hidden1)   # 784 → 512
        self.fc2 = nn.Linear(hidden1, hidden2)       # 512 → 256
        self.fc3 = nn.Linear(hidden2, output_size)   # 256 → 10

        # ── Activations & Regularisation ──────────────────────
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        # ── Weight Initialisation ─────────────────────────────
        # Xavier (Glorot) uniform initialisation scales weights
        # based on fan_in and fan_out for stable gradient flow.
        self._init_weights()

    def _init_weights(self):
        """
        Apply Xavier uniform initialisation to all Linear layers.
        Xavier sets W ~ U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out)).
        Biases are initialised to zero.
        """
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass through the network.
        Softmax is NOT applied here — CrossEntropyLoss handles it internally.

        Args:
            x: tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784)
        Returns:
            logits: raw scores of shape (batch_size, 10)
        """
        x = x.view(-1, 28 * 28)          # flatten to (batch, 784)
        x = self.relu(self.fc1(x))        # hidden layer 1 + ReLU
        x = self.dropout(x)               # dropout
        x = self.relu(self.fc2(x))        # hidden layer 2 + ReLU
        x = self.dropout(x)               # dropout
        x = self.fc3(x)                   # output logits
        return x

    def count_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
