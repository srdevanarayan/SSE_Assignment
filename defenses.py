"""
defenses.py - Extra blue-team defense used at prediction time.

Feature squeezing (bit-depth reduction): turn every pixel into pure black (0)
or pure white (1) before the model sees it. Small adversarial nudges
(e.g. a background pixel going from 0.0 to 0.2) are "squeezed" away.
"""

import torch.nn as nn

import config


class SqueezedModel(nn.Module):
    """Wraps any model: binarize the input, then classify."""

    def __init__(self, model, threshold=config.SQUEEZE_THRESHOLD):
        super().__init__()
        self.model = model
        self.threshold = threshold

    def forward(self, x):
        binary = (x > self.threshold).float()
        # Straight-through trick: the forward pass uses the binarized image, but
        # gradients flow back as if nothing happened. Without this the gradient
        # would be zero and attacks would fail for the WRONG reason (gradient
        # masking), giving fake robustness numbers. This is the "BPDA" attack
        # (Athalye et al., 2018), so our white-box test is an honest adaptive one.
        x = x + (binary - x).detach()
        return self.model(x)
