"""Network shared by every classifier, so differences come from the loss only."""
import torch.nn as nn


class SmallCNN(nn.Module):
    """Light CNN (CPU friendly): 2 conv blocks + 2 FC layers. Input (N, C, H, W) -> logits (N, 3)."""

    def __init__(self, in_ch, n_cls=3):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Dropout(0.25),
            nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.f(x)
