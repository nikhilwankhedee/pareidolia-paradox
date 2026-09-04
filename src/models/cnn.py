"""Small deliberate CNN for signal localization (Experiment 3, Models B-E).

Deliberately simple -- the goal is signal detection, not SOTA. Used identically
for the raw-image (B/C) and official-rotated-image (D/E) ablations.

Input : 1-channel (grayscale) 256x256.
Arch  : 4x (Conv3x3 -> BatchNorm -> ReLU -> MaxPool) -> AdaptiveAvgPool -> Linear.
Param : ~97.8k.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, use_azimuth: bool = False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        feat = 128 + (2 if use_azimuth else 0)
        self.fc = nn.Linear(feat, 1)
        self.use_azimuth = use_azimuth

    def forward(self, x, az_sin=None, az_cos=None):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.use_azimuth:
            a = torch.stack([az_sin, az_cos], dim=1)
            x = torch.cat([x, a], dim=1)
        return self.fc(x).squeeze(1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
