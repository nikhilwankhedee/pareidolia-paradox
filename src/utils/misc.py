"""Shared deterministic utilities."""
from __future__ import annotations

import numpy as np
import torch


def set_seed(seed: int):
    """Fix random seeds across numpy/torch for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def section(title: str, width: int = 90):
    """Print a prominent section divider."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
