# R-LayerNorm: Robust Layer Normalization with Adaptive Noise Suppression

[![PyPI version](https://img.shields.io/pypi/v/robust-layernorm.svg)](https://pypi.org/project/robust-layernorm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **R-LayerNorm**: A noise-aware normalization layer that adapts normalization strength based on local entropy estimates.

## 📊 Key Results
- **+4.95% average improvement** on CIFAR-10-C over BatchNorm (p < 0.001)
- **+14.52% improvement** on contrast corruption
- Statistically significant across 5 random seeds
- Minimal computational overhead (~10%)

## 🚀 Quick Start

```python
import torch
import torch.nn as nn
from robust_layernorm import RobustLayerNorm

# Replace any normalization layer
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    RobustLayerNorm(32, lambda_init=0.01),  # ← Your new layer
    nn.ReLU(),
    nn.MaxPool2d(2)
)
