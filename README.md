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
```

#### 📦 Installation

```python
pip install robust-layernorm
```
Or from source:

```python
git clone https://github.com/R-LayerNorm/R-LayerNorm.git
cd R-LayerNorm
pip install -e .
```
#### 🧪 Run Experiments

```python
# Full CIFAR-10-C experiment (5 seeds)
python experiments/cifar10c_experiment.py --lambda 0.01 --epochs 10

# Lambda parameter study
python experiments/lambda_ablation.py --lambdas 0.005 0.01 0.02 0.03

# Quick demo in Colab
![Open In Colab][https://drive.google.com/drive/folders/1B5C9UdpO8_uZf7uaW2rs2MSI7Al8vxGy-badge.svg]
```

#### 📝 Citation

If you use R-LayerNorm in your research, please cite:

```python
@article{rlayernorm2026,
  title={R-LayerNorm: Robust Layer Normalization with Adaptive Noise Suppression},
  author={Mohsen Mostafa},
  journal={},
  year={2026}
}
```
