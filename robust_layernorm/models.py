"""
Example models for testing R-LayerNorm
"""

import torch
import torch.nn as nn
from .core import RobustLayerNorm

class SimpleTestModel(nn.Module):
    """
    Simple CNN model for comparing normalization layers on CIFAR-10-C
    
    Args:
        use_robust_norm (bool): If True, use R-LayerNorm; else use BatchNorm
        lambda_init (float): Initial value for λ parameter (if using R-LayerNorm)
    """
    
    def __init__(self, use_robust_norm=True, lambda_init=0.01):
        super().__init__()
        
        if use_robust_norm:
            NormLayer = lambda dim: RobustLayerNorm(dim, lambda_init=lambda_init)
            norm_kwargs = {}
        else:
            NormLayer = nn.BatchNorm2d
            norm_kwargs = {}
        
        self.encoder = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, 3, padding=1),
            NormLayer(32, **norm_kwargs),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second block  
            nn.Conv2d(32, 64, 3, padding=1),
            NormLayer(64, **norm_kwargs),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Classifier
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.encoder(x)
    
    @property
    def num_parameters(self):
        """Total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleResNetBlock(nn.Module):
    """
    Simple ResNet-style block with optional R-LayerNorm
    """
    
    def __init__(self, in_channels, out_channels, use_robust_norm=True, lambda_init=0.01):
        super().__init__()
        
        if use_robust_norm:
            NormLayer = lambda dim: RobustLayerNorm(dim, lambda_init=lambda_init)
            norm_kwargs = {}
        else:
            NormLayer = nn.BatchNorm2d
            norm_kwargs = {}
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = NormLayer(out_channels, **norm_kwargs)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = NormLayer(out_channels, **norm_kwargs)
        
        # Shortcut connection if dimensions change
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out += shortcut
        out = self.relu(out)
        
        return out
