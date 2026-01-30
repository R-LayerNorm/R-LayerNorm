"""
R-LayerNorm: Robust Layer Normalization with Adaptive Noise Suppression
Core implementation
"""

import torch
import torch.nn as nn

class RobustLayerNorm(nn.Module):
    """
    R-LayerNorm: LayerNorm with adaptive noise suppression.
    
    Args:
        normalized_shape (int or tuple): Shape of input to normalize
        epsilon (float): Small constant for numerical stability
        lambda_init (float): Initial value for noise sensitivity parameter
    
    Formula:
        R-LN(x) = γ * [(x - μ) / (σ * (1 + λ * E(x)))] + β
        where E(x) ≈ log(1 + local_variance(x))
    """
    
    def __init__(self, normalized_shape, epsilon=1e-5, lambda_init=0.01):
        super().__init__()
        
        # Convert to tuple if integer
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        
        # Learnable parameters (same as standard LayerNorm)
        self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
        self.beta = nn.Parameter(torch.zeros(self.normalized_shape))
        
        # R-LayerNorm specific: noise sensitivity parameter
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        self.epsilon = epsilon
        
        # Local entropy estimator (non-learnable)
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
    
    def estimate_local_entropy(self, x):
        """
        Estimate local noise level via spatial entropy approximation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            entropy: Local entropy map of same spatial dimensions
        """
        if x.dim() == 4:  # 2D convolutional data
            local_mean = self.avg_pool(x)
            local_var = self.avg_pool((x - local_mean) ** 2)
            # Entropy ≈ log(1 + variance) for stability
            entropy = torch.log(1.0 + local_var)
        else:  # 1D/sequence data
            local_var = x.var(dim=-1, keepdim=True)
            entropy = torch.log(1.0 + local_var)
        
        return entropy
    
    def forward(self, x):
        """
        Forward pass of R-LayerNorm.
        
        Args:
            x: Input tensor
        Returns:
            Normalized tensor with adaptive noise suppression
        """
        if x.dim() == 4:  # Handle 2D data (B, C, H, W)
            B, C, H, W = x.shape
            
            # Standard LayerNorm statistics (over spatial dimensions)
            mean = x.mean(dim=[-2, -1], keepdim=True)
            variance = ((x - mean) ** 2).mean(dim=[-2, -1], keepdim=True)
            std = torch.sqrt(variance + self.epsilon)
            
            # Estimate local entropy (noise level)
            entropy = self.estimate_local_entropy(x)
            
            # Adaptive noise-aware scaling
            noise_scale = 1.0 + self.lambda_param * entropy
            
            # Robust normalization
            normalized = (x - mean) / (std * noise_scale)
            
            # Apply learnable scale and shift
            gamma = self.gamma.view(1, C, 1, 1)
            beta = self.beta.view(1, C, 1, 1)
            output = gamma * normalized + beta
            
            return output
        else:  # Fallback for 1D/sequence data
            # Standard LayerNorm
            mean = x.mean(dim=-1, keepdim=True)
            variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
            std = torch.sqrt(variance + self.epsilon)
            
            # Estimate entropy
            entropy = self.estimate_local_entropy(x)
            noise_scale = 1.0 + self.lambda_param * entropy
            
            # Robust normalization
            normalized = (x - mean) / (std * noise_scale)
            output = self.gamma * normalized + self.beta
            
            return output
    
    def extra_repr(self):
        """Display layer configuration"""
        return f"normalized_shape={self.normalized_shape}, lambda={self.lambda_param.item():.4f}"
