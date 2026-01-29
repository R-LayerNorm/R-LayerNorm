# robust_layernorm.py
import torch
import torch.nn as nn

class RobustLayerNorm(nn.Module):
    """
    R-LayerNorm: LayerNorm with adaptive noise suppression.
    Combines statistical normalization with entropy-based noise weighting.
    """
    def __init__(self, normalized_shape, epsilon=1e-5, lambda_init=0.1):
        super().__init__()
        # Standard LayerNorm parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        
        # R-LayerNorm specific: noise sensitivity parameter
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        self.epsilon = epsilon
        
        # Local entropy estimator (fixed, non-learnable)
        self.is_2d = len(normalized_shape) > 1
        if self.is_2d:
            # For images: simple average pooling for local variance
            self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)

    def estimate_local_entropy(self, x):
        """Cheap entropy approximation via local variance."""
        if self.is_2d:
            # For (B, C, H, W) tensors
            local_mean = self.avg_pool(x)
            local_var = self.avg_pool((x - local_mean) ** 2)
            # Entropy ≈ log(1 + variance) for stability
            entropy = torch.log(1.0 + local_var)
        else:
            # For 1D/sequence data: simplified version
            # Mean over the normalized dimension
            local_var = x.var(dim=-1, keepdim=True)
            entropy = torch.log(1.0 + local_var)
        return entropy

    def forward(self, x):
        # 1. Standard LayerNorm statistics
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.epsilon)
        
        # 2. Estimate local entropy (noise level)
        entropy = self.estimate_local_entropy(x)
        
        # 3. Adaptive noise-aware scaling
        noise_scale = 1.0 + self.lambda_param * entropy
        
        # 4. Robust normalization
        normalized = (x - mean) / (std * noise_scale)
        
        # 5. Apply learnable scale and shift
        output = self.gamma * normalized + self.beta
        return output
