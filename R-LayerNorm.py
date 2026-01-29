class RobustLayerNorm(nn.Module):
    """
    R-LayerNorm: LayerNorm with adaptive noise suppression
    Combines statistical normalization with entropy-based noise weighting
    """
    
    def __init__(self, normalized_shape, epsilon=1e-5, lambda_init=0.1):
        super().__init__()
        
        # Standard LayerNorm parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))  # Scale
        self.beta = nn.Parameter(torch.zeros(normalized_shape))  # Shift
        
        # R-LayerNorm specific: noise sensitivity parameter
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        
        # Numerical stability
        self.epsilon = epsilon
        
        # Local entropy estimator (fixed, non-learnable)
        # For 2D data (images): use average pooling
        self.is_2d = len(normalized_shape) > 1
        if self.is_2d:
            self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
    
    def estimate_local_entropy(self, x):
        """
        Cheap entropy approximation via local variance
        Input: (B, C, H, W) or (B, C, ...)
        Output: entropy map same spatial dim as input
        """
        if self.is_2d:
            # For images: variance in 3x3 neighborhood
            local_mean = self.avg_pool(x)
            local_var = self.avg_pool((x - local_mean)**2)
            # Entropy ≈ log(1 + variance) (bounded approximation)
            entropy = torch.log(1.0 + local_var)
        else:
            # For sequences: variance in sliding window
            # Simplified version for 1D
            local_var = x.var(dim=-1, keepdim=True)
            entropy = torch.log(1.0 + local_var)
        
        return entropy
    
    def forward(self, x):
        """
        Forward pass of R-LayerNorm
        x: input tensor of shape (B, C, H, W) or (B, T, C) or (B, C)
        """
        # 1. Standard LayerNorm statistics
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean)**2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.epsilon)
        
        # 2. Estimate local entropy (noise level)
        entropy = self.estimate_local_entropy(x)
        
        # 3. Adaptive noise-aware scaling
        # More noise (high entropy) → larger denominator → gentler normalization
        noise_scale = 1.0 + self.lambda_param * entropy
        
        # 4. Robust normalization
        normalized = (x - mean) / (std * noise_scale)
        
        # 5. Scale and shift
        output = self.gamma * normalized + self.beta
        
        return output
    
    def extra_repr(self):
        """Display lambda value in model summary"""
        return f"lambda={self.lambda_param.item():.3f}"
