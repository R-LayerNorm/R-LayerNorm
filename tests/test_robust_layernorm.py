"""
Unit tests for R-LayerNorm
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from robust_layernorm import RobustLayerNorm, SimpleTestModel

def test_import():
    """Test that the module imports correctly"""
    assert 'RobustLayerNorm' in globals()
    assert 'SimpleTestModel' in globals()

def test_robust_layernorm_initialization():
    """Test initialization of RobustLayerNorm"""
    # Test with integer input
    rln = RobustLayerNorm(64)
    assert isinstance(rln, nn.Module)
    assert rln.gamma.shape == torch.Size([64])
    assert rln.beta.shape == torch.Size([64])
    assert hasattr(rln, 'lambda_param')
    assert rln.lambda_param.item() == pytest.approx(0.01)
    
    # Test with tuple input
    rln2 = RobustLayerNorm((64,))
    assert rln2.gamma.shape == torch.Size([64])
    
    # Test custom lambda_init
    rln3 = RobustLayerNorm(32, lambda_init=0.05)
    assert rln3.lambda_param.item() == pytest.approx(0.05)

def test_forward_pass_2d():
    """Test forward pass with 2D input (convolutional features)"""
    batch_size, channels, height, width = 4, 32, 16, 16
    rln = RobustLayerNorm(32)
    
    # Test input
    x = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    output = rln(x)
    
    # Check output properties
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    # Check that lambda is learnable
    assert rln.lambda_param.requires_grad
    assert rln.gamma.requires_grad
    assert rln.beta.requires_grad

def test_forward_pass_1d():
    """Test forward pass with 1D input (sequence data)"""
    batch_size, seq_len, features = 8, 100, 64
    rln = RobustLayerNorm(64)
    
    # Test input
    x = torch.randn(batch_size, seq_len, features)
    
    # Forward pass
    output = rln(x)
    
    # Check output properties
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_entropy_estimation():
    """Test local entropy estimation function"""
    rln = RobustLayerNorm(32)
    
    # 2D input
    x_2d = torch.randn(4, 32, 16, 16)
    entropy_2d = rln.estimate_local_entropy(x_2d)
    assert entropy_2d.shape == x_2d.shape
    assert entropy_2d.min() >= 0  # log(1 + var) >= 0
    
    # 1D input
    x_1d = torch.randn(4, 100, 32)
    entropy_1d = rln.estimate_local_entropy(x_1d)
    assert entropy_1d.shape[-1] == 1  # Keepdim=True
    
    # Test that entropy is higher for noisy input
    clean = torch.randn(1, 1, 32, 32)
    noisy = clean + torch.randn_like(clean) * 5.0  # Add strong noise
    
    clean_entropy = rln.estimate_local_entropy(clean)
    noisy_entropy = rln.estimate_local_entropy(noisy)
    
    # Noisy should generally have higher entropy
    assert noisy_entropy.mean() > clean_entropy.mean()

def test_gradient_flow():
    """Test that gradients flow through all parameters"""
    rln = RobustLayerNorm(32).double()  # Use double for numerical precision
    
    # Create input and target
    x = torch.randn(4, 32, 16, 16, dtype=torch.double, requires_grad=True)
    target = torch.randn_like(x)
    
    # Forward pass
    output = rln(x)
    
    # Compute loss and backward
    loss = torch.mean((output - target) ** 2)
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert rln.lambda_param.grad is not None
    assert rln.gamma.grad is not None
    assert rln.beta.grad is not None
    
    # Check gradient signs make sense
    assert not torch.isnan(rln.lambda_param.grad).any()

def test_noise_scale_calculation():
    """Test the noise scaling mechanism"""
    rln = RobustLayerNorm(32, lambda_init=0.1)
    
    x = torch.randn(2, 32, 8, 8)
    entropy = rln.estimate_local_entropy(x)
    noise_scale = 1.0 + rln.lambda_param * entropy
    
    # Noise scale should be >= 1.0
    assert noise_scale.min() >= 1.0
    
    # With lambda=0, noise_scale should be exactly 1.0
    rln.lambda_param.data.fill_(0.0)
    noise_scale_zero = 1.0 + rln.lambda_param * entropy
    assert torch.allclose(noise_scale_zero, torch.ones_like(noise_scale_zero))

def test_compatibility_with_optimizer():
    """Test that R-LayerNorm works with optimizers"""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        RobustLayerNorm(16, lambda_init=0.01),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        RobustLayerNorm(32, lambda_init=0.01),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training step
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0  # Should have computed a loss

def test_simple_test_model_creation():
    """Test SimpleTestModel with both normalization types"""
    # With R-LayerNorm
    model_robust = SimpleTestModel(use_robust_norm=True, lambda_init=0.01)
    assert isinstance(model_robust.encoder[1], RobustLayerNorm)
    assert isinstance(model_robust.encoder[4], RobustLayerNorm)
    
    # With BatchNorm
    model_std = SimpleTestModel(use_robust_norm=False)
    assert isinstance(model_std.encoder[1], nn.BatchNorm2d)
    assert isinstance(model_std.encoder[4], nn.BatchNorm2d)
    
    # Parameter count should be similar
    params_robust = sum(p.numel() for p in model_robust.parameters())
    params_std = sum(p.numel() for p in model_std.parameters())
    # R-LayerNorm has 2 extra parameters per layer (lambda)
    assert params_robust - params_std == 4  # 2 layers × 2 params

def test_model_forward():
    """Test SimpleTestModel forward pass"""
    model = SimpleTestModel(use_robust_norm=True, lambda_init=0.01)
    
    # Test input
    x = torch.randn(4, 3, 32, 32)
    
    # Forward pass
    output = model(x)
    
    # Check output
    assert output.shape == (4, 10)  # Batch size × classes
    assert not torch.isnan(output).any()

def test_device_transfer():
    """Test moving R-LayerNorm between devices"""
    if torch.cuda.is_available():
        rln = RobustLayerNorm(32)
        
        # Move to GPU
        rln_gpu = rln.cuda()
        assert rln_gpu.gamma.device.type == 'cuda'
        assert rln_gpu.lambda_param.device.type == 'cuda'
        
        # Test forward on GPU
        x_gpu = torch.randn(4, 32, 16, 16).cuda()
        output_gpu = rln_gpu(x_gpu)
        assert output_gpu.device.type == 'cuda'
        
        # Move back to CPU
        rln_cpu = rln_gpu.cpu()
        assert rln_cpu.gamma.device.type == 'cpu'

def test_state_dict():
    """Test saving and loading state dict"""
    rln1 = RobustLayerNorm(64, lambda_init=0.02)
    
    # Save state
    state_dict = rln1.state_dict()
    
    # Create new instance
    rln2 = RobustLayerNorm(64, lambda_init=0.01)
    
    # Verify they're different initially
    assert not torch.allclose(rln1.lambda_param, rln2.lambda_param)
    
    # Load state
    rln2.load_state_dict(state_dict)
    
    # Verify they're the same now
    assert torch.allclose(rln1.lambda_param, rln2.lambda_param)
    assert torch.allclose(rln1.gamma, rln2.gamma)
    assert torch.allclose(rln1.beta, rln2.beta)

def test_extra_repr():
    """Test the extra representation string"""
    rln = RobustLayerNorm(32, lambda_init=0.015)
    repr_str = rln.extra_repr()
    
    assert 'normalized_shape' in repr_str
    assert 'lambda' in repr_str
    assert '0.015' in repr_str  # Should show lambda value

def test_training_mode_switch():
    """Test that R-LayerNorm handles train/eval modes correctly"""
    rln = RobustLayerNorm(32)
    
    # Initially in train mode
    assert rln.training
    
    # Switch to eval mode
    rln.eval()
    assert not rln.training
    
    # Switch back to train mode
    rln.train()
    assert rln.training

def test_deterministic_behavior():
    """Test that with same seed, R-LayerNorm produces same output"""
    torch.manual_seed(42)
    rln1 = RobustLayerNorm(32, lambda_init=0.01)
    x = torch.randn(2, 32, 8, 8)
    output1 = rln1(x)
    
    torch.manual_seed(42)
    rln2 = RobustLayerNorm(32, lambda_init=0.01)
    output2 = rln2(x)
    
    # Should be identical with same seed
    assert torch.allclose(output1, output2)

def test_lambda_gradient_direction():
    """Test that lambda learns in the right direction"""
    rln = RobustLayerNorm(32, lambda_init=0.01)
    optimizer = torch.optim.SGD([rln.lambda_param], lr=0.1)
    
    # Create synthetic task where more noise suppression helps
    clean = torch.randn(1, 32, 8, 8)
    noise = torch.randn_like(clean) * 2.0
    noisy = clean + noise
    
    # Target is the clean version
    target = clean.clone()
    
    # Train to suppress noise
    for _ in range(10):
        optimizer.zero_grad()
        output = rln(noisy)
        loss = torch.mean((output - target) ** 2)
        loss.backward()
        optimizer.step()
    
    # Lambda should increase to suppress more noise
    assert rln.lambda_param.item() > 0.01

if __name__ == '__main__':
    # Run tests
    print("Running R-LayerNorm tests...")
    
    # Run all test functions
    test_functions = [v for k, v in globals().items() 
                     if k.startswith('test_') and callable(v)]
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
    
    print("\nAll tests completed!")
