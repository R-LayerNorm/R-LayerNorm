#!/usr/bin/env python3
"""
Main CIFAR-10-C experiment for R-LayerNorm
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from robust_layernorm.models import SimpleTestModel
from robust_layernorm.utils import (
    load_cifar10c_corruption,
    create_dataloader,
    evaluate_model,
    set_random_seed
)

# Corruption types to evaluate
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'frost', 'contrast'
]

def train_on_mixed_corruptions(
    model: nn.Module,
    model_name: str = "Model",
    epochs: int = 10,
    device: torch.device = None
) -> nn.Module:
    """Train model on mixture of corruptions"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training {model_name}...")
    
    for epoch in range(epochs):
        # Shuffle corruption order each epoch
        np.random.shuffle(CORRUPTIONS)
        epoch_loss = 0
        batch_count = 0
        
        for corruption in CORRUPTIONS:
            # Load data
            data, labels = load_cifar10c_corruption(corruption, severity=3)
            train_data = data[:1000]
            train_labels = labels[:1000]
            
            # Create dataloader
            dataloader = create_dataloader(
                train_data, train_labels, batch_size=32, shuffle=True, device=device
            )
            
            # Train for a few batches on this corruption
            for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
                if batch_idx >= 3:  # 3 batches (96 images) per corruption
                    break
                
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else epoch_loss
        print(f"  Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='R-LayerNorm CIFAR-10-C Experiment')
    parser.add_argument('--lambda_init', type=float, default=0.01,
                       help='Initial value for lambda parameter')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--data_dir', type=str, default='./data/CIFAR-10-C',
                       help='Directory containing CIFAR-10-C data')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    print("\nCreating models...")
    std_model = SimpleTestModel(use_robust_norm=False).to(device)
    robust_model = SimpleTestModel(use_robust_norm=True, lambda_init=args.lambda_init).to(device)
    
    print(f"Standard model parameters: {std_model.num_parameters:,}")
    print(f"Robust model parameters: {robust_model.num_parameters:,}")
    
    # Train models
    print("\n" + "="*60)
    print("Training Phase")
    print("="*60)
    
    std_model_trained = train_on_mixed_corruptions(std_model, "BatchNorm", args.epochs, device)
    robust_model_trained = train_on_mixed_corruptions(robust_model, "R-LayerNorm", args.epochs, device)
    
    # Test models
    print("\n" + "="*60)
    print("Testing Phase")
    print("="*60)
    
    results = {}
    print(f"{'Corruption':20} | {'BatchNorm':^10} | {'R-LayerNorm':^12} | {'Improvement':^12}")
    print("-" * 65)
    
    for corruption in CORRUPTIONS:
        data, labels = load_cifar10c_corruption(corruption, severity=3, data_dir=args.data_dir)
        test_data = data[1000:1500]  # Unseen images
        test_labels = labels[1000:1500]
        
        std_acc = evaluate_model(std_model_trained, test_data, test_labels, device=device)
        robust_acc = evaluate_model(robust_model_trained, test_data, test_labels, device=device)
        improvement = robust_acc - std_acc
        
        results[corruption] = (std_acc, robust_acc, improvement)
        
        # Color code improvement
        if improvement > 0:
            imp_str = f"\033[92m+{improvement:.2f}%\033[0m"
        else:
            imp_str = f"\033[91m{improvement:.2f}%\033[0m"
        
        print(f"{corruption:20} | {std_acc:9.2f}% | {robust_acc:11.2f}% | {imp_str:>12}")
    
    # Calculate statistics
    improvements = [v[2] for v in results.values()]
    avg_std = np.mean([v[0] for v in results.values()])
    avg_robust = np.mean([v[1] for v in results.values()])
    avg_improvement = np.mean(improvements)
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"Average BatchNorm:      {avg_std:.2f}%")
    print(f"Average R-LayerNorm:    {avg_robust:.2f}%")
    print(f"Average Improvement:    {avg_improvement:+.2f}%")
    
    # Save results
    with open(f'{args.output_dir}/results.txt', 'w') as f:
        f.write("Corruption,BatchNorm,R-LayerNorm,Improvement\n")
        for corruption, (std, robust, imp) in results.items():
            f.write(f"{corruption},{std:.2f},{robust:.2f},{imp:.2f}\n")
        f.write(f"Average,{avg_std:.2f},{avg_robust:.2f},{avg_improvement:.2f}\n")
    
    print(f"\nResults saved to {args.output_dir}/results.txt")

if __name__ == '__main__':
    main()
