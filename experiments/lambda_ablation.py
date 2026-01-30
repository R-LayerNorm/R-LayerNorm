#!/usr/bin/env python3
"""
Lambda parameter ablation study
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from robust_layernorm.models import SimpleTestModel
from robust_layernorm.utils import (
    load_cifar10c_corruption,
    evaluate_model,
    set_random_seed
)

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'frost', 'contrast'
]

def quick_lambda_test(lambda_init: float, device: torch.device, seed: int = 42):
    """Quick test for a specific lambda value"""
    set_random_seed(seed)
    
    # Create and train model
    model = SimpleTestModel(use_robust_norm=True, lambda_init=lambda_init).to(device)
    
    # Simple training (reduced for speed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train on one corruption only for quick test
    data, labels = load_cifar10c_corruption('gaussian_noise', severity=3)
    train_data = data[:500]
    train_labels = labels[:500]
    
    # Convert to tensor
    train_data_tensor = torch.tensor(train_data).float().permute(0, 3, 1, 2) / 255.0
    train_labels_tensor = torch.tensor(train_labels)
    
    dataset = torch.utils.data.TensorDataset(train_data_tensor.to(device), 
                                            train_labels_tensor.to(device))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Quick training (3 epochs)
    for epoch in range(3):
        for batch_data, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    
    # Quick evaluation
    test_data = data[1000:1200]  # Small test set
    test_labels = labels[1000:1200]
    
    accuracy = evaluate_model(model, test_data, test_labels, device=device)
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Lambda Parameter Ablation Study')
    parser.add_argument('--lambdas', type=float, nargs='+', 
                       default=[0.005, 0.01, 0.02, 0.03],
                       help='Lambda values to test')
    parser.add_argument('--data_dir', type=str, default='./data/CIFAR-10-C',
                       help='Directory containing CIFAR-10-C data')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Lambda Ablation Study")
    print("="*50)
    
    results = {}
    for lambda_val in args.lambdas:
        print(f"\nTesting λ = {lambda_val:.3f}")
        
        # Test on multiple seeds for reliability
        accuracies = []
        for seed in [42, 123, 456]:
            acc = quick_lambda_test(lambda_val, device, seed)
            accuracies.append(acc)
            print(f"  Seed {seed}: {acc:.2f}%")
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        results[lambda_val] = (mean_acc, std_acc)
        
        print(f"  Average: {mean_acc:.2f}% ± {std_acc:.2f}")
    
    # Find best lambda
    best_lambda = max(results.keys(), key=lambda x: results[x][0])
    best_acc = results[best_lambda][0]
    
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    for lambda_val in sorted(results.keys()):
        mean_acc, std_acc = results[lambda_val]
        marker = " ← BEST" if lambda_val == best_lambda else ""
        print(f"λ = {lambda_val:.3f}: {mean_acc:.2f}% ± {std_acc:.2f}{marker}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    lambda_vals = sorted(results.keys())
    means = [results[lambda_val][0] for lambda_val in lambda_vals]
    stds = [results[lambda_val][1] for lambda_val in lambda_vals]
    
    plt.errorbar(lambda_vals, means, yerr=stds, fmt='o-', capsize=5, linewidth=2)
    plt.axvline(x=best_lambda, color='red', linestyle='--', alpha=0.5, label=f'Best λ = {best_lambda}')
    plt.xlabel('Lambda Value (λ)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Effect of λ Parameter on R-LayerNorm Performance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plot_path = f'{args.output_dir}/lambda_ablation.png'
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    
    # Save results
    with open(f'{args.output_dir}/lambda_results.csv', 'w') as f:
        f.write("lambda,mean_accuracy,std_accuracy\n")
        for lambda_val in sorted(results.keys()):
            mean_acc, std_acc = results[lambda_val]
            f.write(f"{lambda_val},{mean_acc:.4f},{std_acc:.4f}\n")
    
    print(f"Results saved to {args.output_dir}/lambda_results.csv")

if __name__ == '__main__':
    main()
