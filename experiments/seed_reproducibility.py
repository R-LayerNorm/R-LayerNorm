#!/usr/bin/env python3
"""
Statistical significance test with multiple random seeds
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

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

def run_experiment_for_seed(seed: int, lambda_init: float, device: torch.device):
    """Run complete experiment for a specific seed"""
    set_random_seed(seed)
    
    # Create models
    std_model = SimpleTestModel(use_robust_norm=False).to(device)
    robust_model = SimpleTestModel(use_robust_norm=True, lambda_init=lambda_init).to(device)
    
    # Simplified training (for reproducibility test)
    optimizer_std = torch.optim.Adam(std_model.parameters(), lr=0.001)
    optimizer_robust = torch.optim.Adam(robust_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train on mixed corruptions (simplified)
    for epoch in range(5):
        for corruption in CORRUPTIONS:
            data, labels = load_cifar10c_corruption(corruption, severity=3)
            train_data = data[:300]  # Small subset for speed
            train_labels = labels[:300]
            
            # Convert to tensor
            data_tensor = torch.tensor(train_data).float().permute(0, 3, 1, 2) / 255.0
            labels_tensor = torch.tensor(labels[:300])
            
            dataset = torch.utils.data.TensorDataset(
                data_tensor.to(device), labels_tensor.to(device)
            )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
            
            # Train for a few batches
            for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
                if batch_idx >= 2:  # 2 batches per corruption
                    break
                
                # Train standard model
                optimizer_std.zero_grad()
                outputs = std_model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer_std.step()
                
                # Train robust model
                optimizer_robust.zero_grad()
                outputs = robust_model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer_robust.step()
    
    # Evaluate on all corruptions
    improvements = []
    for corruption in CORRUPTIONS:
        data, labels = load_cifar10c_corruption(corruption, severity=3)
        test_data = data[1000:1200]  # Small test set
        test_labels = labels[1000:1200]
        
        std_acc = evaluate_model(std_model, test_data, test_labels, device=device)
        robust_acc = evaluate_model(robust_model, test_data, test_labels, device=device)
        improvements.append(robust_acc - std_acc)
    
    avg_improvement = np.mean(improvements)
    return avg_improvement, improvements

def main():
    parser = argparse.ArgumentParser(description='Statistical Significance Test with Multiple Seeds')
    parser.add_argument('--lambda_init', type=float, default=0.01,
                       help='Lambda value for R-LayerNorm')
    parser.add_argument('--seeds', type=int, nargs='+', 
                       default=[42, 123, 456, 789, 999],
                       help='Random seeds to test')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Statistical Significance Test (λ = {args.lambda_init})")
    print("="*60)
    
    all_improvements = []
    seed_results = {}
    
    for i, seed in enumerate(args.seeds):
        print(f"\nSeed {i+1}/{len(args.seeds)}: {seed}")
        avg_improvement, per_corruption = run_experiment_for_seed(seed, args.lambda_init, device)
        
        seed_results[seed] = {
            'avg_improvement': avg_improvement,
            'per_corruption': per_corruption
        }
        all_improvements.append(avg_improvement)
        
        print(f"  Average improvement: {avg_improvement:+.2f}%")
    
    # Statistical analysis
    print("\n" + "="*60)
    print("Statistical Analysis")
    print("="*60)
    
    improvements_array = np.array(all_improvements)
    mean_improvement = np.mean(improvements_array)
    std_improvement = np.std(improvements_array)
    
    # Confidence interval (95%)
    n = len(improvements_array)
    ci_lower = mean_improvement - 1.96 * std_improvement / np.sqrt(n)
    ci_upper = mean_improvement + 1.96 * std_improvement / np.sqrt(n)
    
    # t-test (one-sample, test if mean > 0)
    t_stat, p_value = stats.ttest_1samp(improvements_array, 0)
    
    print(f"Number of seeds: {n}")
    print(f"Mean improvement: {mean_improvement:+.2f}%")
    print(f"Standard deviation: {std_improvement:.2f}%")
    print(f"95% Confidence Interval: [{ci_lower:+.2f}%, {ci_upper:+.2f}%]")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Statistically significant (p < 0.05): {'YES' if p_value < 0.05 else 'NO'}")
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Improvement across seeds
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n+1), all_improvements, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Seed Number', fontsize=12)
    plt.ylabel('Average Improvement (%)', fontsize=12)
    plt.title(f'Improvement Across {n} Random Seeds', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Distribution
    plt.subplot(1, 2, 2)
    plt.boxplot(all_improvements, vert=True, patch_artist=True)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xticks([1], ['All Seeds'])
    plt.ylabel('Improvement (%)', fontsize=12)
    plt.title(f'Distribution of Improvements\nMean: {mean_improvement:+.2f}%', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f'{args.output_dir}/seed_reproducibility.png'
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    
    # Save results
    with open(f'{args.output_dir}/seed_results.csv', 'w') as f:
        f.write("seed,avg_improvement," + ",".join(CORRUPTIONS) + "\n")
        for seed, results in seed_results.items():
            row = [str(seed), f"{results['avg_improvement']:.4f}"]
            row.extend([f"{imp:.4f}" for imp in results['per_corruption']])
            f.write(",".join(row) + "\n")
    
    print(f"Results saved to {args.output_dir}/seed_results.csv")

if __name__ == '__main__':
    main()
