"""
Utility functions for R-LayerNorm experiments
"""

import torch
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List

def load_cifar10c_corruption(
    corruption_type: str = 'gaussian_noise',
    severity: int = 3,
    data_dir: str = './data/CIFAR-10-C'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CIFAR-10-C corruption data
    
    Args:
        corruption_type: Type of corruption (e.g., 'gaussian_noise')
        severity: Severity level (1-5)
        data_dir: Directory containing CIFAR-10-C data
    
    Returns:
        images, labels: Tuple of numpy arrays
    """
    data_path = os.path.join(data_dir, f'{corruption_type}.npy')
    labels_path = os.path.join(data_dir, 'labels.npy')
    
    # Load data (50000 images total, 5 severities × 10000 each)
    all_data = np.load(data_path)  # Shape: (50000, 32, 32, 3)
    labels = np.load(labels_path)  # Shape: (50000,)
    
    # Split by severity
    start_idx = (severity - 1) * 10000
    end_idx = severity * 10000
    
    images = all_data[start_idx:end_idx]
    targets = labels[start_idx:end_idx]
    
    return images, targets


def create_dataloader(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    device: torch.device = None
) -> DataLoader:
    """
    Create PyTorch DataLoader from numpy arrays
    
    Args:
        data: Image data in (N, H, W, C) format
        labels: Corresponding labels
        batch_size: Batch size
        shuffle: Whether to shuffle data
        device: Target device for tensors
    
    Returns:
        DataLoader for the data
    """
    # Convert to tensor and normalize
    data_tensor = torch.tensor(data).float().permute(0, 3, 1, 2) / 255.0
    labels_tensor = torch.tensor(labels)
    
    if device:
        data_tensor = data_tensor.to(device)
        labels_tensor = labels_tensor.to(device)
    
    dataset = TensorDataset(data_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train model for one epoch
    
    Returns:
        average_loss, accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_data, batch_labels in dataloader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate_model(
    model: torch.nn.Module,
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    device: torch.device = None
) -> float:
    """
    Evaluate model on test data
    
    Returns:
        Accuracy in percentage
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    dataloader = create_dataloader(data, labels, batch_size, shuffle=False, device=device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    return 100 * correct / total


def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
