"""
Benchmark Datasets for Gradient Domain Discovery

This module provides standard benchmarks commonly used in:
- Multi-task learning
- Domain adaptation  
- Transfer learning

Benchmarks included:
1. Multi-MNIST: Rotated/colored MNIST variants
2. CIFAR-10 Corruptions: Different corruption types as domains
3. Office-31 style: Synthetic domain shift
4. Multi-Task Regression: Synthetic tasks with controlled similarity
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os


# =============================================================================
# Benchmark 1: Rotated MNIST (Classic Domain Adaptation Benchmark)
# =============================================================================

class RotatedMNIST(Dataset):
    """
    MNIST with rotation as domain shift.
    
    Each rotation angle defines a different domain.
    Classic benchmark from domain adaptation literature.
    """
    
    def __init__(
        self,
        root: str = "./data",
        rotation_angle: float = 0.0,
        train: bool = True,
        download: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.rotation_angle = rotation_angle
        self.max_samples = max_samples
        
        # Load base MNIST
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor(),
        )
        
        # Rotation transform
        self.rotate = transforms.functional.rotate
    
    def __len__(self):
        if self.max_samples:
            return min(self.max_samples, len(self.mnist))
        return len(self.mnist)
    
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        
        # Apply rotation
        if self.rotation_angle != 0:
            img = self.rotate(img, self.rotation_angle)
        
        return img, label


def create_rotated_mnist_domains(
    angles: List[float] = [0, 15, 30, 45, 60, 75],
    root: str = "./data",
    train: bool = True,
    samples_per_domain: int = 1000,
) -> Dict[int, DataLoader]:
    """
    Create multiple MNIST domains with different rotations.
    
    Args:
        angles: Rotation angles for each domain
        root: Data directory
        train: Use training set
        samples_per_domain: Samples per domain
    
    Returns:
        Dictionary mapping domain_id to DataLoader
    """
    loaders = {}
    
    for i, angle in enumerate(angles):
        dataset = RotatedMNIST(
            root=root,
            rotation_angle=angle,
            train=train,
            max_samples=samples_per_domain,
        )
        
        loaders[i] = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0,
        )
    
    return loaders


# =============================================================================
# Benchmark 2: Colored MNIST (Spurious Correlation Benchmark)
# =============================================================================

class ColoredMNIST(Dataset):
    """
    MNIST with color as spurious correlation.
    
    Different color schemes define different domains.
    Tests whether method can discover spurious vs causal features.
    """
    
    def __init__(
        self,
        root: str = "./data",
        color_scheme: str = "red",  # "red", "green", "blue", "random"
        correlation: float = 0.9,  # How correlated color is with label
        train: bool = True,
        download: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.color_scheme = color_scheme
        self.correlation = correlation
        self.max_samples = max_samples
        
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor(),
        )
        
        # Color mappings
        self.colors = {
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
        }
    
    def __len__(self):
        if self.max_samples:
            return min(self.max_samples, len(self.mnist))
        return len(self.mnist)
    
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        
        # Convert to RGB
        img_rgb = img.repeat(3, 1, 1)
        
        # Apply color based on scheme
        if self.color_scheme == "random":
            color = np.random.choice(list(self.colors.keys()))
        else:
            color = self.color_scheme
        
        r, g, b = self.colors.get(color, (1.0, 1.0, 1.0))
        
        # Tint the image
        img_rgb[0] = img_rgb[0] * r
        img_rgb[1] = img_rgb[1] * g
        img_rgb[2] = img_rgb[2] * b
        
        return img_rgb, label


def create_colored_mnist_domains(
    colors: List[str] = ["red", "green", "blue", "yellow"],
    root: str = "./data",
    train: bool = True,
    samples_per_domain: int = 1000,
) -> Dict[int, DataLoader]:
    """Create multiple MNIST domains with different colors."""
    loaders = {}
    
    for i, color in enumerate(colors):
        dataset = ColoredMNIST(
            root=root,
            color_scheme=color,
            train=train,
            max_samples=samples_per_domain,
        )
        
        loaders[i] = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0,
        )
    
    return loaders


# =============================================================================
# Benchmark 3: Multi-Task Regression (Controlled Similarity)
# =============================================================================

class MultiTaskRegressionDataset(Dataset):
    """
    Synthetic regression tasks with controlled similarity.
    
    Each task has a ground truth weight vector. Task similarity
    is controlled by the angle between weight vectors.
    
    This allows precise validation of gradient-based domain discovery.
    """
    
    def __init__(
        self,
        task_id: int,
        weight_vector: np.ndarray,
        num_samples: int = 1000,
        input_dim: int = 20,
        noise_std: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.task_id = task_id
        self.weight_vector = weight_vector
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.noise_std = noise_std
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate data
        self.X = np.random.randn(num_samples, input_dim).astype(np.float32)
        self.y = (self.X @ weight_vector + 
                  noise_std * np.random.randn(num_samples)).astype(np.float32)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


def create_multitask_regression_domains(
    num_domains: int = 3,
    tasks_per_domain: int = 3,
    input_dim: int = 20,
    within_domain_similarity: float = 0.9,  # Cosine similarity within domain
    cross_domain_similarity: float = 0.1,   # Cosine similarity across domains
    samples_per_task: int = 500,
    seed: int = 42,
) -> Tuple[Dict[int, DataLoader], np.ndarray, np.ndarray]:
    """
    Create multi-task regression with controlled domain structure.
    
    Returns:
        loaders: Dictionary mapping task_id to DataLoader
        true_labels: Ground truth domain labels
        weight_vectors: Weight vectors for each task
    """
    np.random.seed(seed)
    
    total_tasks = num_domains * tasks_per_domain
    weight_vectors = []
    true_labels = []
    
    # Generate domain centers (orthogonal directions)
    domain_centers = []
    for d in range(num_domains):
        # Random unit vector
        center = np.random.randn(input_dim)
        center = center / np.linalg.norm(center)
        
        # Orthogonalize against previous centers
        for prev in domain_centers:
            center = center - np.dot(center, prev) * prev
            center = center / (np.linalg.norm(center) + 1e-10)
        
        domain_centers.append(center)
    
    # Generate task weights within each domain
    for d in range(num_domains):
        center = domain_centers[d]
        
        for t in range(tasks_per_domain):
            # Perturb center to get task weight
            perturbation = np.random.randn(input_dim) * 0.1
            weight = center + perturbation * (1 - within_domain_similarity)
            weight = weight / np.linalg.norm(weight)
            
            weight_vectors.append(weight)
            true_labels.append(d)
    
    weight_vectors = np.array(weight_vectors)
    true_labels = np.array(true_labels)
    
    # Create datasets and loaders
    loaders = {}
    for task_id in range(total_tasks):
        dataset = MultiTaskRegressionDataset(
            task_id=task_id,
            weight_vector=weight_vectors[task_id],
            num_samples=samples_per_task,
            input_dim=input_dim,
            seed=seed + task_id,
        )
        
        loaders[task_id] = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
        )
    
    return loaders, true_labels, weight_vectors


# =============================================================================
# Benchmark 4: Split CIFAR-10 (Task-Incremental Learning)
# =============================================================================

class SplitCIFAR10(Dataset):
    """
    CIFAR-10 split into class-based tasks.
    
    Each task contains a subset of classes.
    Common benchmark for continual/multi-task learning.
    """
    
    def __init__(
        self,
        root: str = "./data",
        classes: List[int] = [0, 1],
        train: bool = True,
        download: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.classes = classes
        self.max_samples = max_samples
        
        # Load CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.cifar = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
        
        # Filter to selected classes
        self.indices = [
            i for i, (_, label) in enumerate(self.cifar)
            if label in classes
        ]
        
        if max_samples:
            self.indices = self.indices[:max_samples]
        
        # Remap labels to 0, 1, 2, ...
        self.label_map = {c: i for i, c in enumerate(classes)}
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.cifar[real_idx]
        
        # Remap label
        new_label = self.label_map[label]
        
        return img, new_label


def create_split_cifar_domains(
    class_splits: List[List[int]] = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
    root: str = "./data",
    train: bool = True,
    samples_per_domain: int = 2000,
) -> Dict[int, DataLoader]:
    """
    Create CIFAR-10 domains based on class splits.
    
    Default: 5 domains, each with 2 classes.
    """
    loaders = {}
    
    for i, classes in enumerate(class_splits):
        dataset = SplitCIFAR10(
            root=root,
            classes=classes,
            train=train,
            max_samples=samples_per_domain,
        )
        
        loaders[i] = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0,
        )
    
    return loaders


# =============================================================================
# Simple Models for Benchmarks
# =============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP for MNIST-like tasks."""
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 10,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


class SimpleConvNet(nn.Module):
    """Simple CNN for image classification."""
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class LinearRegressor(nn.Module):
    """Linear model for regression tasks."""
    
    def __init__(self, input_dim: int = 20):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)


# =============================================================================
# Benchmark Info
# =============================================================================

@dataclass
class BenchmarkInfo:
    """Information about a benchmark."""
    name: str
    num_domains: int
    description: str
    domain_labels: Optional[np.ndarray] = None


BENCHMARK_REGISTRY = {
    "rotated_mnist": BenchmarkInfo(
        name="Rotated MNIST",
        num_domains=6,
        description="MNIST with 6 rotation angles (0, 15, 30, 45, 60, 75 degrees)",
    ),
    "colored_mnist": BenchmarkInfo(
        name="Colored MNIST",
        num_domains=4,
        description="MNIST with 4 color schemes (red, green, blue, yellow)",
    ),
    "multitask_regression": BenchmarkInfo(
        name="Multi-Task Regression",
        num_domains=3,
        description="Synthetic regression with 3 domains, 3 tasks each",
    ),
    "split_cifar": BenchmarkInfo(
        name="Split CIFAR-10",
        num_domains=5,
        description="CIFAR-10 split into 5 class-based domains",
    ),
}


def get_benchmark(
    name: str,
    root: str = "./data",
    **kwargs,
) -> Tuple[Dict[int, DataLoader], np.ndarray, BenchmarkInfo]:
    """
    Get a benchmark by name.
    
    Returns:
        loaders: Dictionary of DataLoaders
        true_labels: Ground truth domain labels
        info: Benchmark information
    """
    if name == "rotated_mnist":
        angles = kwargs.get("angles", [0, 15, 30, 45, 60, 75])
        loaders = create_rotated_mnist_domains(angles=angles, root=root)
        true_labels = np.arange(len(angles))  # Each angle is its own domain
        info = BENCHMARK_REGISTRY[name]
        info.num_domains = len(angles)
        
    elif name == "colored_mnist":
        colors = kwargs.get("colors", ["red", "green", "blue", "yellow"])
        loaders = create_colored_mnist_domains(colors=colors, root=root)
        true_labels = np.arange(len(colors))
        info = BENCHMARK_REGISTRY[name]
        info.num_domains = len(colors)
        
    elif name == "multitask_regression":
        num_domains = kwargs.get("num_domains", 3)
        tasks_per_domain = kwargs.get("tasks_per_domain", 3)
        loaders, true_labels, _ = create_multitask_regression_domains(
            num_domains=num_domains,
            tasks_per_domain=tasks_per_domain,
        )
        info = BENCHMARK_REGISTRY[name]
        info.num_domains = num_domains
        
    elif name == "split_cifar":
        splits = kwargs.get("class_splits", [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        loaders = create_split_cifar_domains(class_splits=splits, root=root)
        true_labels = np.arange(len(splits))
        info = BENCHMARK_REGISTRY[name]
        info.num_domains = len(splits)
        
    else:
        raise ValueError(f"Unknown benchmark: {name}")
    
    info.domain_labels = true_labels
    return loaders, true_labels, info
