"""
Gradient Domain Discovery - Benchmark Experiments

This script runs the GDD algorithm on standard benchmarks and validates
that gradient-based domain discovery works as predicted by the theory.

Experiments:
1. Domain Recovery: Can we recover true domains from gradients?
2. Transfer Prediction: Does gradient conflict predict negative transfer?
3. Temporal Dynamics: How does domain structure evolve during training?
4. Model Size Effect: Does capacity affect domain separation?

Usage:
    python run_experiments.py --benchmark rotated_mnist --experiment all
    python run_experiments.py --benchmark multitask_regression --experiment recovery
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from gdd_core import (
    GradientCollector,
    GradientDomainDiscovery,
    DomainDiscoveryResult,
    validate_domain_discovery,
    compute_transfer_gap,
)
from benchmarks import (
    get_benchmark,
    SimpleMLP,
    SimpleConvNet,
    LinearRegressor,
    BenchmarkInfo,
)


# =============================================================================
# Experiment 1: Domain Recovery
# =============================================================================

def experiment_domain_recovery(
    benchmark_name: str,
    output_dir: Path,
    device: torch.device,
    num_epochs: int = 5,
    samples_per_source: int = 50,
    **kwargs,
) -> Dict:
    """
    Test whether GDD can recover true domain structure.
    
    Protocol:
    1. Train model on all sources jointly
    2. Collect gradients per source
    3. Run GDD clustering
    4. Compare to ground truth
    """
    print(f"\n{'='*60}")
    print(f"Experiment 1: Domain Recovery on {benchmark_name}")
    print(f"{'='*60}\n")
    
    # Load benchmark
    loaders, true_labels, info = get_benchmark(benchmark_name, **kwargs)
    num_sources = len(loaders)
    
    print(f"Benchmark: {info.name}")
    print(f"Sources: {num_sources}")
    print(f"True domains: {info.num_domains}")
    print(f"True labels: {true_labels}")
    
    # Create model
    if benchmark_name == "multitask_regression":
        model = LinearRegressor(input_dim=20).to(device)
        loss_fn = nn.MSELoss()
    elif benchmark_name in ["rotated_mnist"]:
        model = SimpleMLP(input_dim=784, output_dim=10).to(device)
        loss_fn = nn.CrossEntropyLoss()
    elif benchmark_name in ["colored_mnist"]:
        model = SimpleMLP(input_dim=784*3, output_dim=10).to(device)
        loss_fn = nn.CrossEntropyLoss()
    else:
        model = SimpleConvNet(in_channels=3, num_classes=10).to(device)
        loss_fn = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Gradient collector
    collector = GradientCollector(
        num_sources=num_sources,
        max_samples_per_source=samples_per_source,
        use_random_projection=True,
        projection_dim=512,
    )
    
    # Training loop with gradient collection
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {i: [] for i in range(num_sources)}
        
        # Iterate through all sources
        for source_id, loader in loaders.items():
            for batch_idx, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                
                # Collect gradient
                step = epoch * 1000 + source_id * 100 + batch_idx
                collector.collect(source_id, model, step, loss.item())
                
                optimizer.step()
                epoch_losses[source_id].append(loss.item())
        
        # Print progress
        mean_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        print(f"  Epoch {epoch+1}/{num_epochs} - Losses: {mean_losses}")
    
    # Run domain discovery
    print(f"\nRunning GDD with {collector.total_samples()} gradient samples...")
    
    gdd = GradientDomainDiscovery(
        similarity_metric="cosine",
        clustering_method="spectral",
        num_domains=None,  # Auto-select
    )
    
    result = gdd.discover_domains(collector, true_labels)
    
    print(f"\n{result.summary()}")
    
    # Validate against ground truth
    validation = validate_domain_discovery(result, true_labels)
    
    print(f"\nValidation Metrics:")
    for key, value in validation.items():
        print(f"  {key}: {value:.4f}")
    
    # Visualize similarity matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Similarity matrix
    sns.heatmap(
        result.similarity_matrix,
        ax=axes[0],
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
    )
    axes[0].set_title("Gradient Cosine Similarity")
    axes[0].set_xlabel("Source")
    axes[0].set_ylabel("Source")
    
    # Conflict matrix
    sns.heatmap(
        result.conflict_matrix,
        ax=axes[1],
        cmap="Reds",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
    )
    axes[1].set_title("Gradient Conflict Rate")
    axes[1].set_xlabel("Source")
    axes[1].set_ylabel("Source")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{benchmark_name}_similarity_matrix.png", dpi=150)
    plt.close()
    
    # Save results
    results = {
        "benchmark": benchmark_name,
        "num_sources": num_sources,
        "true_domains": info.num_domains,
        "discovered_domains": result.num_domains,
        "true_labels": true_labels.tolist(),
        "discovered_labels": result.domain_assignments.tolist(),
        "validation": validation,
        "similarity_matrix": result.similarity_matrix.tolist(),
        "conflict_matrix": result.conflict_matrix.tolist(),
    }
    
    with open(output_dir / f"{benchmark_name}_recovery_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# Experiment 2: Transfer Prediction
# =============================================================================

def experiment_transfer_prediction(
    benchmark_name: str,
    output_dir: Path,
    device: torch.device,
    num_epochs: int = 10,
    **kwargs,
) -> Dict:
    """
    Test whether gradient conflict predicts negative transfer.
    
    Protocol:
    1. Train separate models on each source
    2. Evaluate each model on all sources
    3. Compute transfer gaps
    4. Correlate with gradient similarity
    """
    print(f"\n{'='*60}")
    print(f"Experiment 2: Transfer Prediction on {benchmark_name}")
    print(f"{'='*60}\n")
    
    # Load benchmark
    loaders, true_labels, info = get_benchmark(benchmark_name, **kwargs)
    num_sources = len(loaders)
    
    # Train separate model for each source
    source_models = {}
    
    for source_id in tqdm(range(num_sources), desc="Training source models"):
        if benchmark_name == "multitask_regression":
            model = LinearRegressor(input_dim=20).to(device)
            loss_fn = nn.MSELoss()
        elif benchmark_name in ["rotated_mnist"]:
            model = SimpleMLP(input_dim=784, output_dim=10).to(device)
            loss_fn = nn.CrossEntropyLoss()
        elif benchmark_name in ["colored_mnist"]:
            model = SimpleMLP(input_dim=784*3, output_dim=10).to(device)
            loss_fn = nn.CrossEntropyLoss()
        else:
            model = SimpleConvNet(in_channels=3, num_classes=10).to(device)
            loss_fn = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Train on this source only
        for epoch in range(num_epochs):
            model.train()
            for x, y in loaders[source_id]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
        
        source_models[source_id] = model
    
    # Compute transfer matrix
    print("\nComputing transfer matrix...")
    
    transfer_matrix = np.zeros((num_sources, num_sources))
    
    for i in range(num_sources):
        model = source_models[i]
        model.eval()
        
        for j in range(num_sources):
            # Evaluate model i on source j
            total_loss = 0
            n_samples = 0
            
            with torch.no_grad():
                for x, y in loaders[j]:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    
                    if benchmark_name == "multitask_regression":
                        loss = nn.MSELoss()(pred, y)
                    else:
                        loss = nn.CrossEntropyLoss()(pred, y)
                    
                    total_loss += loss.item() * len(x)
                    n_samples += len(x)
            
            transfer_matrix[i, j] = total_loss / n_samples
    
    # Compute transfer gaps (off-diagonal - diagonal)
    transfer_gaps = np.zeros((num_sources, num_sources))
    for i in range(num_sources):
        for j in range(num_sources):
            # Gap = loss on j using model trained on i, minus loss on j using model trained on j
            transfer_gaps[i, j] = transfer_matrix[i, j] - transfer_matrix[j, j]
    
    # Now collect gradients and compute similarity
    print("\nCollecting gradients for similarity computation...")
    
    # Use a fresh model for gradient collection
    if benchmark_name == "multitask_regression":
        model = LinearRegressor(input_dim=20).to(device)
        loss_fn = nn.MSELoss()
    elif benchmark_name in ["rotated_mnist"]:
        model = SimpleMLP(input_dim=784, output_dim=10).to(device)
        loss_fn = nn.CrossEntropyLoss()
    elif benchmark_name in ["colored_mnist"]:
        model = SimpleMLP(input_dim=784*3, output_dim=10).to(device)
        loss_fn = nn.CrossEntropyLoss()
    else:
        model = SimpleConvNet(in_channels=3, num_classes=10).to(device)
        loss_fn = nn.CrossEntropyLoss()
    
    collector = GradientCollector(
        num_sources=num_sources,
        max_samples_per_source=30,
        use_random_projection=True,
        projection_dim=256,
    )
    
    # Collect gradients
    for source_id, loader in loaders.items():
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= 30:
                break
            
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            collector.collect(source_id, model, batch_idx, loss.item())
    
    # Compute similarity
    gdd = GradientDomainDiscovery()
    similarity, conflict = gdd.compute_similarity_matrix(collector)
    
    # Compute correlation between gradient similarity and transfer gap
    # (excluding diagonal)
    sim_flat = []
    gap_flat = []
    
    for i in range(num_sources):
        for j in range(num_sources):
            if i != j:
                sim_flat.append(similarity[i, j])
                gap_flat.append(transfer_gaps[i, j])
    
    correlation = np.corrcoef(sim_flat, gap_flat)[0, 1]
    
    print(f"\nCorrelation between gradient similarity and transfer gap: {correlation:.4f}")
    print("(Negative correlation expected: high similarity -> low gap)")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Transfer matrix
    sns.heatmap(
        transfer_matrix,
        ax=axes[0],
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        square=True,
    )
    axes[0].set_title("Transfer Matrix (Loss)")
    axes[0].set_xlabel("Target Source")
    axes[0].set_ylabel("Trained on Source")
    
    # Transfer gaps
    sns.heatmap(
        transfer_gaps,
        ax=axes[1],
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".3f",
        square=True,
    )
    axes[1].set_title("Transfer Gaps")
    axes[1].set_xlabel("Target Source")
    axes[1].set_ylabel("Trained on Source")
    
    # Scatter plot
    axes[2].scatter(sim_flat, gap_flat, alpha=0.7)
    axes[2].set_xlabel("Gradient Similarity")
    axes[2].set_ylabel("Transfer Gap")
    axes[2].set_title(f"Correlation: {correlation:.3f}")
    
    # Add trend line
    z = np.polyfit(sim_flat, gap_flat, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(sim_flat), max(sim_flat), 100)
    axes[2].plot(x_line, p(x_line), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{benchmark_name}_transfer_prediction.png", dpi=150)
    plt.close()
    
    results = {
        "benchmark": benchmark_name,
        "correlation": correlation,
        "transfer_matrix": transfer_matrix.tolist(),
        "transfer_gaps": transfer_gaps.tolist(),
        "similarity_matrix": similarity.tolist(),
    }
    
    with open(output_dir / f"{benchmark_name}_transfer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# Experiment 3: Temporal Dynamics
# =============================================================================

def experiment_temporal_dynamics(
    benchmark_name: str,
    output_dir: Path,
    device: torch.device,
    num_epochs: int = 20,
    checkpoints: int = 10,
    **kwargs,
) -> Dict:
    """
    Track how domain structure evolves during training.
    
    Protocol:
    1. Train model on all sources
    2. At regular intervals, compute similarity matrix
    3. Track how similarities change over time
    """
    print(f"\n{'='*60}")
    print(f"Experiment 3: Temporal Dynamics on {benchmark_name}")
    print(f"{'='*60}\n")
    
    # Load benchmark
    loaders, true_labels, info = get_benchmark(benchmark_name, **kwargs)
    num_sources = len(loaders)
    
    # Create model
    if benchmark_name == "multitask_regression":
        model = LinearRegressor(input_dim=20).to(device)
        loss_fn = nn.MSELoss()
    elif benchmark_name in ["rotated_mnist"]:
        model = SimpleMLP(input_dim=784, output_dim=10).to(device)
        loss_fn = nn.CrossEntropyLoss()
    elif benchmark_name in ["colored_mnist"]:
        model = SimpleMLP(input_dim=784*3, output_dim=10).to(device)
        loss_fn = nn.CrossEntropyLoss()
    else:
        model = SimpleConvNet(in_channels=3, num_classes=10).to(device)
        loss_fn = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Track similarities over time
    checkpoint_epochs = np.linspace(0, num_epochs - 1, checkpoints, dtype=int)
    similarity_history = []
    conflict_history = []
    
    gdd = GradientDomainDiscovery()
    
    print(f"Training for {num_epochs} epochs, checkpoints at: {checkpoint_epochs.tolist()}")
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        
        # Training step
        for source_id, loader in loaders.items():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
        
        # Checkpoint: collect gradients and compute similarity
        if epoch in checkpoint_epochs:
            collector = GradientCollector(
                num_sources=num_sources,
                max_samples_per_source=20,
                use_random_projection=True,
                projection_dim=256,
            )
            
            for source_id, loader in loaders.items():
                for batch_idx, (x, y) in enumerate(loader):
                    if batch_idx >= 20:
                        break
                    
                    x, y = x.to(device), y.to(device)
                    model.zero_grad()
                    loss = loss_fn(model(x), y)
                    loss.backward()
                    collector.collect(source_id, model, batch_idx, loss.item())
            
            similarity, conflict = gdd.compute_similarity_matrix(collector)
            similarity_history.append(similarity.copy())
            conflict_history.append(conflict.copy())
    
    # Analyze temporal dynamics
    print("\nAnalyzing temporal dynamics...")
    
    # Compute within-domain and cross-domain similarity over time
    within_sim = []
    cross_sim = []
    
    for sim in similarity_history:
        within = []
        cross = []
        
        for i in range(num_sources):
            for j in range(i + 1, num_sources):
                if true_labels[i] == true_labels[j]:
                    within.append(sim[i, j])
                else:
                    cross.append(sim[i, j])
        
        within_sim.append(np.mean(within) if within else 0)
        cross_sim.append(np.mean(cross) if cross else 0)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Similarity over time
    axes[0].plot(checkpoint_epochs, within_sim, 'b-o', label='Within-domain')
    axes[0].plot(checkpoint_epochs, cross_sim, 'r-o', label='Cross-domain')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Mean Gradient Similarity')
    axes[0].set_title('Domain Separation Over Training')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gap over time
    gap = [w - c for w, c in zip(within_sim, cross_sim)]
    axes[1].plot(checkpoint_epochs, gap, 'g-o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Separability Gap')
    axes[1].set_title('Within - Cross Domain Similarity')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{benchmark_name}_temporal_dynamics.png", dpi=150)
    plt.close()
    
    # Visualize similarity matrices at different times
    n_show = min(4, len(similarity_history))
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
    
    if n_show == 1:
        axes = [axes]
    
    indices = np.linspace(0, len(similarity_history) - 1, n_show, dtype=int)
    
    for ax, idx in zip(axes, indices):
        sns.heatmap(
            similarity_history[idx],
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar=False,
        )
        ax.set_title(f"Epoch {checkpoint_epochs[idx]}")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{benchmark_name}_similarity_evolution.png", dpi=150)
    plt.close()
    
    results = {
        "benchmark": benchmark_name,
        "checkpoint_epochs": checkpoint_epochs.tolist(),
        "within_domain_similarity": within_sim,
        "cross_domain_similarity": cross_sim,
        "separability_gap": gap,
    }
    
    with open(output_dir / f"{benchmark_name}_temporal_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# Experiment 4: Model Size Effect
# =============================================================================

def experiment_model_size(
    benchmark_name: str,
    output_dir: Path,
    device: torch.device,
    num_epochs: int = 10,
    **kwargs,
) -> Dict:
    """
    Test how model capacity affects domain separation.
    
    Protocol:
    1. Train models of different sizes
    2. Compare gradient conflict across sizes
    """
    print(f"\n{'='*60}")
    print(f"Experiment 4: Model Size Effect on {benchmark_name}")
    print(f"{'='*60}\n")
    
    if benchmark_name != "multitask_regression":
        print("This experiment is designed for multitask_regression benchmark.")
        print("Skipping...")
        return {}
    
    # Load benchmark
    loaders, true_labels, info = get_benchmark(benchmark_name, **kwargs)
    num_sources = len(loaders)
    
    # Different model sizes
    hidden_dims_list = [
        [],           # Linear (smallest)
        [16],         # Small
        [64],         # Medium
        [128, 64],    # Large
        [256, 128, 64],  # Very large
    ]
    
    size_names = ["Linear", "Small", "Medium", "Large", "Very Large"]
    
    results_by_size = {}
    
    for hidden_dims, size_name in zip(hidden_dims_list, size_names):
        print(f"\nTesting {size_name} model (hidden: {hidden_dims})...")
        
        # Create model
        if hidden_dims:
            layers = []
            prev_dim = 20
            for h in hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.ReLU())
                prev_dim = h
            layers.append(nn.Linear(prev_dim, 1))
            model = nn.Sequential(*layers).to(device)
        else:
            model = LinearRegressor(input_dim=20).to(device)
        
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params}")
        
        # Train
        for epoch in range(num_epochs):
            model.train()
            for source_id, loader in loaders.items():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = loss_fn(model(x), y)
                    loss.backward()
                    optimizer.step()
        
        # Collect gradients
        collector = GradientCollector(
            num_sources=num_sources,
            max_samples_per_source=30,
            use_random_projection=True,
            projection_dim=min(256, num_params),
        )
        
        for source_id, loader in loaders.items():
            for batch_idx, (x, y) in enumerate(loader):
                if batch_idx >= 30:
                    break
                
                x, y = x.to(device), y.to(device)
                model.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                collector.collect(source_id, model, batch_idx, loss.item())
        
        # Compute similarity
        gdd = GradientDomainDiscovery()
        similarity, conflict = gdd.compute_similarity_matrix(collector)
        
        # Compute metrics
        within_sim = []
        cross_sim = []
        within_conf = []
        cross_conf = []
        
        for i in range(num_sources):
            for j in range(i + 1, num_sources):
                if true_labels[i] == true_labels[j]:
                    within_sim.append(similarity[i, j])
                    within_conf.append(conflict[i, j])
                else:
                    cross_sim.append(similarity[i, j])
                    cross_conf.append(conflict[i, j])
        
        results_by_size[size_name] = {
            "num_params": num_params,
            "hidden_dims": hidden_dims,
            "within_similarity": np.mean(within_sim),
            "cross_similarity": np.mean(cross_sim),
            "within_conflict": np.mean(within_conf),
            "cross_conflict": np.mean(cross_conf),
            "separability_gap": np.mean(within_sim) - np.mean(cross_sim),
        }
        
        print(f"  Within-domain similarity: {np.mean(within_sim):.4f}")
        print(f"  Cross-domain similarity: {np.mean(cross_sim):.4f}")
        print(f"  Separability gap: {np.mean(within_sim) - np.mean(cross_sim):.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sizes = [r["num_params"] for r in results_by_size.values()]
    within = [r["within_similarity"] for r in results_by_size.values()]
    cross = [r["cross_similarity"] for r in results_by_size.values()]
    gaps = [r["separability_gap"] for r in results_by_size.values()]
    
    # Similarity by size
    axes[0].plot(sizes, within, 'b-o', label='Within-domain')
    axes[0].plot(sizes, cross, 'r-o', label='Cross-domain')
    axes[0].set_xlabel('Number of Parameters')
    axes[0].set_ylabel('Gradient Similarity')
    axes[0].set_title('Domain Separation vs Model Size')
    axes[0].legend()
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Gap by size
    axes[1].plot(sizes, gaps, 'g-o')
    axes[1].set_xlabel('Number of Parameters')
    axes[1].set_ylabel('Separability Gap')
    axes[1].set_title('Domain Separability vs Model Size')
    axes[1].set_xscale('log')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{benchmark_name}_model_size.png", dpi=150)
    plt.close()
    
    with open(output_dir / f"{benchmark_name}_model_size_results.json", "w") as f:
        json.dump(results_by_size, f, indent=2)
    
    return results_by_size


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GDD Benchmark Experiments")
    
    parser.add_argument(
        "--benchmark",
        type=str,
        default="multitask_regression",
        choices=["rotated_mnist", "colored_mnist", "multitask_regression", "split_cifar"],
        help="Benchmark to run",
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["recovery", "transfer", "temporal", "model_size", "all"],
        help="Experiment to run",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/gdd_experiments",
        help="Output directory",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    all_results = {}
    
    if args.experiment in ["recovery", "all"]:
        all_results["recovery"] = experiment_domain_recovery(
            args.benchmark, output_dir, device
        )
    
    if args.experiment in ["transfer", "all"]:
        all_results["transfer"] = experiment_transfer_prediction(
            args.benchmark, output_dir, device
        )
    
    if args.experiment in ["temporal", "all"]:
        all_results["temporal"] = experiment_temporal_dynamics(
            args.benchmark, output_dir, device
        )
    
    if args.experiment in ["model_size", "all"]:
        all_results["model_size"] = experiment_model_size(
            args.benchmark, output_dir, device
        )
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Output: {output_dir}")
    print(f"Results saved to JSON files.")
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"all_results_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
