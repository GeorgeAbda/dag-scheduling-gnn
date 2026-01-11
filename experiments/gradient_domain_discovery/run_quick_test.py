"""
Quick test of GDD without matplotlib (for environments with numpy conflicts).
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from gdd_core import (
    GradientCollector,
    GradientDomainDiscovery,
    validate_domain_discovery,
)
from benchmarks import (
    get_benchmark,
    LinearRegressor,
    SimpleMLP,
)


def run_domain_recovery_test(
    benchmark_name: str = "multitask_regression",
    num_epochs: int = 5,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """Test domain recovery on a benchmark."""
    
    print(f"\n{'='*60}")
    print(f"Domain Recovery Test: {benchmark_name}")
    print(f"{'='*60}\n")
    
    # Load benchmark
    if benchmark_name == "multitask_regression":
        loaders, true_labels, info = get_benchmark(
            benchmark_name, 
            num_domains=3, 
            tasks_per_domain=3,
        )
        model = LinearRegressor(input_dim=20).to(device)
        loss_fn = nn.MSELoss()
    else:
        loaders, true_labels, info = get_benchmark(benchmark_name)
        model = SimpleMLP(input_dim=784, output_dim=10).to(device)
        loss_fn = nn.CrossEntropyLoss()
    
    num_sources = len(loaders)
    
    print(f"Benchmark: {info.name}")
    print(f"Number of sources: {num_sources}")
    print(f"True domains: {info.num_domains}")
    print(f"True labels: {true_labels}")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Gradient collector
    collector = GradientCollector(
        num_sources=num_sources,
        max_samples_per_source=50,
        use_random_projection=True,
        projection_dim=256,
    )
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
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
                epoch_losses.append(loss.item())
        
        print(f"  Epoch {epoch+1}/{num_epochs} - Mean loss: {np.mean(epoch_losses):.4f}")
    
    # Run domain discovery
    print(f"\nRunning GDD with {collector.total_samples()} gradient samples...")
    
    gdd = GradientDomainDiscovery(
        similarity_metric="cosine",
        clustering_method="spectral",
        num_domains=None,  # Auto-select
    )
    
    result = gdd.discover_domains(collector, true_labels)
    
    print(f"\n{result.summary()}")
    
    # Validate
    validation = validate_domain_discovery(result, true_labels)
    
    print(f"\nValidation Metrics:")
    for key, value in validation.items():
        print(f"  {key}: {value:.4f}")
    
    # Print similarity matrix
    print(f"\nGradient Similarity Matrix:")
    print("  " + "  ".join([f"S{i}" for i in range(num_sources)]))
    for i in range(num_sources):
        row = " ".join([f"{result.similarity_matrix[i,j]:5.2f}" for j in range(num_sources)])
        print(f"S{i} {row}")
    
    # Print conflict matrix
    print(f"\nGradient Conflict Rate Matrix:")
    print("  " + "  ".join([f"S{i}" for i in range(num_sources)]))
    for i in range(num_sources):
        row = " ".join([f"{result.conflict_matrix[i,j]:5.2f}" for j in range(num_sources)])
        print(f"S{i} {row}")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    # Check if within-domain similarity > cross-domain similarity
    within_sim = validation["same_domain_similarity"]
    cross_sim = validation["diff_domain_similarity"]
    gap = within_sim - cross_sim
    
    print(f"\nWithin-domain similarity: {within_sim:.4f}")
    print(f"Cross-domain similarity:  {cross_sim:.4f}")
    print(f"Separability gap:         {gap:.4f}")
    
    if gap > 0.2:
        print("✓ Strong domain separation detected!")
    elif gap > 0.1:
        print("~ Moderate domain separation detected.")
    else:
        print("✗ Weak domain separation.")
    
    # Check conflict rates
    within_conf = validation["same_domain_conflict"]
    cross_conf = validation["diff_domain_conflict"]
    
    print(f"\nWithin-domain conflict rate: {within_conf:.4f}")
    print(f"Cross-domain conflict rate:  {cross_conf:.4f}")
    
    if cross_conf > within_conf + 0.1:
        print("✓ Cross-domain gradients conflict more than within-domain!")
    
    # Check recovery
    ari = validation["adjusted_rand_index"]
    print(f"\nAdjusted Rand Index: {ari:.4f}")
    
    if ari > 0.8:
        print("✓ Excellent domain recovery!")
    elif ari > 0.5:
        print("~ Good domain recovery.")
    elif ari > 0.2:
        print("~ Partial domain recovery.")
    else:
        print("✗ Poor domain recovery.")
    
    return {
        "benchmark": benchmark_name,
        "num_sources": num_sources,
        "true_domains": info.num_domains,
        "discovered_domains": result.num_domains,
        "true_labels": true_labels.tolist(),
        "discovered_labels": result.domain_assignments.tolist(),
        "validation": validation,
    }


def main():
    print("\n" + "="*60)
    print("GRADIENT DOMAIN DISCOVERY - QUICK TEST")
    print("="*60)
    
    # Set seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Device
    device = torch.device("cpu")
    print(f"\nUsing device: {device}")
    
    # Run test
    results = run_domain_recovery_test(
        benchmark_name="multitask_regression",
        num_epochs=5,
        device=device,
    )
    
    # Save results
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "quick_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir / 'quick_test_results.json'}")


if __name__ == "__main__":
    main()
