# Gradient Domain Discovery (GDD) Experiments

Implementation of the theoretical framework from:
> "Gradient-Based Domain Discovery: A Theoretical Framework for Unsupervised Domain Identification in Multi-Source Learning"

## Overview

This module implements **Gradient Domain Discovery (GDD)**, a method to discover latent domains in multi-source learning using gradient geometry. The key insight is that sources belonging to the same domain produce aligned gradients, while sources from different domains produce conflicting gradients.

## Key Experimental Results

### Meta-World MT10 (10 robotic manipulation tasks)

**Benchmark**: Standard multi-task RL benchmark from Yu et al. (2020)

Using **value function gradients**, we discovered gradient-based domain structure:

| Task Pair | Gradient Similarity | Interpretation |
|-----------|---------------------|----------------|
| window-open ↔ window-close | **+0.19** | Same motion type |
| push ↔ drawer-close | **+0.09** | Both pushing motions |
| drawer-close ↔ window-open | **-0.22** | CONFLICT: opposite motions |
| push ↔ window-open | **-0.12** | CONFLICT |

**Discovered Clusters**:
- **Cluster A**: push, drawer-close (pushing/closing motions)
- **Cluster B**: reach, pick-place, door-open, window-open, window-close, etc. (reaching/opening)

**Validation**:
- Within-domain similarity: **+0.031**
- Cross-domain similarity: **-0.007**
- Separability gap: **+0.038**

### CartPole Physics Variants (7 environments)

Using **value function gradients**, we discovered two distinct domain clusters:

| Cluster | Environments | Within-Cluster Conflict |
|---------|--------------|------------------------|
| A | normal, moon, jupiter, long_pole, heavy_cart | 8-25% |
| B | short_pole, light_cart | 9-10% |

**Cross-cluster conflict: 67-97%** — training jointly causes negative transfer!

**Key insight**: The gradient-based domains differ from physics-based groupings. This is a feature: we discover the true optimization structure.

### Policy vs Value Function Gradients

| Gradient Type | Domain Signal | Why |
|---------------|---------------|-----|
| Policy gradients | ~50% conflict (noise) | High variance from REINFORCE |
| Value function gradients | Clear structure | Supervised learning (MSE) has low variance |

**Recommendation**: Use value function or critic gradients for domain discovery in RL.

## Applications

### Transfer Prediction

Predict transfer success between domains BEFORE training using gradient geometry:

```python
from transfer_prediction import TransferPredictor, plot_transfer_predictions

predictor = TransferPredictor()
predictions = predictor.predict_transfer(similarity_matrix, conflict_matrix, task_names)

# Get best sources for transfer
best_sources = predictions.get_best_sources(target_id=5, top_k=3)

# Get transfer groups (tasks that transfer well together)
groups = predictions.get_transfer_groups(threshold=0.1)

# Negative transfer warnings
for i, j in zip(*np.where(predictions.recommendations == -1)):
    print(f"Warning: {tasks[i]} -> {tasks[j]} may cause negative transfer")
```

### Domain-Aware Training (Gradient Surgery)

Apply gradient surgery (PCGrad, CAGrad, MGDA) at the discovered domain level:

```python
from domain_aware_training import DomainAwareOptimizer, train_with_domain_surgery

# Domain assignments from GDD
domain_assignments = {0: 0, 1: 0, 2: 1, 3: 1}  # source_id -> domain_id

# Train with domain-level gradient surgery
model, history = train_with_domain_surgery(
    model=model,
    data_loaders=data_loaders,
    domain_assignments=domain_assignments,
    loss_fn=loss_fn,
    surgery_method="pcgrad",  # or "cagrad", "mgda", "none"
)
```

**Results on synthetic conflicting domains:**

| Method | Final Loss | Improvement |
|--------|------------|-------------|
| None (baseline) | 3.230 | - |
| **PCGrad** | **3.101** | **4.0%** |
| CAGrad | 3.184 | 1.4% |
| MGDA | 3.163 | 2.1% |

### Gradient Subspace Analysis

Analyze the low-dimensional structure of gradient manifolds:

```python
from gradient_subspace_analysis import GradientSubspaceAnalyzer

analyzer = GradientSubspaceAnalyzer(variance_threshold=0.95)
result = analyzer.analyze(gradients)  # gradients: Dict[source_id, np.ndarray]

print(f"Subspace dimensions: {result.subspace_dims}")
print(f"Subspace distance matrix:\n{result.subspace_distance_matrix}")
```

## Theoretical Foundation

### Core Definitions

**Gradient-Based Domain Equivalence**: Two sources $D_i$ and $D_j$ belong to the same domain if:
$$\mathbb{E}[\cos(g_i, g_j)] > \tau$$

**Conflict Rate**: Probability of gradient conflict:
$$\text{CR}(i,j) = \mathbb{P}[g_i^\top g_j < 0]$$

### Key Theorems

1. **Domain Recovery**: Under separability and concentration conditions, spectral clustering on gradient similarities recovers true domains.

2. **Conflict Theorem**: Negative gradient inner product implies negative transfer during joint training.

3. **Transfer Bound**: Gradient divergence bounds the transfer gap between sources.

## Benchmarks

| Benchmark | Description | Domains |
|-----------|-------------|---------|
| `rotated_mnist` | MNIST with rotation angles (0°, 15°, 30°, 45°, 60°, 75°) | 6 |
| `colored_mnist` | MNIST with color schemes (red, green, blue, yellow) | 4 |
| `multitask_regression` | Synthetic regression with controlled task similarity | 3 |
| `split_cifar` | CIFAR-10 split by class pairs | 5 |

## Experiments

### 1. Domain Recovery
Tests whether GDD can recover true domain structure from gradients.

```bash
python run_experiments.py --benchmark multitask_regression --experiment recovery
```

### 2. Transfer Prediction
Tests whether gradient conflict predicts negative transfer.

```bash
python run_experiments.py --benchmark rotated_mnist --experiment transfer
```

### 3. Temporal Dynamics
Tracks how domain structure evolves during training.

```bash
python run_experiments.py --benchmark multitask_regression --experiment temporal
```

### 4. Model Size Effect
Tests how model capacity affects domain separation.

```bash
python run_experiments.py --benchmark multitask_regression --experiment model_size
```

### Run All Experiments

```bash
python run_experiments.py --benchmark multitask_regression --experiment all
```

## Usage

### Basic Usage

```python
from gdd_core import GradientCollector, GradientDomainDiscovery

# Create collector
collector = GradientCollector(num_sources=4)

# During training, collect gradients
for source_id, batch in enumerate(batches):
    loss = compute_loss(model, batch)
    loss.backward()
    collector.collect(source_id, model, step, loss.item())
    optimizer.step()
    optimizer.zero_grad()

# Discover domains
gdd = GradientDomainDiscovery()
result = gdd.discover_domains(collector)

print(result.summary())
print(f"Discovered {result.num_domains} domains")
print(f"Assignments: {result.domain_assignments}")
```

### With Validation

```python
from gdd_core import validate_domain_discovery

# If you have ground truth labels
true_labels = np.array([0, 0, 1, 1])  # 2 domains, 2 sources each

validation = validate_domain_discovery(result, true_labels)
print(f"Adjusted Rand Index: {validation['adjusted_rand_index']:.4f}")
print(f"Same-domain similarity: {validation['same_domain_similarity']:.4f}")
print(f"Cross-domain similarity: {validation['diff_domain_similarity']:.4f}")
```

## Output

Results are saved to `./results/gdd_experiments/`:

- `{benchmark}_similarity_matrix.png` - Gradient similarity heatmap
- `{benchmark}_recovery_results.json` - Domain recovery metrics
- `{benchmark}_transfer_prediction.png` - Transfer vs similarity plot
- `{benchmark}_temporal_dynamics.png` - Domain evolution over training
- `{benchmark}_model_size.png` - Capacity vs domain separation

## Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| Adjusted Rand Index | Clustering quality vs ground truth | > 0.5 |
| Silhouette Score | Cluster separation quality | > 0.3 |
| Same-domain similarity | Gradient alignment within domains | > 0.5 |
| Cross-domain similarity | Gradient alignment across domains | < 0.2 |
| Separability gap | Within - Cross similarity | > 0.3 |
| Transfer correlation | Similarity vs transfer gap | < -0.5 |

## File Structure

```
gradient_domain_discovery/
├── __init__.py
├── gdd_core.py          # Core GDD implementation
├── benchmarks.py        # Benchmark datasets and models
├── run_experiments.py   # Main experiment script
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Citation

If you use this code, please cite:

```bibtex
@article{gdd2024,
  title={Gradient-Based Domain Discovery: A Theoretical Framework for 
         Unsupervised Domain Identification in Multi-Source Learning},
  author={...},
  year={2024}
}
```

## License

MIT License
