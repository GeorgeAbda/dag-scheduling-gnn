# Quick Start Guide: Gradient-Based Domain Clustering for MORL

This guide gets you up and running with gradient-based domain clustering in under 5 minutes.

## Installation

### Minimal Installation (Synthetic Domains Only)

```bash
cd experiments/morl_gradient_clustering
pip install numpy torch scikit-learn matplotlib seaborn
```

### Full Installation (With MO-Gymnasium)

```bash
cd experiments/morl_gradient_clustering
pip install -r requirements.txt
pip install mo-gymnasium
```

### With MuJoCo Support

```bash
pip install "mo-gymnasium[mujoco]"
```

## Quick Test

Run the test script to verify everything works:

```bash
python test_implementation.py
```

Expected output:
```
[Bonus Test] Testing policy network...
✓ Policy network works

[Test 1] Creating synthetic domains...
✓ Created 6 synthetic domains

[Test 2] Initializing clustering...
✓ Clustering initialized

[Test 3] Computing gradients...
✓ Computed gradients for 6 domains

[Test 4] Building similarity matrix...
✓ Built similarity matrix: shape=(6, 6)

[Test 5] Applying spectral clustering...
✓ Clustering complete

[Test 6] Evaluating clustering...
✓ Evaluation complete

[Test 7] Getting cluster summary...
✓ Cluster summary

ALL TESTS PASSED ✓
```

## Run Your First Experiment

### Option 1: Synthetic Domains (Fastest)

```bash
python run_clustering_experiment.py --experiment synthetic --n_domains 9 --n_clusters 3
```

This will:
1. Create 9 synthetic domains with known cluster structure
2. Compute policy gradients for different reward trade-offs
3. Build similarity matrix and apply spectral clustering
4. Generate visualization and save results to `results/`

**Runtime**: ~2-5 minutes

### Option 2: MO-Gymnasium Environments

```bash
python run_clustering_experiment.py --experiment mo_gym_default --n_clusters 3
```

This will:
1. Load MO-Gymnasium environments (Deep Sea Treasure, Minecart, etc.)
2. Compute gradients and cluster domains
3. Generate results

**Runtime**: ~10-20 minutes

### Option 3: MO-MuJoCo Environments

```bash
python run_clustering_experiment.py --experiment mo_gym_mujoco --n_clusters 3
```

This will:
1. Load MO-MuJoCo continuous control tasks
2. Compute gradients and cluster domains
3. Generate results

**Runtime**: ~30-60 minutes (requires MuJoCo)

## Understanding the Output

### Console Output

```
================================================================================
GRADIENT-BASED DOMAIN CLUSTERING: SYNTHETIC EXPERIMENT
================================================================================

[Step 1] Creating 9 synthetic domains...
Created domains: ['domain_0', 'domain_1', ..., 'domain_8']

Domain characteristics:
  domain_0: reward_weights = [0.823, 0.177]  # Favors objective 1
  domain_1: reward_weights = [0.791, 0.209]
  domain_2: reward_weights = [0.856, 0.144]
  domain_3: reward_weights = [0.512, 0.488]  # Balanced
  domain_4: reward_weights = [0.489, 0.511]
  domain_5: reward_weights = [0.534, 0.466]
  domain_6: reward_weights = [0.198, 0.802]  # Favors objective 2
  domain_7: reward_weights = [0.223, 0.777]
  domain_8: reward_weights = [0.176, 0.824]

[Step 2] Configuring gradient-based clustering...
  Number of clusters: 3
  Gradient samples per domain: 50
  Alpha values: [0.0, 0.25, 0.5, 0.75, 1.0]

[Step 3] Computing policy gradients...
This may take a few minutes...
Processing domain: domain_0
  Computing gradient for α=0.00
  Computing gradient for α=0.25
  ...

[Step 4] Building similarity matrix...
Similarity matrix shape: (9, 9)
Similarity range: [-0.234, 1.000]

[Step 5] Applying spectral clustering...
Cluster labels: [0 0 0 1 1 1 2 2 2]

[Step 6] Evaluating clustering results...

Clustering Metrics:
  adjusted_rand_index: 0.9234
  normalized_mutual_info: 0.8567
  silhouette_score: 0.4521

Cluster Summary:
  Cluster 0: ['domain_0', 'domain_1', 'domain_2']
  Cluster 1: ['domain_3', 'domain_4', 'domain_5']
  Cluster 2: ['domain_6', 'domain_7', 'domain_8']

[Step 7] Visualizing results...
Figure saved to results/clustering_results_synthetic.png

Results saved to results/clustering_results_synthetic.txt

================================================================================
EXPERIMENT COMPLETE
================================================================================
```

### Generated Files

After running, you'll find in the `results/` directory:

1. **`clustering_results_synthetic.png`**: Visualization showing:
   - Similarity matrix heatmap
   - Cluster assignments

2. **`clustering_results_synthetic.txt`**: Text summary with:
   - Configuration parameters
   - Clustering metrics
   - Cluster assignments
   - True vs. predicted labels

## Interpreting Results

### Similarity Matrix

The heatmap shows cosine similarity between domain gradients:
- **Red (1.0)**: Very similar domains (same reward structure)
- **White (0.0)**: Orthogonal domains
- **Blue (-1.0)**: Opposite domains

Expect to see block structure if clustering is successful.

### Clustering Metrics

**Adjusted Rand Index (ARI)**:
- 1.0 = Perfect clustering
- 0.8-1.0 = Excellent
- 0.6-0.8 = Good
- 0.4-0.6 = Moderate
- < 0.4 = Poor

**Normalized Mutual Information (NMI)**:
- 1.0 = Perfect information sharing
- 0.7-1.0 = Strong
- 0.5-0.7 = Moderate
- < 0.5 = Weak

**Silhouette Score**:
- 1.0 = Perfect separation
- 0.5-1.0 = Well-separated
- 0.3-0.5 = Reasonable
- < 0.3 = Weak structure

## Customization

### Adjust Number of Gradient Samples

More samples = better gradient estimates but slower:

```bash
python run_clustering_experiment.py \
    --experiment synthetic \
    --n_gradient_samples 100  # Default: 50
```

### Change Number of Clusters

```bash
python run_clustering_experiment.py \
    --experiment synthetic \
    --n_clusters 4  # Default: 3
```

### Adjust Domain Complexity

```bash
python run_clustering_experiment.py \
    --experiment synthetic \
    --n_domains 12 \
    --obs_dim 8 \
    --action_dim 4
```

## Programmatic Usage

For more control, use the Python API directly:

```python
from gradient_domain_clustering import GradientDomainClustering, ClusteringConfig
from mo_gymnasium_loader import create_synthetic_mo_domains

# Create domains
domains = create_synthetic_mo_domains(n_domains=9)

# Configure
config = ClusteringConfig(
    n_clusters=3,
    n_gradient_samples=50,
    alpha_values=[0.0, 0.5, 1.0]
)

# Run clustering
clustering = GradientDomainClustering(config)
clustering.compute_domain_gradients(domains, obs_dim=4, action_dim=2)
clustering.build_similarity_matrix()
labels = clustering.apply_spectral_clustering()

# Evaluate
metrics = clustering.evaluate_clustering(true_labels)
print(f"ARI: {metrics['adjusted_rand_index']:.3f}")

# Visualize
clustering.visualize_results(list(domains.keys()), 'my_results.png')
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:
```bash
pip install numpy torch scikit-learn matplotlib seaborn
```

### MO-Gymnasium Not Found

If you want to use real MORL environments:
```bash
pip install mo-gymnasium
```

### Slow Execution

Reduce gradient samples for faster testing:
```bash
python run_clustering_experiment.py --experiment synthetic --n_gradient_samples 10
```

### Memory Issues

Reduce number of domains or use smaller networks:
```python
config = ClusteringConfig(
    policy_hidden_dims=[16, 16]  # Smaller network
)
```

## Next Steps

1. **Read the full README**: `README.md` for detailed documentation
2. **Explore datasets**: `DATASET_DETAILS.md` for information on available datasets
3. **Check implementation**: `IMPLEMENTATION_SUMMARY.md` for technical details
4. **Modify the code**: All source files are well-documented

## Common Use Cases

### Research: Test on New Environments

```python
# Add your custom environment
from mo_gymnasium_loader import MOEnvironmentWrapper

my_env = MOEnvironmentWrapper('my-custom-env-v0')
domains = {'my_env': my_env, ...}

# Run clustering
clustering.compute_domain_gradients(domains, obs_dim, action_dim)
```

### Education: Demonstrate Clustering

Use synthetic domains with clear structure:
```bash
python run_clustering_experiment.py \
    --experiment synthetic \
    --n_domains 6 \
    --n_clusters 2 \
    --n_gradient_samples 20
```

### Benchmarking: Compare Methods

Run on standard MO-Gymnasium environments:
```bash
python run_clustering_experiment.py --experiment mo_gym_default
```

## Getting Help

- Check `README.md` for detailed documentation
- Run `python test_implementation.py` to verify setup
- Review example code in `run_clustering_experiment.py`
- Check error messages for specific issues

## Summary

You now have a working implementation of gradient-based domain clustering for MORL! The method:

✓ Works with multiple MORL datasets (MO-Gymnasium, D4MORL, synthetic)  
✓ Computes policy gradients across different reward trade-offs  
✓ Uses spectral clustering to group similar domains  
✓ Provides comprehensive evaluation metrics  
✓ Generates publication-quality visualizations  

Start with synthetic domains to verify everything works, then move to real MORL environments for your research.
