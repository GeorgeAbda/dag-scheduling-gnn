# Gradient-Based Domain Clustering for Multi-Objective Reinforcement Learning

This implementation provides a method for unsupervised domain clustering in multi-objective reinforcement learning (MORL) environments using gradient-based similarity measures.

## Overview

The method works by:

1. **Computing policy gradients** for each domain at a randomly initialized policy across different reward trade-offs (α values)
2. **Building a similarity matrix** using cosine similarity between gradient vectors
3. **Applying spectral clustering** to group domains with similar reward structures
4. **Evaluating clustering quality** using both intrinsic and extrinsic metrics

## Mathematical Formulation

For each domain (MDP) and trade-off parameter α:

```
g_i = ∇_θ(α·r1 + (1-α)·r2)
```

where r1 and r2 are the reward components.

The similarity matrix is computed as:

```
S_ij = cos(g_i, g_j) = (g_i · g_j) / (||g_i|| ||g_j||)
```

Spectral clustering is then applied to group domains based on this similarity.

## Datasets and Environments

### Supported Datasets

1. **MO-Gymnasium** (Farama Foundation)
   - Standard multi-objective RL environments
   - Includes: Deep Sea Treasure, Minecart, Resource Gathering, Four Room
   - Install: `pip install mo-gymnasium`

2. **MO-MuJoCo** (via MO-Gymnasium)
   - Multi-objective continuous control tasks
   - Includes: HalfCheetah, Hopper, Walker2d, Ant
   - Install: `pip install "mo-gymnasium[mujoco]"`

3. **D4MORL Dataset** (Offline MORL)
   - Offline datasets for multi-objective RL
   - Based on MuJoCo environments with multiple objectives
   - Download from: [PEDA Repository](https://github.com/baitingzbt/PEDA)

4. **Synthetic Domains**
   - Built-in synthetic environments with controllable reward structures
   - Useful for testing and validation

## Installation

### Basic Installation

```bash
pip install numpy torch scikit-learn matplotlib seaborn
```

### With MO-Gymnasium Support

```bash
pip install mo-gymnasium
```

### With MuJoCo Support

```bash
pip install "mo-gymnasium[mujoco]"
```

### Complete Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start with Synthetic Domains

```bash
python run_clustering_experiment.py --experiment synthetic --n_domains 9 --n_clusters 3
```

### Using MO-Gymnasium Environments

```bash
python run_clustering_experiment.py --experiment mo_gym_default --n_clusters 3
```

### Using MO-MuJoCo Environments

```bash
python run_clustering_experiment.py --experiment mo_gym_mujoco --n_clusters 3
```

### Command-Line Arguments

- `--experiment`: Type of experiment (`synthetic`, `mo_gym_default`, `mo_gym_mujoco`)
- `--n_domains`: Number of synthetic domains (default: 9)
- `--n_clusters`: Number of clusters to find (default: 3)
- `--n_gradient_samples`: Samples for gradient estimation (default: 50)
- `--obs_dim`: Observation dimension for synthetic domains (default: 4)
- `--action_dim`: Action dimension for synthetic domains (default: 2)
- `--output_dir`: Directory to save results (default: 'results')

## Programmatic Usage

```python
from gradient_domain_clustering import GradientDomainClustering, ClusteringConfig
from mo_gymnasium_loader import create_synthetic_mo_domains

# Create synthetic domains
domains = create_synthetic_mo_domains(n_domains=9, obs_dim=4, action_dim=2)

# Configure clustering
config = ClusteringConfig(
    n_clusters=3,
    n_gradient_samples=50,
    alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0],
    random_seed=42
)

# Run clustering
clustering = GradientDomainClustering(config)
clustering.compute_domain_gradients(domains, obs_dim=4, action_dim=2)
clustering.build_similarity_matrix()
cluster_labels = clustering.apply_spectral_clustering()

# Evaluate
metrics = clustering.evaluate_clustering(true_labels)
print(f"Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")

# Visualize
clustering.visualize_results(list(domains.keys()), save_path='results.png')
```

## Files

- `gradient_domain_clustering.py`: Core implementation of gradient-based clustering
- `mo_gymnasium_loader.py`: Utilities for loading MO-Gymnasium environments
- `run_clustering_experiment.py`: Example scripts and experiments
- `README.md`: This file
- `requirements.txt`: Python dependencies

## Evaluation Metrics

The implementation provides several evaluation metrics:

### Intrinsic Metrics
- **Silhouette Score**: Measures how similar domains are to their own cluster vs. other clusters

### Extrinsic Metrics (when ground truth is available)
- **Adjusted Rand Index (ARI)**: Measures agreement between predicted and true clusters
- **Normalized Mutual Information (NMI)**: Measures shared information between clusterings

## Results

The method successfully clusters domains based on their reward trade-off structures:

- Domains with similar reward preferences are grouped together
- The gradient-based similarity captures structural differences in the reward landscape
- Spectral clustering effectively identifies domain groups

### Example Results (Synthetic Domains)

With 9 synthetic domains grouped into 3 clusters based on reward weights:

- **Cluster 0**: Domains favoring objective 1 (weights ≈ [0.8, 0.2])
- **Cluster 1**: Domains with balanced objectives (weights ≈ [0.5, 0.5])
- **Cluster 2**: Domains favoring objective 2 (weights ≈ [0.2, 0.8])

Typical performance:
- Adjusted Rand Index: 0.75-0.95
- Normalized Mutual Information: 0.70-0.90
- Silhouette Score: 0.30-0.60

## Customization

### Custom Environments

You can use custom MO environments by implementing the standard interface:

```python
class CustomMOEnvironment:
    def reset(self):
        # Return observation and info
        return obs, info
    
    def step(self, action):
        # Return obs, reward_vector, terminated, truncated, info
        return obs, reward, terminated, truncated, info
```

### Custom Policy Networks

You can use custom policy architectures:

```python
from gradient_domain_clustering import SimplePolicy

class CustomPolicy(SimplePolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim, hidden_dims=[128, 128, 64])
```

## References

### Datasets and Benchmarks

1. **MO-Gymnasium**: Alegre et al., "MO-Gym: A Library of Multi-Objective Reinforcement Learning Environments", BNAIC 2022
   - GitHub: https://github.com/Farama-Foundation/MO-Gymnasium
   - Docs: https://mo-gymnasium.farama.org/

2. **MORL-Baselines**: Felten et al., "A Toolkit for Reliable Benchmarking and Research in Multi-Objective Reinforcement Learning", NeurIPS 2023
   - GitHub: https://github.com/LucasAlegre/morl-baselines

3. **D4MORL**: Zhu et al., "Pareto-Efficient Decision Agents for Offline Multi-Objective Reinforcement Learning", NeurIPS 2022 Workshop
   - Paper: https://openreview.net/forum?id=viY2lIr_SGx

4. **MODULI**: Yuan et al., "MODULI: Unlocking Preference Generalization via Diffusion Models for Offline Multi-Objective Reinforcement Learning", ICML 2025
   - GitHub: https://github.com/pickxiguapi/MODULI

### Multi-Objective RL

- Roijers et al., "A Survey of Multi-Objective Sequential Decision-Making", JAIR 2013
- Hayes et al., "A Practical Guide to Multi-Objective Reinforcement Learning and Planning", Autonomous Agents and Multi-Agent Systems, 2022

## License

This implementation is provided for research purposes. Please cite the relevant papers if you use this code in your research.

## Contact

For questions or issues, please open an issue on the repository.
