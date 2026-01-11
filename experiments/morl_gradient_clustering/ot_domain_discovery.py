"""
Optimal Transport Enhanced Domain Discovery for Multi-Objective RL

This module integrates Optimal Transport (OT) into gradient-based domain discovery:
1. Wasserstein distance between gradient distributions
2. Sliced Wasserstein for scalability
3. OT barycenters for domain prototypes
4. Gradient flow analysis via OT

Key insight: Instead of comparing mean gradients, compare the DISTRIBUTION of gradients
using OT metrics. This captures richer information about domain structure.
"""

import numpy as np
import torch
import torch.nn as nn
import ot  # POT: Python Optimal Transport
import mo_gymnasium as mo_gym
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class DiscretePolicy(nn.Module):
    """Simple MLP policy"""
    
    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_actions))
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs):
        return torch.softmax(self.network(obs), dim=-1)
    
    def sample_action(self, obs):
        probs = self.forward(obs)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)


def compute_single_gradient(env, policy, alpha: float, max_steps: int = 200) -> np.ndarray:
    """Compute gradient from a single episode"""
    obs, _ = env.reset()
    done = False
    log_probs, rewards = [], []
    step_count = 0
    
    while not done and step_count < max_steps:
        if isinstance(obs, dict):
            obs = np.concatenate([v.flatten() for v in obs.values()])
        obs = np.array(obs).flatten()
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action, log_prob = policy.sample_action(obs_tensor)
        log_probs.append(log_prob)
        
        obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        step_count += 1
        
        if isinstance(reward, np.ndarray):
            rewards.append(reward)
        else:
            rewards.append(np.array([reward, 0.0]))
    
    if len(rewards) == 0:
        return None
    
    rewards = np.array(rewards)
    r1 = rewards[:, 0].sum()
    r2 = rewards[:, 1].sum() if rewards.shape[1] >= 2 else 0.0
    aggregate_return = alpha * r1 + (1 - alpha) * r2
    
    if len(log_probs) > 0:
        policy_loss = -sum(log_probs) * aggregate_return
        
        for param in policy.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        policy_loss.backward()
        
        grad_vector = []
        for param in policy.parameters():
            if param.grad is not None:
                grad_vector.append(param.grad.detach().cpu().numpy().flatten())
        
        if len(grad_vector) > 0:
            return np.concatenate(grad_vector)
    
    return None


def collect_gradient_samples(
    env,
    obs_dim: int,
    n_actions: int,
    n_samples: int = 100,
    alpha_values: List[float] = None,
    seed: int = 42
) -> np.ndarray:
    """
    Collect multiple gradient samples from an environment.
    Returns a distribution of gradients (n_samples x grad_dim).
    """
    if alpha_values is None:
        alpha_values = [0.0, 0.5, 1.0]
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    gradients = []
    samples_per_alpha = n_samples // len(alpha_values)
    
    for alpha in alpha_values:
        policy = DiscretePolicy(obs_dim, n_actions, hidden_dims=[64, 64])
        
        for _ in range(samples_per_alpha):
            grad = compute_single_gradient(env, policy, alpha)
            if grad is not None:
                gradients.append(grad)
    
    return np.array(gradients) if gradients else None


def get_env_dims(env):
    """Get environment dimensions"""
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs_dim = sum(np.array(v).flatten().shape[0] for v in obs.values())
    else:
        obs_dim = np.array(obs).flatten().shape[0]
    n_actions = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    return obs_dim, n_actions


# =============================================================================
# OPTIMAL TRANSPORT METRICS
# =============================================================================

def wasserstein_distance(X: np.ndarray, Y: np.ndarray, p: int = 2) -> float:
    """
    Compute Wasserstein-p distance between two empirical distributions.
    
    Args:
        X: (n, d) array - samples from distribution 1
        Y: (m, d) array - samples from distribution 2
        p: order of Wasserstein distance
    
    Returns:
        Wasserstein-p distance
    """
    n, m = len(X), len(Y)
    
    # Uniform weights
    a = np.ones(n) / n
    b = np.ones(m) / m
    
    # Cost matrix (squared Euclidean)
    M = ot.dist(X, Y, metric='sqeuclidean')
    
    # Compute OT distance
    if p == 2:
        return np.sqrt(ot.emd2(a, b, M))
    else:
        return ot.emd2(a, b, M ** (p/2)) ** (1/p)


def sliced_wasserstein_distance(X: np.ndarray, Y: np.ndarray, n_projections: int = 100) -> float:
    """
    Compute Sliced Wasserstein distance (scalable approximation).
    
    Projects distributions onto random 1D subspaces and averages 1D Wasserstein distances.
    Complexity: O(n log n) per projection vs O(n^3) for exact Wasserstein.
    """
    d = X.shape[1]
    
    # Random projection directions
    np.random.seed(42)
    projections = np.random.randn(n_projections, d)
    projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)
    
    sw_distances = []
    for proj in projections:
        # Project to 1D
        X_proj = X @ proj
        Y_proj = Y @ proj
        
        # Sort and compute 1D Wasserstein
        X_sorted = np.sort(X_proj)
        Y_sorted = np.sort(Y_proj)
        
        # Interpolate to same length if needed
        if len(X_sorted) != len(Y_sorted):
            n_interp = max(len(X_sorted), len(Y_sorted))
            X_interp = np.interp(np.linspace(0, 1, n_interp), 
                                 np.linspace(0, 1, len(X_sorted)), X_sorted)
            Y_interp = np.interp(np.linspace(0, 1, n_interp),
                                 np.linspace(0, 1, len(Y_sorted)), Y_sorted)
            sw_distances.append(np.mean((X_interp - Y_interp) ** 2))
        else:
            sw_distances.append(np.mean((X_sorted - Y_sorted) ** 2))
    
    return np.sqrt(np.mean(sw_distances))


def sinkhorn_distance(X: np.ndarray, Y: np.ndarray, reg: float = 0.1) -> float:
    """
    Compute Sinkhorn distance (entropy-regularized OT).
    
    Faster than exact Wasserstein, differentiable.
    """
    n, m = len(X), len(Y)
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(X, Y, metric='sqeuclidean')
    
    return np.sqrt(ot.sinkhorn2(a, b, M, reg))


def compute_ot_barycenter(distributions: List[np.ndarray], weights: np.ndarray = None) -> np.ndarray:
    """
    Compute Wasserstein barycenter of multiple distributions.
    
    Args:
        distributions: List of (n_i, d) arrays
        weights: Weights for each distribution (uniform if None)
    
    Returns:
        Barycenter samples (n, d)
    """
    k = len(distributions)
    if weights is None:
        weights = np.ones(k) / k
    
    # Use first distribution's size as reference
    n = len(distributions[0])
    d = distributions[0].shape[1]
    
    # Initialize barycenter as weighted mean
    barycenter = np.zeros((n, d))
    for i, dist in enumerate(distributions):
        # Resample to same size if needed
        if len(dist) != n:
            indices = np.random.choice(len(dist), n, replace=True)
            dist = dist[indices]
        barycenter += weights[i] * dist
    
    # Iterative refinement (simplified)
    for _ in range(10):
        new_barycenter = np.zeros((n, d))
        for i, dist in enumerate(distributions):
            if len(dist) != n:
                indices = np.random.choice(len(dist), n, replace=True)
                dist = dist[indices]
            
            # Compute transport plan
            a = np.ones(n) / n
            b = np.ones(len(dist)) / len(dist)
            M = ot.dist(barycenter, dist, metric='sqeuclidean')
            T = ot.emd(a, b, M)
            
            # Transport barycenter toward this distribution
            new_barycenter += weights[i] * (n * T @ dist)
        
        barycenter = new_barycenter
    
    return barycenter


# =============================================================================
# OT-ENHANCED DOMAIN DISCOVERY
# =============================================================================

class OTDomainDiscovery:
    """
    Optimal Transport enhanced domain discovery.
    
    Key differences from basic GDD:
    1. Collects DISTRIBUTION of gradients per domain (not just mean)
    2. Uses Wasserstein/Sliced-Wasserstein distance for similarity
    3. Can compute domain prototypes via OT barycenters
    """
    
    def __init__(
        self,
        n_gradient_samples: int = 100,
        alpha_values: List[float] = None,
        ot_method: str = 'sliced',  # 'wasserstein', 'sliced', 'sinkhorn'
        n_clusters: int = 3,
        seed: int = 42
    ):
        self.n_gradient_samples = n_gradient_samples
        self.alpha_values = alpha_values or np.linspace(0, 1, 5).tolist()
        self.ot_method = ot_method
        self.n_clusters = n_clusters
        self.seed = seed
        
        self.gradient_distributions = {}
        self.distance_matrix = None
        self.cluster_labels = None
    
    def collect_gradients(self, domains: Dict[str, any]) -> Dict[str, np.ndarray]:
        """Collect gradient distributions for all domains"""
        print(f"Collecting gradient distributions ({self.n_gradient_samples} samples per domain)...")
        
        for name, info in domains.items():
            env = info['env']
            obs_dim = info['obs_dim']
            n_actions = info['n_actions']
            
            print(f"  {name}...")
            grads = collect_gradient_samples(
                env, obs_dim, n_actions,
                n_samples=self.n_gradient_samples,
                alpha_values=self.alpha_values,
                seed=self.seed
            )
            
            if grads is not None:
                self.gradient_distributions[name] = grads
                print(f"    Collected {len(grads)} gradient samples, dim={grads.shape[1]}")
        
        return self.gradient_distributions
    
    def compute_distance_matrix(self) -> np.ndarray:
        """Compute pairwise OT distances between gradient distributions"""
        domain_names = list(self.gradient_distributions.keys())
        n = len(domain_names)
        
        print(f"\nComputing {self.ot_method} distance matrix...")
        
        # Pad gradients to same dimension
        max_dim = max(g.shape[1] for g in self.gradient_distributions.values())
        padded_grads = {}
        for name, grads in self.gradient_distributions.items():
            if grads.shape[1] < max_dim:
                padded = np.zeros((grads.shape[0], max_dim))
                padded[:, :grads.shape[1]] = grads
                padded_grads[name] = padded
            else:
                padded_grads[name] = grads
        
        # Compute distances
        self.distance_matrix = np.zeros((n, n))
        
        for i, name_i in enumerate(domain_names):
            for j, name_j in enumerate(domain_names):
                if i < j:
                    X = padded_grads[name_i]
                    Y = padded_grads[name_j]
                    
                    if self.ot_method == 'wasserstein':
                        dist = wasserstein_distance(X, Y)
                    elif self.ot_method == 'sliced':
                        dist = sliced_wasserstein_distance(X, Y)
                    elif self.ot_method == 'sinkhorn':
                        dist = sinkhorn_distance(X, Y)
                    else:
                        raise ValueError(f"Unknown OT method: {self.ot_method}")
                    
                    self.distance_matrix[i, j] = dist
                    self.distance_matrix[j, i] = dist
                    
                    print(f"  {name_i} <-> {name_j}: {dist:.4f}")
        
        return self.distance_matrix
    
    def cluster(self) -> np.ndarray:
        """Apply clustering based on OT distance matrix"""
        # Convert distance to similarity
        max_dist = self.distance_matrix.max()
        similarity_matrix = 1 - self.distance_matrix / (max_dist + 1e-8)
        
        # Spectral clustering
        clustering = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            random_state=self.seed
        )
        self.cluster_labels = clustering.fit_predict(similarity_matrix)
        
        return self.cluster_labels
    
    def compute_barycenters(self) -> Dict[int, np.ndarray]:
        """Compute OT barycenter for each cluster"""
        barycenters = {}
        domain_names = list(self.gradient_distributions.keys())
        
        for cluster_id in np.unique(self.cluster_labels):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_domains = [domain_names[i] for i in range(len(domain_names)) if cluster_mask[i]]
            
            cluster_distributions = [self.gradient_distributions[name] for name in cluster_domains]
            
            # Pad to same dimension
            max_dim = max(g.shape[1] for g in cluster_distributions)
            padded = []
            for g in cluster_distributions:
                if g.shape[1] < max_dim:
                    p = np.zeros((g.shape[0], max_dim))
                    p[:, :g.shape[1]] = g
                    padded.append(p)
                else:
                    padded.append(g)
            
            barycenter = compute_ot_barycenter(padded)
            barycenters[cluster_id] = barycenter
            
            print(f"Cluster {cluster_id} barycenter: {barycenter.shape}")
        
        return barycenters
    
    def evaluate(self, true_labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality"""
        metrics = {
            'adjusted_rand_index': adjusted_rand_score(true_labels, self.cluster_labels),
            'normalized_mutual_info': normalized_mutual_info_score(true_labels, self.cluster_labels),
        }
        
        # Silhouette on distance matrix
        if len(np.unique(self.cluster_labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(
                self.distance_matrix, self.cluster_labels, metric='precomputed'
            )
        
        return metrics


# =============================================================================
# GRADIENT FLOW ANALYSIS VIA OT
# =============================================================================

class GradientFlowAnalysis:
    """
    Analyze gradient flow using OT perspective.
    
    Key insight: Policy optimization can be viewed as gradient flow in Wasserstein space.
    Different domains induce different "velocity fields" in this space.
    """
    
    def __init__(self, gradient_distributions: Dict[str, np.ndarray]):
        self.gradient_distributions = gradient_distributions
    
    def compute_velocity_field(self, domain_name: str, grid_points: np.ndarray) -> np.ndarray:
        """
        Compute the gradient "velocity field" at given points.
        
        This represents how the policy would update at each point in parameter space.
        """
        grads = self.gradient_distributions[domain_name]
        
        # Use kernel density estimation to interpolate gradient field
        from scipy.interpolate import RBFInterpolator
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        grads_2d = pca.fit_transform(grads)
        
        # Compute mean gradient direction at each point
        velocities = np.zeros((len(grid_points), 2))
        
        for i, point in enumerate(grid_points):
            # Find nearest gradients
            distances = np.linalg.norm(grads_2d - point, axis=1)
            weights = np.exp(-distances ** 2 / (2 * 0.5 ** 2))
            weights /= weights.sum()
            
            # Weighted average gradient
            velocities[i] = np.average(grads_2d, axis=0, weights=weights)
        
        return velocities
    
    def compute_transport_cost(self, domain_i: str, domain_j: str) -> Tuple[float, np.ndarray]:
        """
        Compute OT cost and plan between two domain gradient distributions.
        
        The transport plan shows how gradients from domain_i map to domain_j.
        """
        X = self.gradient_distributions[domain_i]
        Y = self.gradient_distributions[domain_j]
        
        # Pad to same dimension
        max_dim = max(X.shape[1], Y.shape[1])
        X_pad = np.zeros((X.shape[0], max_dim))
        Y_pad = np.zeros((Y.shape[0], max_dim))
        X_pad[:, :X.shape[1]] = X
        Y_pad[:, :Y.shape[1]] = Y
        
        # Compute OT
        n, m = len(X_pad), len(Y_pad)
        a = np.ones(n) / n
        b = np.ones(m) / m
        M = ot.dist(X_pad, Y_pad, metric='sqeuclidean')
        
        T = ot.emd(a, b, M)
        cost = np.sum(T * M)
        
        return cost, T
    
    def analyze_gradient_conflict(self) -> Dict[str, float]:
        """
        Analyze gradient conflict between domains using OT.
        
        High transport cost = high conflict (gradients point in different directions)
        """
        domain_names = list(self.gradient_distributions.keys())
        conflicts = {}
        
        for i, name_i in enumerate(domain_names):
            for j, name_j in enumerate(domain_names):
                if i < j:
                    cost, _ = self.compute_transport_cost(name_i, name_j)
                    conflicts[f"{name_i} <-> {name_j}"] = cost
        
        return conflicts


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_ot_experiment():
    """Run OT-enhanced domain discovery experiment"""
    
    print("=" * 80)
    print("OPTIMAL TRANSPORT ENHANCED DOMAIN DISCOVERY")
    print("=" * 80)
    
    # Create domains
    env_types = [
        {'env_id': 'deep-sea-treasure-v0', 'name': 'DST', 'group': 0},
        {'env_id': 'four-room-v0', 'name': 'FourRoom', 'group': 1},
        {'env_id': 'minecart-deterministic-v0', 'name': 'Minecart', 'group': 2},
    ]
    
    domains = {}
    true_labels = []
    
    print("\n[1] Creating domains...")
    for env_type in env_types:
        for instance_idx in range(3):
            domain_name = f"{env_type['name']}_{instance_idx}"
            env = mo_gym.make(env_type['env_id'])
            obs_dim, n_actions = get_env_dims(env)
            
            domains[domain_name] = {
                'env': env,
                'obs_dim': obs_dim,
                'n_actions': n_actions,
                'group': env_type['group']
            }
            true_labels.append(env_type['group'])
            print(f"  Created {domain_name}")
    
    true_labels = np.array(true_labels)
    
    # Compare different OT methods
    methods = ['sliced', 'sinkhorn']
    results = {}
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Testing OT method: {method.upper()}")
        print(f"{'='*80}")
        
        # Run OT domain discovery
        ot_discovery = OTDomainDiscovery(
            n_gradient_samples=50,
            alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0],
            ot_method=method,
            n_clusters=3,
            seed=42
        )
        
        ot_discovery.collect_gradients(domains)
        ot_discovery.compute_distance_matrix()
        predicted_labels = ot_discovery.cluster()
        
        metrics = ot_discovery.evaluate(true_labels)
        results[method] = metrics
        
        print(f"\nResults for {method}:")
        print(f"  ARI: {metrics['adjusted_rand_index']:.4f}")
        print(f"  NMI: {metrics['normalized_mutual_info']:.4f}")
        print(f"  Silhouette: {metrics.get('silhouette_score', 'N/A')}")
        print(f"  True labels:      {true_labels}")
        print(f"  Predicted labels: {predicted_labels}")
    
    # Clean up
    for info in domains.values():
        info['env'].close()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: OT Methods Comparison")
    print("=" * 80)
    print(f"\n{'Method':<15} {'ARI':>10} {'NMI':>10}")
    print("-" * 40)
    for method, metrics in results.items():
        print(f"{method:<15} {metrics['adjusted_rand_index']:>10.4f} {metrics['normalized_mutual_info']:>10.4f}")
    
    return results


if __name__ == '__main__':
    # Install POT if needed: pip install POT
    try:
        import ot
    except ImportError:
        print("Installing POT (Python Optimal Transport)...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'POT'])
        import ot
    
    results = run_ot_experiment()
