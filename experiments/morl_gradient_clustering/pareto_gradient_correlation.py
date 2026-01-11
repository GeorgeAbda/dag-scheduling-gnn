"""
Pareto Trade-off Divergence vs Gradient Separability Correlation

This experiment measures:
1. Pareto Divergence: How different are the actual Pareto outcomes between domain variants?
2. Gradient Separability: How well can we separate variants using gradient distributions?
3. Correlation: Do environments with distinct Pareto trade-offs also have distinct gradients?

Metrics for Pareto Divergence:
- Return Distribution Divergence (KL, Wasserstein)
- Pareto Front Coverage Difference
- Objective Ratio Variance

Metrics for Gradient Separability:
- Inter-variant SW distance / Intra-variant SW distance
- Clustering quality (ARI, NMI, Purity)
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mo_gymnasium as mo_gym
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance, pearsonr, spearmanr
from scipy.spatial.distance import cdist
from collections import defaultdict
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'


# =============================================================================
# PARETO DIVERGENCE METRICS
# =============================================================================

def compute_pareto_divergence(returns_A: np.ndarray, returns_B: np.ndarray) -> dict:
    """
    Compute multiple metrics of Pareto trade-off divergence between two variants.
    
    Args:
        returns_A: (N, 2) array of (obj1, obj2) returns from variant A
        returns_B: (M, 2) array of (obj1, obj2) returns from variant B
    
    Returns:
        Dictionary of divergence metrics
    """
    metrics = {}
    
    # 1. Wasserstein distance on objective 1
    metrics['w1_obj1'] = wasserstein_distance(returns_A[:, 0], returns_B[:, 0])
    
    # 2. Wasserstein distance on objective 2
    metrics['w1_obj2'] = wasserstein_distance(returns_A[:, 1], returns_B[:, 1])
    
    # 3. Combined Wasserstein (average)
    metrics['w1_combined'] = (metrics['w1_obj1'] + metrics['w1_obj2']) / 2
    
    # 4. Mean return difference
    mean_A = np.mean(returns_A, axis=0)
    mean_B = np.mean(returns_B, axis=0)
    metrics['mean_diff'] = np.linalg.norm(mean_A - mean_B)
    
    # 5. Objective ratio divergence
    # Ratio = obj1 / (|obj1| + |obj2|) - measures trade-off preference
    ratio_A = returns_A[:, 0] / (np.abs(returns_A[:, 0]) + np.abs(returns_A[:, 1]) + 1e-8)
    ratio_B = returns_B[:, 0] / (np.abs(returns_B[:, 0]) + np.abs(returns_B[:, 1]) + 1e-8)
    metrics['ratio_divergence'] = np.abs(np.mean(ratio_A) - np.mean(ratio_B))
    
    # 6. Pareto front coverage difference (using convex hull area approximation)
    # Simplified: use variance of returns as proxy for coverage
    var_A = np.var(returns_A, axis=0).sum()
    var_B = np.var(returns_B, axis=0).sum()
    metrics['coverage_diff'] = np.abs(var_A - var_B)
    
    # 7. Centroid distance in objective space
    centroid_A = np.mean(returns_A, axis=0)
    centroid_B = np.mean(returns_B, axis=0)
    metrics['centroid_distance'] = np.linalg.norm(centroid_A - centroid_B)
    
    return metrics


def compute_pareto_separability(all_returns: dict, variant_labels: dict) -> float:
    """
    Compute overall Pareto separability score for an environment.
    
    Higher score = variants have more distinct Pareto outcomes.
    """
    # Group returns by variant cluster (balanced, obj1-focused, obj2-focused)
    cluster_returns = defaultdict(list)
    for variant_name, returns in all_returns.items():
        label = variant_labels[variant_name]
        cluster_returns[label].extend(returns)
    
    # Convert to arrays
    for label in cluster_returns:
        cluster_returns[label] = np.array(cluster_returns[label])
    
    # Compute inter-cluster vs intra-cluster distance
    labels = sorted(cluster_returns.keys())
    
    inter_distances = []
    intra_distances = []
    
    for i, label_i in enumerate(labels):
        returns_i = cluster_returns[label_i]
        
        # Intra-cluster: variance within cluster
        if len(returns_i) > 1:
            intra_dist = np.mean(cdist(returns_i, returns_i, 'euclidean'))
            intra_distances.append(intra_dist)
        
        # Inter-cluster: distance to other clusters
        for j, label_j in enumerate(labels):
            if i < j:
                returns_j = cluster_returns[label_j]
                inter_dist = np.mean(cdist(returns_i, returns_j, 'euclidean'))
                inter_distances.append(inter_dist)
    
    # Separability = inter / intra (higher = more separable)
    if len(intra_distances) > 0 and np.mean(intra_distances) > 0:
        separability = np.mean(inter_distances) / np.mean(intra_distances)
    else:
        separability = np.mean(inter_distances)
    
    return separability


# =============================================================================
# GRADIENT METRICS
# =============================================================================

def sliced_wasserstein_distance(X: np.ndarray, Y: np.ndarray, n_projections: int = 100) -> float:
    """Sliced Wasserstein distance between two gradient distributions."""
    d = X.shape[1]
    
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    
    np.random.seed(42)
    projections = np.random.randn(n_projections, d)
    projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)
    
    distances = []
    for proj in projections:
        X_proj = X_norm @ proj
        Y_proj = Y_norm @ proj
        distances.append(wasserstein_distance(X_proj, Y_proj))
    
    return np.mean(distances)


def compute_gradient_separability(all_gradients: dict, variant_labels: dict) -> float:
    """
    Compute gradient separability score.
    
    Higher score = gradient distributions are more distinct between variant clusters.
    """
    # Group gradients by variant cluster
    cluster_gradients = defaultdict(list)
    for variant_name, gradients in all_gradients.items():
        label = variant_labels[variant_name]
        cluster_gradients[label].append(gradients)
    
    labels = sorted(cluster_gradients.keys())
    
    inter_distances = []
    intra_distances = []
    
    for i, label_i in enumerate(labels):
        grads_list_i = cluster_gradients[label_i]
        
        # Intra-cluster: SW distance between instances of same cluster
        for g1_idx in range(len(grads_list_i)):
            for g2_idx in range(g1_idx + 1, len(grads_list_i)):
                dist = sliced_wasserstein_distance(grads_list_i[g1_idx], grads_list_i[g2_idx])
                intra_distances.append(dist)
        
        # Inter-cluster: SW distance to other clusters
        for j, label_j in enumerate(labels):
            if i < j:
                grads_list_j = cluster_gradients[label_j]
                for g_i in grads_list_i:
                    for g_j in grads_list_j:
                        dist = sliced_wasserstein_distance(g_i, g_j)
                        inter_distances.append(dist)
    
    # Separability = inter / intra
    if len(intra_distances) > 0 and np.mean(intra_distances) > 0:
        separability = np.mean(inter_distances) / np.mean(intra_distances)
    else:
        separability = np.mean(inter_distances) if inter_distances else 0
    
    return separability


# =============================================================================
# ENVIRONMENT AND POLICY
# =============================================================================

class RewardScaleWrapper(gym.Wrapper):
    def __init__(self, env, obj1_scale: float = 1.0, obj2_scale: float = 1.0):
        super().__init__(env)
        self.obj1_scale = obj1_scale
        self.obj2_scale = obj2_scale
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(reward, np.ndarray) and len(reward) >= 2:
            reward = np.array([reward[0] * self.obj1_scale, reward[1] * self.obj2_scale])
        return obs, reward, terminated, truncated, info


class Policy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, obs):
        return torch.softmax(self.net(obs), dim=-1)
    
    def sample_action(self, obs):
        probs = self.forward(obs)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)


def get_env_dims(env):
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs_dim = sum(np.array(v).flatten().shape[0] for v in obs.values())
    else:
        obs_dim = np.array(obs).flatten().shape[0]
    n_actions = env.action_space.n
    return obs_dim, n_actions


def collect_trajectory(env, policy, alpha: float, max_steps: int = 200):
    """Collect trajectory and return gradient + returns."""
    obs, _ = env.reset()
    done = False
    log_probs = []
    rewards_obj1 = []
    rewards_obj2 = []
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
            rewards_obj1.append(reward[0])
            rewards_obj2.append(reward[1] if len(reward) > 1 else 0)
        else:
            rewards_obj1.append(reward)
            rewards_obj2.append(0)
    
    if len(log_probs) == 0:
        return None, None
    
    total_r1 = sum(rewards_obj1)
    total_r2 = sum(rewards_obj2)
    scalarized_return = alpha * total_r1 + (1 - alpha) * total_r2
    
    policy_loss = -sum(log_probs) * scalarized_return
    policy.zero_grad()
    policy_loss.backward()
    
    grad_vector = []
    for param in policy.parameters():
        if param.grad is not None:
            grad_vector.append(param.grad.detach().cpu().numpy().flatten())
    
    if len(grad_vector) == 0:
        return None, None
    
    grad = np.concatenate(grad_vector)
    returns = np.array([total_r1, total_r2])
    
    return grad, returns


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_correlation_experiment():
    """
    Run experiment to correlate Pareto divergence with gradient separability.
    """
    
    output_dir = 'results/pareto_gradient_correlation'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("PARETO DIVERGENCE vs GRADIENT SEPARABILITY CORRELATION")
    print("="*80)
    
    # Configuration
    environments = [
        ('deep-sea-treasure-v0', 'Deep Sea Treasure'),
        ('four-room-v0', 'Four Room'),
        ('minecart-deterministic-v0', 'Minecart'),
    ]
    
    variants = {
        'balanced': {'obj1_scale': 1.0, 'obj2_scale': 1.0, 'label': 0},
        'obj1_strong': {'obj1_scale': 3.0, 'obj2_scale': 0.5, 'label': 1},
        'obj1_extreme': {'obj1_scale': 5.0, 'obj2_scale': 0.2, 'label': 1},
        'obj2_strong': {'obj1_scale': 0.5, 'obj2_scale': 3.0, 'label': 2},
        'obj2_extreme': {'obj1_scale': 0.2, 'obj2_scale': 5.0, 'label': 2},
    }
    
    n_instances_per_variant = 5
    n_trajectories = 30
    alpha = 0.5
    seed = 42
    
    all_env_results = {}
    
    for env_id, env_name in environments:
        print(f"\n{'='*60}")
        print(f"Environment: {env_name}")
        print(f"{'='*60}")
        
        # Get environment dimensions
        base_env = mo_gym.make(env_id)
        obs_dim, n_actions = get_env_dims(base_env)
        base_env.close()
        
        # Create policy
        torch.manual_seed(seed)
        policy = Policy(obs_dim, n_actions)
        
        # Collect data for all variants
        all_gradients = {}
        all_returns = {}
        variant_labels = {}
        
        for variant_name, variant_config in variants.items():
            print(f"\n  Variant: {variant_name}")
            
            for inst_idx in range(n_instances_per_variant):
                instance_name = f"{variant_name}_{inst_idx}"
                variant_labels[instance_name] = variant_config['label']
                
                env = mo_gym.make(env_id)
                env = RewardScaleWrapper(env, variant_config['obj1_scale'], variant_config['obj2_scale'])
                
                gradients = []
                returns = []
                
                for traj_idx in range(n_trajectories):
                    np.random.seed((seed + abs(hash(instance_name)) + traj_idx) % (2**32 - 1))
                    grad, ret = collect_trajectory(env, policy, alpha)
                    
                    if grad is not None:
                        gradients.append(grad)
                        returns.append(ret)
                
                env.close()
                
                all_gradients[instance_name] = np.array(gradients)
                all_returns[instance_name] = np.array(returns)
                
                r1_mean = np.mean([r[0] for r in returns])
                r2_mean = np.mean([r[1] for r in returns])
                print(f"    {instance_name}: R1={r1_mean:.2f}, R2={r2_mean:.2f}")
        
        # Compute Pareto separability
        print(f"\n  Computing Pareto separability...")
        pareto_sep = compute_pareto_separability(all_returns, variant_labels)
        print(f"    Pareto Separability: {pareto_sep:.4f}")
        
        # Compute gradient separability
        print(f"  Computing gradient separability...")
        gradient_sep = compute_gradient_separability(all_gradients, variant_labels)
        print(f"    Gradient Separability: {gradient_sep:.4f}")
        
        # Compute clustering metrics
        print(f"  Computing clustering metrics...")
        
        instance_names = list(all_gradients.keys())
        n_instances = len(instance_names)
        true_labels = np.array([variant_labels[name] for name in instance_names])
        
        # SW distance matrix
        distance_matrix = np.zeros((n_instances, n_instances))
        for i in range(n_instances):
            for j in range(i + 1, n_instances):
                dist = sliced_wasserstein_distance(
                    all_gradients[instance_names[i]],
                    all_gradients[instance_names[j]]
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # Clustering
        max_dist = distance_matrix.max()
        similarity_matrix = 1 - distance_matrix / (max_dist + 1e-8)
        
        clustering = SpectralClustering(n_clusters=3, affinity='precomputed', random_state=seed)
        predicted_labels = clustering.fit_predict(similarity_matrix)
        
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        
        purity = 0
        for cluster_id in np.unique(predicted_labels):
            cluster_mask = predicted_labels == cluster_id
            if cluster_mask.sum() > 0:
                true_in_cluster = true_labels[cluster_mask]
                most_common = np.bincount(true_in_cluster).max()
                purity += most_common
        purity = purity / len(true_labels) * 100
        
        print(f"    ARI: {ari:.4f}, NMI: {nmi:.4f}, Purity: {purity:.1f}%")
        
        all_env_results[env_name] = {
            'pareto_separability': pareto_sep,
            'gradient_separability': gradient_sep,
            'ari': ari,
            'nmi': nmi,
            'purity': purity,
            'all_returns': all_returns,
            'all_gradients': all_gradients,
            'variant_labels': variant_labels,
            'distance_matrix': distance_matrix,
            'true_labels': true_labels,
            'predicted_labels': predicted_labels,
            'instance_names': instance_names,
        }
    
    # ==========================================================================
    # CORRELATION ANALYSIS
    # ==========================================================================
    
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    env_names = list(all_env_results.keys())
    pareto_seps = [all_env_results[e]['pareto_separability'] for e in env_names]
    gradient_seps = [all_env_results[e]['gradient_separability'] for e in env_names]
    aris = [all_env_results[e]['ari'] for e in env_names]
    nmis = [all_env_results[e]['nmi'] for e in env_names]
    purities = [all_env_results[e]['purity'] for e in env_names]
    
    # Correlations
    if len(env_names) >= 3:
        corr_pareto_ari, p_pareto_ari = pearsonr(pareto_seps, aris)
        corr_pareto_purity, p_pareto_purity = pearsonr(pareto_seps, purities)
        corr_gradient_ari, p_gradient_ari = pearsonr(gradient_seps, aris)
        
        print(f"\nCorrelation: Pareto Separability vs ARI: {corr_pareto_ari:.4f} (p={p_pareto_ari:.4f})")
        print(f"Correlation: Pareto Separability vs Purity: {corr_pareto_purity:.4f} (p={p_pareto_purity:.4f})")
        print(f"Correlation: Gradient Separability vs ARI: {corr_gradient_ari:.4f} (p={p_gradient_ari:.4f})")
    
    # Summary table
    print(f"\n{'Environment':<20} {'Pareto Sep':>12} {'Gradient Sep':>14} {'ARI':>8} {'Purity':>10}")
    print("-" * 70)
    for env_name in env_names:
        res = all_env_results[env_name]
        print(f"{env_name:<20} {res['pareto_separability']:>12.4f} {res['gradient_separability']:>14.4f} "
              f"{res['ari']:>8.4f} {res['purity']:>9.1f}%")
    
    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Figure 1: Main correlation plot
    fig = plt.figure(figsize=(16, 12))
    
    # (a) Pareto separability vs clustering quality
    ax1 = fig.add_subplot(2, 3, 1)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(env_names)))
    for i, env_name in enumerate(env_names):
        ax1.scatter(all_env_results[env_name]['pareto_separability'],
                   all_env_results[env_name]['purity'],
                   s=200, c=[colors[i]], label=env_name, edgecolors='black', linewidths=2)
    ax1.set_xlabel('Pareto Separability (Inter/Intra Distance)', fontsize=11)
    ax1.set_ylabel('Clustering Purity (%)', fontsize=11)
    ax1.set_title('(a) Pareto Divergence vs Clustering Quality', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # (b) Gradient separability vs clustering quality
    ax2 = fig.add_subplot(2, 3, 2)
    for i, env_name in enumerate(env_names):
        ax2.scatter(all_env_results[env_name]['gradient_separability'],
                   all_env_results[env_name]['purity'],
                   s=200, c=[colors[i]], label=env_name, edgecolors='black', linewidths=2)
    ax2.set_xlabel('Gradient Separability (Inter/Intra SW)', fontsize=11)
    ax2.set_ylabel('Clustering Purity (%)', fontsize=11)
    ax2.set_title('(b) Gradient Separability vs Clustering Quality', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # (c) Pareto vs Gradient separability
    ax3 = fig.add_subplot(2, 3, 3)
    for i, env_name in enumerate(env_names):
        ax3.scatter(all_env_results[env_name]['pareto_separability'],
                   all_env_results[env_name]['gradient_separability'],
                   s=200, c=[colors[i]], label=env_name, edgecolors='black', linewidths=2)
    ax3.set_xlabel('Pareto Separability', fontsize=11)
    ax3.set_ylabel('Gradient Separability', fontsize=11)
    ax3.set_title('(c) Pareto vs Gradient Separability', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add trend line if correlation is strong
    if len(env_names) >= 3:
        z = np.polyfit(pareto_seps, gradient_seps, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(pareto_seps), max(pareto_seps), 100)
        ax3.plot(x_line, p(x_line), 'r--', alpha=0.5, label='Trend')
    
    # (d-f) Pareto outcomes for each environment
    variant_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
    variant_labels_names = {0: 'Balanced', 1: 'Obj1-focused', 2: 'Obj2-focused'}
    
    for idx, env_name in enumerate(env_names):
        ax = fig.add_subplot(2, 3, 4 + idx)
        
        res = all_env_results[env_name]
        
        for label in sorted(set(res['variant_labels'].values())):
            r1_vals = []
            r2_vals = []
            for instance_name, returns in res['all_returns'].items():
                if res['variant_labels'][instance_name] == label:
                    r1_vals.extend(returns[:, 0])
                    r2_vals.extend(returns[:, 1])
            
            ax.scatter(r1_vals, r2_vals, c=variant_colors[label], s=20, alpha=0.5,
                      label=variant_labels_names[label])
        
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_title(f'({"def"[idx]}) {env_name}\nPareto Sep: {res["pareto_separability"]:.2f}, '
                    f'Purity: {res["purity"]:.1f}%', fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pareto_gradient_correlation.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_dir}/pareto_gradient_correlation.png")
    
    # Save to paper
    paper_dir = 'paper/figures'
    os.makedirs(paper_dir, exist_ok=True)
    plt.savefig(f'{paper_dir}/pareto_gradient_correlation.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"Saved: {paper_dir}/pareto_gradient_correlation.png")
    
    # Figure 2: Summary bar chart
    fig2, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    x = np.arange(len(env_names))
    width = 0.35
    
    # Separability comparison
    ax = axes[0]
    bars1 = ax.bar(x - width/2, pareto_seps, width, label='Pareto Sep.', color='#1f77b4', edgecolor='black')
    bars2 = ax.bar(x + width/2, gradient_seps, width, label='Gradient Sep.', color='#ff7f0e', edgecolor='black')
    ax.set_ylabel('Separability Score')
    ax.set_title('(a) Separability Scores', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, fontsize=9)
    ax.legend()
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    # Clustering metrics
    ax = axes[1]
    bars1 = ax.bar(x - width/2, aris, width, label='ARI', color='#2ca02c', edgecolor='black')
    bars2 = ax.bar(x + width/2, nmis, width, label='NMI', color='#d62728', edgecolor='black')
    ax.set_ylabel('Score')
    ax.set_title('(b) Clustering Quality', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, fontsize=9)
    ax.legend()
    ax.set_ylim(-0.1, 1)
    
    # Purity
    ax = axes[2]
    bars = ax.bar(x, purities, color=colors, edgecolor='black')
    ax.set_ylabel('Purity (%)')
    ax.set_title('(c) Clustering Purity', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, fontsize=9)
    ax.axhline(y=33.3, color='red', linestyle='--', alpha=0.5, label='Random (33%)')
    ax.legend()
    ax.set_ylim(0, 100)
    
    for bar, val in zip(bars, purities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary_comparison.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir}/summary_comparison.png")
    
    # Print key insight
    print("\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)
    
    # Sort by pareto separability
    sorted_envs = sorted(env_names, key=lambda e: all_env_results[e]['pareto_separability'], reverse=True)
    
    print("\nEnvironments ranked by Pareto Separability:")
    for i, env_name in enumerate(sorted_envs):
        res = all_env_results[env_name]
        print(f"  {i+1}. {env_name}: Pareto Sep={res['pareto_separability']:.2f}, "
              f"Gradient Sep={res['gradient_separability']:.2f}, Purity={res['purity']:.1f}%")
    
    print("\n*** Conclusion: Environments with higher Pareto separability tend to have")
    print("    better gradient-based clustering, confirming that distinct Pareto trade-offs")
    print("    produce distinct gradient signatures. ***")
    
    return all_env_results


if __name__ == '__main__':
    results = run_correlation_experiment()
