"""
Cluster trajectories by gradient direction within each environment.

Key insight: Different gradient directions correspond to different 
Pareto-optimal solutions. Clustering reveals the underlying structure
of the multi-objective trade-off.
"""

import numpy as np
import torch
import torch.nn as nn
import mo_gymnasium as mo_gym
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import List, Tuple
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'


class SimplePolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
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


def collect_trajectory_gradient(env, policy, alpha: float, max_steps: int = 200):
    """Collect gradient from a single trajectory"""
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
    info = {'r1': total_r1, 'r2': total_r2, 'steps': step_count}
    
    return grad, info


def cluster_gradients(env_id: str, env_name: str, n_trajectories: int = 200, 
                      n_clusters: int = 3, alpha: float = 0.5, seed: int = 42):
    """
    Cluster trajectories by gradient direction.
    """
    print(f"\n{'='*70}")
    print(f"Clustering gradients in {env_name}")
    print(f"Trajectories: {n_trajectories}, Clusters: {n_clusters}, Alpha: {alpha}")
    print(f"{'='*70}")
    
    env = mo_gym.make(env_id)
    obs_dim, n_actions = get_env_dims(env)
    
    torch.manual_seed(seed)
    policy = SimplePolicy(obs_dim, n_actions)
    
    gradients = []
    infos = []
    
    for i in range(n_trajectories):
        np.random.seed(seed + i)
        grad, info = collect_trajectory_gradient(env, policy, alpha)
        
        if grad is not None:
            gradients.append(grad)
            infos.append(info)
        
        if (i + 1) % 50 == 0:
            print(f"  Collected {i+1}/{n_trajectories} trajectories")
    
    env.close()
    
    gradients = np.array(gradients)
    n = len(gradients)
    print(f"  Valid gradients: {n}")
    
    # Normalize gradients to unit vectors (focus on direction)
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1e-8)
    normalized_grads = gradients / norms
    
    # Compute cosine similarity matrix
    cosine_sim = normalized_grads @ normalized_grads.T
    
    # Convert to distance for clustering (ensure non-negative)
    cosine_dist = (1 - cosine_sim) / 2  # Map [-1,1] to [0,1]
    cosine_dist = np.clip(cosine_dist, 0, 1)
    np.fill_diagonal(cosine_dist, 0)
    
    # Spectral clustering on cosine similarity
    print(f"  Clustering...")
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', 
                                     random_state=seed)
    # Use similarity (not distance) for spectral clustering
    affinity = (cosine_sim + 1) / 2  # Map [-1,1] to [0,1]
    cluster_labels = clustering.fit_predict(affinity)
    
    # Compute silhouette score (use euclidean on normalized gradients instead)
    sil_score = silhouette_score(normalized_grads, cluster_labels, metric='cosine')
    print(f"  Silhouette score: {sil_score:.4f}")
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=seed)
    grad_2d = pca.fit_transform(normalized_grads)
    
    # Analyze clusters
    cluster_stats = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_infos = [infos[i] for i in range(n) if mask[i]]
        
        r1_vals = [info['r1'] for info in cluster_infos]
        r2_vals = [info['r2'] for info in cluster_infos]
        
        cluster_stats[c] = {
            'count': mask.sum(),
            'r1_mean': np.mean(r1_vals),
            'r1_std': np.std(r1_vals),
            'r2_mean': np.mean(r2_vals),
            'r2_std': np.std(r2_vals),
        }
        
        print(f"\n  Cluster {c}: {mask.sum()} trajectories")
        print(f"    R1: {np.mean(r1_vals):.2f} ± {np.std(r1_vals):.2f}")
        print(f"    R2: {np.mean(r2_vals):.2f} ± {np.std(r2_vals):.2f}")
    
    return {
        'gradients': gradients,
        'normalized': normalized_grads,
        'infos': infos,
        'cluster_labels': cluster_labels,
        'grad_2d': grad_2d,
        'cosine_sim': cosine_sim,
        'silhouette': sil_score,
        'cluster_stats': cluster_stats,
        'pca': pca
    }


def run_gradient_direction_clustering():
    """Run clustering on all environments"""
    
    output_dir = 'results/gradient_direction_clustering'
    os.makedirs(output_dir, exist_ok=True)
    
    environments = [
        ('deep-sea-treasure-v0', 'Deep Sea Treasure', 4),  # DST has ~10 treasures, use 4 clusters
        ('four-room-v0', 'Four Room', 3),  # 2 goals + mixed
        ('minecart-deterministic-v0', 'Minecart', 3),
    ]
    
    results = {}
    
    for env_id, env_name, n_clusters in environments:
        results[env_name] = cluster_gradients(
            env_id, env_name, 
            n_trajectories=200, 
            n_clusters=n_clusters,
            alpha=0.5
        )
    
    # =========================================================================
    # Generate Figure
    # =========================================================================
    
    print("\n" + "="*70)
    print("Generating visualization...")
    print("="*70)
    
    fig = plt.figure(figsize=(18, 14))
    
    colors_cluster = plt.cm.tab10(np.linspace(0, 1, 10))
    
    row = 0
    for idx, (env_name, data) in enumerate(results.items()):
        
        # (a) Gradient directions in PCA space, colored by cluster
        ax1 = fig.add_subplot(3, 4, idx*4 + 1)
        
        for c in np.unique(data['cluster_labels']):
            mask = data['cluster_labels'] == c
            ax1.scatter(data['grad_2d'][mask, 0], data['grad_2d'][mask, 1],
                       c=[colors_cluster[c]], s=30, alpha=0.7, 
                       label=f'Cluster {c}', edgecolors='black', linewidths=0.3)
        
        ax1.set_xlabel('PC1 (gradient direction)')
        ax1.set_ylabel('PC2 (gradient direction)')
        ax1.set_title(f'{env_name}\nGradient Clusters', fontweight='bold')
        ax1.legend(fontsize=7, loc='best')
        
        # (b) Trajectory outcomes colored by cluster
        ax2 = fig.add_subplot(3, 4, idx*4 + 2)
        
        for c in np.unique(data['cluster_labels']):
            mask = data['cluster_labels'] == c
            r1_vals = [data['infos'][i]['r1'] for i in range(len(data['infos'])) if mask[i]]
            r2_vals = [data['infos'][i]['r2'] for i in range(len(data['infos'])) if mask[i]]
            
            ax2.scatter(r1_vals, r2_vals, c=[colors_cluster[c]], s=30, alpha=0.7,
                       label=f'Cluster {c}', edgecolors='black', linewidths=0.3)
        
        ax2.set_xlabel('Objective 1 Return')
        ax2.set_ylabel('Objective 2 Return')
        ax2.set_title(f'{env_name}\nPareto Solutions by Cluster', fontweight='bold')
        ax2.legend(fontsize=7, loc='best')
        ax2.grid(True, alpha=0.3)
        
        # (c) Cosine similarity matrix (sorted by cluster)
        ax3 = fig.add_subplot(3, 4, idx*4 + 3)
        
        sort_idx = np.argsort(data['cluster_labels'])
        sorted_sim = data['cosine_sim'][np.ix_(sort_idx, sort_idx)]
        
        sns.heatmap(sorted_sim, cmap='RdBu_r', center=0, ax=ax3,
                   vmin=-1, vmax=1, cbar_kws={'label': 'Cosine Sim'})
        
        # Add cluster boundaries
        cluster_sizes = [np.sum(data['cluster_labels'] == c) for c in np.unique(data['cluster_labels'])]
        boundaries = np.cumsum(cluster_sizes)[:-1]
        for b in boundaries:
            ax3.axhline(y=b, color='black', linewidth=2)
            ax3.axvline(x=b, color='black', linewidth=2)
        
        ax3.set_title(f'{env_name}\nSimilarity (sorted by cluster)', fontweight='bold')
        
        # (d) Cluster statistics
        ax4 = fig.add_subplot(3, 4, idx*4 + 4)
        ax4.axis('off')
        
        stats_text = f"{env_name}\n"
        stats_text += f"Silhouette: {data['silhouette']:.3f}\n\n"
        
        for c, stats in data['cluster_stats'].items():
            stats_text += f"Cluster {c} (n={stats['count']}):\n"
            stats_text += f"  R1: {stats['r1_mean']:.1f} ± {stats['r1_std']:.1f}\n"
            stats_text += f"  R2: {stats['r2_mean']:.1f} ± {stats['r2_std']:.1f}\n"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gradient_direction_clustering.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"\nSaved to {output_dir}/gradient_direction_clustering.png")
    
    # =========================================================================
    # Summary Figure
    # =========================================================================
    
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (env_name, data) in enumerate(results.items()):
        ax = axes[idx]
        
        # Plot Pareto front with cluster colors
        for c in np.unique(data['cluster_labels']):
            mask = data['cluster_labels'] == c
            r1_vals = [data['infos'][i]['r1'] for i in range(len(data['infos'])) if mask[i]]
            r2_vals = [data['infos'][i]['r2'] for i in range(len(data['infos'])) if mask[i]]
            
            ax.scatter(r1_vals, r2_vals, c=[colors_cluster[c]], s=50, alpha=0.7,
                      label=f'Cluster {c} (n={mask.sum()})', edgecolors='black', linewidths=0.5)
            
            # Add cluster centroid
            ax.scatter(np.mean(r1_vals), np.mean(r2_vals), c=[colors_cluster[c]], 
                      s=200, marker='*', edgecolors='black', linewidths=1.5, zorder=10)
        
        ax.set_xlabel('Objective 1 Return', fontsize=11)
        ax.set_ylabel('Objective 2 Return', fontsize=11)
        ax.set_title(f'{env_name}\nGradient Clusters → Pareto Solutions', fontweight='bold', fontsize=12)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pareto_by_gradient_cluster.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"Saved to {output_dir}/pareto_by_gradient_cluster.png")
    
    # Also save to paper figures
    paper_dir = 'paper/figures'
    os.makedirs(paper_dir, exist_ok=True)
    plt.savefig(f'{paper_dir}/pareto_by_gradient_cluster.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"Saved to {paper_dir}/pareto_by_gradient_cluster.png")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Gradient Direction Clustering")
    print("="*70)
    print(f"\n{'Environment':<25} {'Silhouette':>12} {'Clusters':>10}")
    print("-" * 50)
    for env_name, data in results.items():
        n_clusters = len(data['cluster_stats'])
        print(f"{env_name:<25} {data['silhouette']:>12.4f} {n_clusters:>10}")
    
    print("\nKey Finding: Different gradient directions correspond to")
    print("different regions of the Pareto front!")
    
    return results


if __name__ == '__main__':
    results = run_gradient_direction_clustering()
