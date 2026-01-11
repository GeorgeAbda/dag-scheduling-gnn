"""
Intra-Domain Variant Clustering

Setup:
- For EACH environment (Deep Sea Treasure, Four Room, Minecart):
  - Train ONE agent on that environment
  - Create multiple DOMAIN VARIANTS with different reward scalings
  - Each variant leads to different Pareto trade-offs
  - Cluster variants based on gradient conflict patterns

This simulates: An agent deployed in one environment but facing different
"conditions" that change the trade-off structure (e.g., different energy costs,
different market conditions, etc.)
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
from scipy.stats import wasserstein_distance
from collections import defaultdict
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'


# =============================================================================
# DOMAIN VARIANT CONFIGURATIONS
# =============================================================================

def get_domain_variants():
    """
    Create domain variants with different reward scalings.
    Each variant represents a different trade-off condition.
    """
    variants = {
        'balanced': {'obj1_scale': 1.0, 'obj2_scale': 1.0, 'label': 0},
        'obj1_strong': {'obj1_scale': 3.0, 'obj2_scale': 0.5, 'label': 1},
        'obj1_extreme': {'obj1_scale': 5.0, 'obj2_scale': 0.2, 'label': 1},
        'obj2_strong': {'obj1_scale': 0.5, 'obj2_scale': 3.0, 'label': 2},
        'obj2_extreme': {'obj1_scale': 0.2, 'obj2_scale': 5.0, 'label': 2},
    }
    return variants


# =============================================================================
# ENVIRONMENT WRAPPER
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


# =============================================================================
# POLICY
# =============================================================================

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


# =============================================================================
# GRADIENT COLLECTION
# =============================================================================

def collect_trajectory_gradient(env, policy, alpha: float, max_steps: int = 200):
    """Collect gradient from a single trajectory."""
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


# =============================================================================
# DISTANCE METRICS
# =============================================================================

def sliced_wasserstein_distance(X: np.ndarray, Y: np.ndarray, n_projections: int = 100) -> float:
    """Sliced Wasserstein distance between two gradient distributions."""
    d = X.shape[1]
    
    # Normalize to unit vectors (focus on direction)
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


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_intra_domain_clustering(env_id: str, env_name: str, n_instances_per_variant: int = 5,
                                 n_trajectories: int = 30, alpha: float = 0.5, seed: int = 42):
    """
    Run clustering experiment for ONE environment with multiple domain variants.
    
    Args:
        env_id: MO-Gymnasium environment ID
        env_name: Human-readable name
        n_instances_per_variant: Number of instances per variant (for robustness)
        n_trajectories: Trajectories per instance
        alpha: Preference weight
        seed: Random seed
    """
    print(f"\n{'='*80}")
    print(f"INTRA-DOMAIN VARIANT CLUSTERING: {env_name}")
    print(f"{'='*80}")
    
    # Get environment dimensions
    base_env = mo_gym.make(env_id)
    obs_dim, n_actions = get_env_dims(base_env)
    base_env.close()
    
    print(f"Environment: {env_id}")
    print(f"Obs dim: {obs_dim}, Actions: {n_actions}")
    
    # Create ONE policy for this environment (trained on all variants)
    torch.manual_seed(seed)
    policy = Policy(obs_dim, n_actions)
    
    # Get domain variants
    variants = get_domain_variants()
    n_variants = len(variants)
    
    print(f"Domain variants: {list(variants.keys())}")
    print(f"Instances per variant: {n_instances_per_variant}")
    print(f"Trajectories per instance: {n_trajectories}")
    
    # Collect gradients for each variant instance
    all_gradients = {}
    all_infos = {}
    true_labels = []
    instance_names = []
    
    print(f"\nCollecting gradients...")
    
    for variant_name, variant_config in variants.items():
        print(f"\n  Variant: {variant_name} (scale: {variant_config['obj1_scale']}/{variant_config['obj2_scale']})")
        
        for inst_idx in range(n_instances_per_variant):
            instance_name = f"{variant_name}_{inst_idx}"
            instance_names.append(instance_name)
            true_labels.append(variant_config['label'])
            
            # Create wrapped environment
            env = mo_gym.make(env_id)
            env = RewardScaleWrapper(env, variant_config['obj1_scale'], variant_config['obj2_scale'])
            
            gradients = []
            infos = []
            
            for traj_idx in range(n_trajectories):
                np.random.seed((seed + abs(hash(instance_name)) + traj_idx) % (2**32 - 1))
                grad, info = collect_trajectory_gradient(env, policy, alpha)
                
                if grad is not None:
                    gradients.append(grad)
                    infos.append(info)
            
            env.close()
            
            all_gradients[instance_name] = np.array(gradients)
            all_infos[instance_name] = infos
            
            r1_mean = np.mean([info['r1'] for info in infos])
            r2_mean = np.mean([info['r2'] for info in infos])
            print(f"    {instance_name}: R1={r1_mean:.2f}, R2={r2_mean:.2f}")
    
    true_labels = np.array(true_labels)
    n_instances = len(instance_names)
    n_clusters = len(set(true_labels))  # 3 clusters: balanced, obj1-focused, obj2-focused
    
    # Compute pairwise SW distances
    print(f"\nComputing pairwise SW distances...")
    
    distance_matrix = np.zeros((n_instances, n_instances))
    n_pairs = n_instances * (n_instances - 1) // 2
    
    with tqdm(total=n_pairs, desc="  SW distances") as pbar:
        for i in range(n_instances):
            for j in range(i + 1, n_instances):
                dist = sliced_wasserstein_distance(
                    all_gradients[instance_names[i]],
                    all_gradients[instance_names[j]]
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                pbar.update(1)
    
    # Clustering
    max_dist = distance_matrix.max()
    similarity_matrix = 1 - distance_matrix / (max_dist + 1e-8)
    
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=seed)
    predicted_labels = clustering.fit_predict(similarity_matrix)
    
    # Metrics
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    
    try:
        sil = silhouette_score(distance_matrix, predicted_labels, metric='precomputed')
    except:
        sil = 0.0
    
    # Purity
    purity = 0
    for cluster_id in np.unique(predicted_labels):
        cluster_mask = predicted_labels == cluster_id
        if cluster_mask.sum() > 0:
            true_in_cluster = true_labels[cluster_mask]
            most_common = np.bincount(true_in_cluster).max()
            purity += most_common
    purity = purity / len(true_labels) * 100
    
    print(f"\nClustering Results for {env_name}:")
    print(f"  ARI: {ari:.4f}")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Purity: {purity:.1f}%")
    print(f"  Silhouette: {sil:.4f}")
    
    return {
        'env_name': env_name,
        'ari': ari,
        'nmi': nmi,
        'purity': purity,
        'silhouette': sil,
        'distance_matrix': distance_matrix,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'instance_names': instance_names,
        'all_gradients': all_gradients,
        'all_infos': all_infos,
        'variants': variants,
    }


def run_all_environments():
    """Run intra-domain variant clustering for all environments."""
    
    output_dir = 'results/intra_domain_variants'
    os.makedirs(output_dir, exist_ok=True)
    
    environments = [
        ('deep-sea-treasure-v0', 'Deep Sea Treasure'),
        ('four-room-v0', 'Four Room'),
        ('minecart-deterministic-v0', 'Minecart'),
    ]
    
    all_results = {}
    
    for env_id, env_name in environments:
        results = run_intra_domain_clustering(
            env_id, env_name,
            n_instances_per_variant=5,
            n_trajectories=30,
            alpha=0.5,
            seed=42
        )
        all_results[env_name] = results
    
    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Figure: One row per environment
    fig = plt.figure(figsize=(18, 15))
    
    variant_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}  # balanced, obj1, obj2
    variant_labels = {0: 'Balanced', 1: 'Obj1-focused', 2: 'Obj2-focused'}
    
    for row_idx, (env_name, results) in enumerate(all_results.items()):
        
        # (a) Distance matrix
        ax1 = fig.add_subplot(3, 4, row_idx * 4 + 1)
        
        sort_idx = np.argsort(results['true_labels'])
        sorted_dist = results['distance_matrix'][np.ix_(sort_idx, sort_idx)]
        
        sns.heatmap(sorted_dist, cmap='viridis_r', ax=ax1, cbar_kws={'label': 'SW Dist'})
        ax1.set_title(f'{env_name}\nSW Distance Matrix', fontweight='bold', fontsize=10)
        
        # Add boundaries
        n_per_label = [np.sum(results['true_labels'] == l) for l in sorted(set(results['true_labels']))]
        boundaries = np.cumsum(n_per_label)[:-1]
        for b in boundaries:
            ax1.axhline(y=b, color='white', linewidth=2)
            ax1.axvline(x=b, color='white', linewidth=2)
        
        # (b) PCA of mean gradients - true labels
        ax2 = fig.add_subplot(3, 4, row_idx * 4 + 2)
        
        mean_grads = np.array([np.mean(results['all_gradients'][name], axis=0) 
                               for name in results['instance_names']])
        pca = PCA(n_components=2, random_state=42)
        grad_2d = pca.fit_transform(mean_grads)
        
        for label in sorted(set(results['true_labels'])):
            mask = results['true_labels'] == label
            ax2.scatter(grad_2d[mask, 0], grad_2d[mask, 1],
                       c=variant_colors[label], s=80, alpha=0.7,
                       label=variant_labels[label], edgecolors='black', linewidths=0.5)
        
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title(f'{env_name}\nTrue Labels', fontweight='bold', fontsize=10)
        ax2.legend(fontsize=7)
        
        # (c) PCA - predicted clusters
        ax3 = fig.add_subplot(3, 4, row_idx * 4 + 3)
        
        for cluster_id in np.unique(results['predicted_labels']):
            mask = results['predicted_labels'] == cluster_id
            ax3.scatter(grad_2d[mask, 0], grad_2d[mask, 1],
                       c=[plt.cm.tab10(cluster_id)], s=80, alpha=0.7,
                       label=f'Cluster {cluster_id}', edgecolors='black', linewidths=0.5)
        
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        ax3.set_title(f'{env_name}\nPredicted (Purity: {results["purity"]:.1f}%)', 
                     fontweight='bold', fontsize=10)
        ax3.legend(fontsize=7)
        
        # (d) Pareto outcomes by variant
        ax4 = fig.add_subplot(3, 4, row_idx * 4 + 4)
        
        for label in sorted(set(results['true_labels'])):
            r1_vals = []
            r2_vals = []
            for i, name in enumerate(results['instance_names']):
                if results['true_labels'][i] == label:
                    for info in results['all_infos'][name]:
                        r1_vals.append(info['r1'])
                        r2_vals.append(info['r2'])
            
            ax4.scatter(r1_vals, r2_vals, c=variant_colors[label], s=20, alpha=0.5,
                       label=variant_labels[label])
        
        ax4.set_xlabel('Objective 1')
        ax4.set_ylabel('Objective 2')
        ax4.set_title(f'{env_name}\nPareto Outcomes', fontweight='bold', fontsize=10)
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/intra_domain_variant_clustering.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_dir}/intra_domain_variant_clustering.png")
    
    # Save to paper figures
    paper_dir = 'paper/figures'
    os.makedirs(paper_dir, exist_ok=True)
    plt.savefig(f'{paper_dir}/intra_domain_variant_clustering.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"Saved: {paper_dir}/intra_domain_variant_clustering.png")
    
    # Summary figure
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    env_names = list(all_results.keys())
    x = np.arange(len(env_names))
    width = 0.2
    
    ari_vals = [all_results[e]['ari'] for e in env_names]
    nmi_vals = [all_results[e]['nmi'] for e in env_names]
    purity_vals = [all_results[e]['purity'] / 100 for e in env_names]
    
    bars1 = ax.bar(x - width, ari_vals, width, label='ARI', color='#1f77b4', edgecolor='black')
    bars2 = ax.bar(x, nmi_vals, width, label='NMI', color='#ff7f0e', edgecolor='black')
    bars3 = ax.bar(x + width, purity_vals, width, label='Purity', color='#2ca02c', edgecolor='black')
    
    ax.set_ylabel('Score')
    ax.set_title('Intra-Domain Variant Clustering Performance\n(One Agent per Environment, Multiple Trade-off Variants)', 
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(env_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary_metrics.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir}/summary_metrics.png")
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY: Intra-Domain Variant Clustering")
    print("="*80)
    print(f"\n{'Environment':<25} {'ARI':>10} {'NMI':>10} {'Purity':>10}")
    print("-" * 60)
    for env_name, results in all_results.items():
        print(f"{env_name:<25} {results['ari']:>10.4f} {results['nmi']:>10.4f} "
              f"{results['purity']:>9.1f}%")
    
    return all_results


if __name__ == '__main__':
    all_results = run_all_environments()
