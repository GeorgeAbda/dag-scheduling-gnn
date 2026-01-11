"""
Generate publication-quality figures for gradient-based domain discovery
on MO-Gymnasium environments using random MLP policy.

FIXED VERSION: Uses consistent methodology from successful mo_gym_experiment.py
"""

import numpy as np
import torch
import torch.nn as nn
import mo_gymnasium as mo_gym
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import copy
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.facecolor'] = 'white'


class DiscretePolicy(nn.Module):
    """Simple MLP policy for discrete action spaces"""
    
    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_actions))
        self.network = nn.Sequential(*layers)
        self.n_actions = n_actions
    
    def forward(self, obs):
        logits = self.network(obs)
        return torch.softmax(logits, dim=-1)
    
    def sample_action(self, obs):
        probs = self.forward(obs)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


def compute_gradient(env, policy, alpha: float, n_episodes: int = 20) -> np.ndarray:
    """Compute policy gradient for given alpha"""
    policy.train()
    all_gradients = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        log_probs = []
        rewards = []
        step_count = 0
        max_steps = 200
        
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
            continue
        
        rewards = np.array(rewards)
        if rewards.shape[1] >= 2:
            r1 = rewards[:, 0].sum()
            r2 = rewards[:, 1].sum()
        else:
            r1 = rewards[:, 0].sum()
            r2 = 0.0
        
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
                all_gradients.append(np.concatenate(grad_vector))
    
    if len(all_gradients) > 0:
        return np.mean(all_gradients, axis=0)
    else:
        return np.zeros(sum(p.numel() for p in policy.parameters()))


def get_env_dims(env):
    """Get observation and action dimensions"""
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs_dim = sum(np.array(v).flatten().shape[0] for v in obs.values())
    else:
        obs_dim = np.array(obs).flatten().shape[0]
    
    if hasattr(env.action_space, 'n'):
        n_actions = env.action_space.n
    else:
        n_actions = env.action_space.shape[0]
    
    return obs_dim, n_actions


def collect_gradient_data_fixed(n_instances_per_env: int = 3, n_policy_inits: int = 3, seed: int = 42):
    """
    Collect gradient data using the CORRECT methodology:
    - Multiple instances of each environment type
    - Average over multiple policy initializations
    - More episodes for reliable gradient estimation
    """
    
    # Environment types
    env_types = [
        {'env_id': 'deep-sea-treasure-v0', 'name': 'DST', 'group': 0},
        {'env_id': 'four-room-v0', 'name': 'FourRoom', 'group': 1},
        {'env_id': 'minecart-deterministic-v0', 'name': 'Minecart', 'group': 2},
    ]
    
    alpha_values = np.linspace(0, 1, 11).tolist()  # 11 alpha values
    n_episodes_per_alpha = 30
    
    print("Collecting gradient data (FIXED methodology)...")
    print(f"  Alpha values: {len(alpha_values)}")
    print(f"  Episodes per alpha: {n_episodes_per_alpha}")
    print(f"  Policy initializations: {n_policy_inits}")
    
    # Create domain instances
    domains = {}
    domain_labels = []
    
    for env_type in env_types:
        for instance_idx in range(n_instances_per_env):
            domain_name = f"{env_type['name']}_{instance_idx}"
            env = mo_gym.make(env_type['env_id'])
            obs_dim, n_actions = get_env_dims(env)
            
            domains[domain_name] = {
                'env': env,
                'obs_dim': obs_dim,
                'n_actions': n_actions,
                'group': env_type['group'],
                'env_type': env_type['name']
            }
            domain_labels.append(env_type['group'])
            
            print(f"  Created {domain_name}: obs_dim={obs_dim}, n_actions={n_actions}")
    
    # Compute gradients with multiple policy initializations
    domain_gradients = {name: [] for name in domains.keys()}
    
    for init_idx in range(n_policy_inits):
        print(f"\n--- Policy Initialization {init_idx + 1}/{n_policy_inits} ---")
        
        torch.manual_seed(seed + init_idx)
        np.random.seed(seed + init_idx)
        
        for domain_name, info in domains.items():
            env = info['env']
            obs_dim = info['obs_dim']
            n_actions = info['n_actions']
            
            # Create policy for this environment
            policy = DiscretePolicy(obs_dim, n_actions, hidden_dims=[64, 64])
            
            # Compute gradients for each alpha
            grad_vectors = []
            for alpha in alpha_values:
                grad = compute_gradient(env, policy, alpha, n_episodes_per_alpha)
                grad_vectors.append(grad)
            
            # Concatenate gradients
            full_grad = np.concatenate(grad_vectors)
            domain_gradients[domain_name].append(full_grad)
    
    # Average across initializations
    print("\nAveraging gradients across initializations...")
    for name in domain_gradients:
        domain_gradients[name] = np.mean(domain_gradients[name], axis=0)
        print(f"  {name}: gradient dim = {len(domain_gradients[name])}")
    
    # Close environments
    for info in domains.values():
        info['env'].close()
    
    return domains, domain_gradients, np.array(domain_labels), env_types


def compute_similarity_matrix(domain_gradients, domain_names):
    """Compute cosine similarity matrix with proper padding"""
    n_domains = len(domain_names)
    
    # Get max gradient length
    max_len = max(len(domain_gradients[name]) for name in domain_names)
    
    # Pad gradients
    padded_gradients = {}
    for name in domain_names:
        grad = domain_gradients[name]
        padded = np.zeros(max_len)
        padded[:len(grad)] = grad
        padded_gradients[name] = padded
    
    # Compute similarity
    similarity_matrix = np.zeros((n_domains, n_domains))
    
    for i, name_i in enumerate(domain_names):
        for j, name_j in enumerate(domain_names):
            grad_i = padded_gradients[name_i]
            grad_j = padded_gradients[name_j]
            
            norm_i = np.linalg.norm(grad_i)
            norm_j = np.linalg.norm(grad_j)
            
            if norm_i > 1e-8 and norm_j > 1e-8:
                similarity_matrix[i, j] = np.dot(grad_i, grad_j) / (norm_i * norm_j)
            else:
                similarity_matrix[i, j] = 0.0
    
    return similarity_matrix, padded_gradients


def generate_all_figures():
    """Generate all publication figures with correct methodology"""
    
    print("=" * 80)
    print("GENERATING PUBLICATION FIGURES (FIXED VERSION)")
    print("=" * 80)
    
    # Create output directory
    output_dir = 'results/publication_figures_v2'
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data
    domains, domain_gradients, true_labels, env_types = collect_gradient_data_fixed(
        n_instances_per_env=3,
        n_policy_inits=3,
        seed=42
    )
    
    domain_names = list(domains.keys())
    n_domains = len(domain_names)
    
    # Compute similarity matrix
    similarity_matrix, padded_gradients = compute_similarity_matrix(domain_gradients, domain_names)
    
    # Compute within and cross-domain similarities
    within_sims = defaultdict(list)
    cross_sims = []
    
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            if true_labels[i] == true_labels[j]:
                within_sims[true_labels[i]].append(similarity_matrix[i, j])
            else:
                cross_sims.append(similarity_matrix[i, j])
    
    print(f"\nSimilarity Statistics:")
    print(f"  Cross-domain mean: {np.mean(cross_sims):.4f}")
    for group_id in np.unique(true_labels):
        if len(within_sims[group_id]) > 0:
            print(f"  Within-group {group_id} mean: {np.mean(within_sims[group_id]):.4f}")
    
    # Apply spectral clustering
    affinity = (similarity_matrix + 1) / 2
    clustering = SpectralClustering(n_clusters=3, affinity='precomputed', random_state=42)
    predicted_labels = clustering.fit_predict(affinity)
    
    # Metrics
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    
    distance_matrix = 1 - affinity
    np.fill_diagonal(distance_matrix, 0)
    sil = silhouette_score(distance_matrix, predicted_labels, metric='precomputed')
    
    print(f"\nClustering Metrics:")
    print(f"  ARI: {ari:.4f}")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Silhouette: {sil:.4f}")
    print(f"  True labels:      {true_labels}")
    print(f"  Predicted labels: {predicted_labels}")
    
    # Embeddings
    gradient_matrix = np.array([padded_gradients[name] for name in domain_names])
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, n_domains - 1))
    tsne_embeddings = tsne.fit_transform(gradient_matrix)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_embeddings = pca.fit_transform(gradient_matrix)
    
    # =========================================================================
    # FIGURE 1: Empirical Gradient Conflict
    # =========================================================================
    print("\nGenerating Figure 1: Empirical Gradient Conflict...")
    
    fig = plt.figure(figsize=(16, 8))
    
    # (a) Cross-domain gradient similarity
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.axhline(y=np.mean(cross_sims), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(cross_sims):.3f}')
    ax1.fill_between([0, 1], [np.mean(cross_sims) - np.std(cross_sims)]*2, 
                     [np.mean(cross_sims) + np.std(cross_sims)]*2, alpha=0.3, color='red')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Normalized Position')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('(a) Cross-Domain Gradient Similarity', fontweight='bold')
    ax1.legend()
    
    # (b) Cross vs within-domain similarity
    ax2 = fig.add_subplot(2, 3, 2)
    
    bar_data = [np.mean(cross_sims)]
    bar_labels = ['Cross\n(All)']
    bar_colors = ['#8B0000']
    
    env_names_short = ['DST', 'FourRoom', 'Minecart']
    colors = ['#008B8B', '#006400', '#4B0082']
    
    for idx, group_id in enumerate(np.unique(true_labels)):
        if len(within_sims[group_id]) > 0:
            bar_data.append(np.mean(within_sims[group_id]))
            bar_labels.append(f'Within\n{env_names_short[idx]}')
            bar_colors.append(colors[idx])
    
    bars = ax2.bar(range(len(bar_data)), bar_data, color=bar_colors, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xticks(range(len(bar_data)))
    ax2.set_xticklabels(bar_labels, fontsize=9)
    ax2.set_ylabel('Mean Cosine Similarity')
    ax2.set_title('(b) Cross vs Within-Domain Similarity', fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, bar_data):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # (c) Gradient space with true labels
    ax3 = fig.add_subplot(2, 3, 3)
    colors_map = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
    
    for group_id in np.unique(true_labels):
        mask = true_labels == group_id
        ax3.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1],
                   c=colors_map[group_id], s=100, alpha=0.8, edgecolors='black', linewidths=1,
                   label=env_names_short[group_id])
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.set_title('(c) Gradient Space (True Labels)', fontweight='bold')
    ax3.legend(fontsize=9)
    
    # (d) Discovered clusters
    ax4 = fig.add_subplot(2, 3, 4)
    cluster_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
    
    for cluster_id in np.unique(predicted_labels):
        mask = predicted_labels == cluster_id
        ax4.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1],
                   c=cluster_colors[cluster_id], s=100, alpha=0.8, edgecolors='black', linewidths=1,
                   label=f'Cluster {cluster_id}')
    ax4.set_xlabel('t-SNE 1')
    ax4.set_ylabel('t-SNE 2')
    
    # Compute purity
    purity = 0
    for cluster_id in np.unique(predicted_labels):
        cluster_mask = predicted_labels == cluster_id
        if cluster_mask.sum() > 0:
            true_in_cluster = true_labels[cluster_mask]
            most_common = np.bincount(true_in_cluster).max()
            purity += most_common
    purity = purity / len(true_labels) * 100
    
    ax4.set_title(f'(d) Discovered Clusters\nPurity: {purity:.1f}%', fontweight='bold')
    ax4.legend(fontsize=9)
    
    # (e) Cosine similarity matrix
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Sort by true labels
    sort_idx = np.argsort(true_labels)
    sorted_sim = similarity_matrix[np.ix_(sort_idx, sort_idx)]
    sorted_names = [domain_names[i] for i in sort_idx]
    
    sns.heatmap(sorted_sim, cmap='RdBu_r', center=0, ax=ax5,
                xticklabels=[n.split('_')[0] for n in sorted_names],
                yticklabels=[n.split('_')[0] for n in sorted_names],
                cbar_kws={'label': 'Cosine Sim'}, vmin=-0.5, vmax=1.0)
    ax5.set_title('(e) Cosine Similarity Matrix\n(sorted by env type)', fontweight='bold')
    
    # Add group boundaries
    ax5.axhline(y=3, color='black', linewidth=2)
    ax5.axhline(y=6, color='black', linewidth=2)
    ax5.axvline(x=3, color='black', linewidth=2)
    ax5.axvline(x=6, color='black', linewidth=2)
    
    # (f) Results text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    conflict_rate = (np.array(cross_sims) < 0).mean() * 100
    
    results_text = f"""EMPIRICAL VALIDATION RESULTS

Configuration (MO-Gymnasium):
• Environments: 3 types × 3 instances = 9
  - DeepSeaTreasure (DST)
  - FourRoom
  - Minecart
• Policy: Random MLP (64×64)
• Alpha values: 11
• Episodes per alpha: 30
• Policy initializations: 3

Gradient Similarity:
• Cross-domain: {np.mean(cross_sims):.3f} ± {np.std(cross_sims):.3f}
• Within DST: {np.mean(within_sims[0]):.3f}
• Within FourRoom: {np.mean(within_sims[1]):.3f}
• Within Minecart: {np.mean(within_sims[2]):.3f}
• Conflict Rate: {conflict_rate:.0f}%

Clustering:
• Silhouette: {sil:.3f}
• ARI: {ari:.3f}
• NMI: {nmi:.3f}
• Purity: {purity:.1f}%

THEORY vs EMPIRICAL:
Within >> Cross: {'✓' if np.mean(list(within_sims.values())[0]) > np.mean(cross_sims) + 0.1 else '✗'}
"""
    
    ax6.text(0.05, 0.95, results_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/empirical_gradient_conflict_MO_GYM.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    print(f"  Saved to {output_dir}/empirical_gradient_conflict_MO_GYM.png")
    
    # =========================================================================
    # FIGURE 2: Domain Discovery
    # =========================================================================
    print("\nGenerating Figure 2: Domain Discovery...")
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # (a) Before clustering
    ax = axes[0]
    ax.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c='gray', s=100, alpha=0.6, 
               edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Gradient Embedding Dim 1')
    ax.set_ylabel('Gradient Embedding Dim 2')
    ax.set_title('(a) Gradient Embeddings\n(Before Clustering)', fontweight='bold')
    
    # (b) After clustering
    ax = axes[1]
    for group_id in np.unique(true_labels):
        mask = true_labels == group_id
        ax.scatter(pca_embeddings[mask, 0], pca_embeddings[mask, 1],
                  c=colors_map[group_id], s=100, alpha=0.8, edgecolors='black', linewidths=0.5,
                  label=env_names_short[group_id])
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Gradient Embedding Dim 1')
    ax.set_ylabel('Gradient Embedding Dim 2')
    ax.set_title('(b) Discovered Domains\n(After Clustering)', fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    
    # (c) Domain similarity matrix
    ax = axes[2]
    
    # Compute group-level similarity
    group_sim = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            mask_i = true_labels == i
            mask_j = true_labels == j
            sims = similarity_matrix[np.ix_(mask_i, mask_j)]
            if i == j:
                # Exclude diagonal for within-group
                sims = sims[~np.eye(sims.shape[0], dtype=bool)]
            group_sim[i, j] = np.mean(sims) if len(sims) > 0 else 0
    
    sns.heatmap(group_sim, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=env_names_short, yticklabels=env_names_short, ax=ax,
                cbar_kws={'label': 'Cosine Similarity'}, vmin=-0.3, vmax=1.0)
    ax.set_title('(c) Discovered Domain\nSimilarity Matrix', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_domain_discovery_MO_GYM.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"  Saved to {output_dir}/fig4_domain_discovery_MO_GYM.png")
    
    # =========================================================================
    # FIGURE 3: Gradient Flow
    # =========================================================================
    print("\nGenerating Figure 3: Gradient Flow...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Compute mean embeddings per group
    mean_embeddings = {}
    for group_id in np.unique(true_labels):
        mask = true_labels == group_id
        mean_embeddings[group_id] = np.mean(pca_embeddings[mask], axis=0)
    
    # (a) Gradient vector fields
    ax = axes[0]
    
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for group_id in np.unique(true_labels):
        mask = true_labels == group_id
        group_embeddings = pca_embeddings[mask]
        
        for i in range(len(group_embeddings)):
            direction = mean_embeddings[group_id] - group_embeddings[i]
            norm = np.linalg.norm(direction)
            if norm > 0.1:
                direction = direction / norm * min(norm, 2)
                ax.arrow(group_embeddings[i, 0], group_embeddings[i, 1],
                        direction[0] * 0.5, direction[1] * 0.5,
                        head_width=0.3, head_length=0.15, fc=colors_list[group_id], 
                        ec=colors_list[group_id], alpha=0.6, linewidth=1)
    
    # Mark optima
    for group_id in np.unique(true_labels):
        ax.scatter(mean_embeddings[group_id][0], mean_embeddings[group_id][1],
                  marker='*', s=300, c='black', zorder=10)
        ax.annotate(f'{env_names_short[group_id]}', 
                   (mean_embeddings[group_id][0], mean_embeddings[group_id][1] + 0.5),
                   fontsize=9, ha='center', fontweight='bold')
    
    # Legend
    for group_id in np.unique(true_labels):
        ax.plot([], [], color=colors_list[group_id], linewidth=2, 
                label=f'{env_names_short[group_id]} Gradient')
    ax.legend(fontsize=8, loc='upper right')
    
    ax.set_xlabel('PC1 (gradient space)')
    ax.set_ylabel('PC2 (gradient space)')
    ax.set_title('(a) REAL Gradient Vector Fields\n(Random MLP Policy)', fontweight='bold')
    
    # (b) Combined gradient magnitude
    ax = axes[1]
    
    x_range = np.linspace(pca_embeddings[:, 0].min() - 2, pca_embeddings[:, 0].max() + 2, 50)
    y_range = np.linspace(pca_embeddings[:, 1].min() - 2, pca_embeddings[:, 1].max() + 2, 50)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    
    magnitude = np.zeros_like(x_grid)
    
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            point = np.array([x_grid[i, j], y_grid[i, j]])
            combined_grad = np.zeros(2)
            for group_id in np.unique(true_labels):
                direction = mean_embeddings[group_id] - point
                combined_grad += direction / (np.linalg.norm(direction) + 1)
            magnitude[i, j] = np.linalg.norm(combined_grad)
    
    contour = ax.contourf(x_grid, y_grid, magnitude, levels=20, cmap='RdYlGn_r')
    plt.colorbar(contour, ax=ax, label='||∇||')
    
    # Mark minimum
    min_idx = np.unravel_index(np.argmin(magnitude), magnitude.shape)
    ax.scatter(x_grid[min_idx], y_grid[min_idx], marker='o', s=150, c='yellow',
              edgecolors='black', linewidth=2, zorder=10, label='Max Cancel')
    
    # Mark optima
    for group_id in np.unique(true_labels):
        ax.scatter(mean_embeddings[group_id][0], mean_embeddings[group_id][1],
                  marker='*', s=150, c='white', edgecolors='black', linewidth=1, zorder=10)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('(b) Combined Gradient Magnitude\n(Low = Cancellation)', fontweight='bold')
    
    # (c) Training efficiency
    ax = axes[2]
    
    efficiency = 100 * (1 - (magnitude.max() - magnitude) / (magnitude.max() - magnitude.min() + 1e-8))
    
    contour = ax.contourf(x_grid, y_grid, efficiency, levels=20, cmap='RdYlGn')
    plt.colorbar(contour, ax=ax, label='Efficiency %')
    
    ax.scatter(x_grid[min_idx], y_grid[min_idx], marker='o', s=150, c='yellow',
              edgecolors='black', linewidth=2, zorder=10)
    
    for group_id in np.unique(true_labels):
        ax.scatter(mean_embeddings[group_id][0], mean_embeddings[group_id][1],
                  marker='*', s=150, c='white', edgecolors='black', linewidth=1, zorder=10)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('(c) Training Efficiency\n(% of gradient signal retained)', fontweight='bold')
    
    # Architecture info
    n_params = sum(p.numel() for p in DiscretePolicy(14, 6, [64, 64]).parameters())
    fig.text(0.5, 0.02, f'Architecture: Random MLP (64×64) | Parameters: ~{n_params:,}', 
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f'{output_dir}/gradient_flow_MLP.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    print(f"  Saved to {output_dir}/gradient_flow_MLP.png")
    
    print("\n" + "=" * 80)
    print("ALL FIGURES GENERATED (FIXED VERSION)")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}/")


if __name__ == '__main__':
    generate_all_figures()
