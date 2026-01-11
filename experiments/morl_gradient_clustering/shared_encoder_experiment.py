"""
Shared Encoder Approach for Fair Gradient Comparison

Key idea: Map all observations to a common latent space using environment-specific
encoders, then use a SHARED policy head. This ensures gradients are comparable
across environments with different observation/action spaces.

Architecture:
    obs -> [Env-Specific Encoder] -> latent (fixed dim) -> [Shared Policy Head] -> action logits

The gradients of the SHARED policy head are comparable across all environments.
"""

import numpy as np
import torch
import torch.nn as nn
import mo_gymnasium as mo_gym
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

# Style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'

# Common latent dimension for all environments
LATENT_DIM = 64
MAX_ACTIONS = 6  # Maximum action space size across environments


class EnvironmentEncoder(nn.Module):
    """Environment-specific encoder that maps observations to common latent space"""
    
    def __init__(self, obs_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
    
    def forward(self, obs):
        return self.encoder(obs)


class SharedPolicyHead(nn.Module):
    """Shared policy head that operates on latent representations"""
    
    def __init__(self, latent_dim: int = LATENT_DIM, max_actions: int = MAX_ACTIONS):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_actions)
        )
        self.max_actions = max_actions
    
    def forward(self, latent, n_actions: int):
        logits = self.policy_head(latent)
        # Mask invalid actions
        if n_actions < self.max_actions:
            logits[:, n_actions:] = float('-inf')
        return torch.softmax(logits[:, :n_actions], dim=-1)


class SharedEncoderPolicy(nn.Module):
    """
    Full policy with environment-specific encoder and shared policy head.
    
    Only the shared policy head gradients are used for comparison.
    """
    
    def __init__(self, obs_dim: int, n_actions: int, 
                 latent_dim: int = LATENT_DIM, 
                 shared_head: SharedPolicyHead = None):
        super().__init__()
        self.encoder = EnvironmentEncoder(obs_dim, latent_dim)
        self.shared_head = shared_head if shared_head else SharedPolicyHead(latent_dim)
        self.n_actions = n_actions
    
    def forward(self, obs):
        latent = self.encoder(obs)
        return self.shared_head(latent, self.n_actions)
    
    def sample_action(self, obs):
        probs = self.forward(obs)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)
    
    def get_shared_head_gradient(self) -> np.ndarray:
        """Extract gradient of shared policy head only"""
        grad_vector = []
        for param in self.shared_head.parameters():
            if param.grad is not None:
                grad_vector.append(param.grad.detach().cpu().numpy().flatten())
        if grad_vector:
            return np.concatenate(grad_vector)
        return None


def get_env_dims(env):
    """Get environment dimensions"""
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs_dim = sum(np.array(v).flatten().shape[0] for v in obs.values())
    else:
        obs_dim = np.array(obs).flatten().shape[0]
    n_actions = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    return obs_dim, n_actions


def compute_shared_head_gradient(env, policy, alpha: float, n_episodes: int = 10) -> np.ndarray:
    """
    Compute gradient of the SHARED policy head for a given environment and alpha.
    
    This ensures gradients are comparable across environments.
    """
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
        r1 = rewards[:, 0].sum()
        r2 = rewards[:, 1].sum() if rewards.shape[1] >= 2 else 0.0
        aggregate_return = alpha * r1 + (1 - alpha) * r2
        
        if len(log_probs) > 0:
            policy_loss = -sum(log_probs) * aggregate_return
            
            # Zero all gradients
            policy.zero_grad()
            policy_loss.backward()
            
            # Extract only shared head gradient
            grad = policy.get_shared_head_gradient()
            if grad is not None:
                all_gradients.append(grad)
    
    if len(all_gradients) > 0:
        return np.mean(all_gradients, axis=0)
    return None


def sliced_wasserstein_distance(X, Y, n_projections=100, seed=42):
    """Compute Sliced Wasserstein distance"""
    d = X.shape[1]
    np.random.seed(seed)
    projections = np.random.randn(n_projections, d)
    projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)
    
    sw_distances = []
    for proj in projections:
        X_proj = X @ proj
        Y_proj = Y @ proj
        X_sorted = np.sort(X_proj)
        Y_sorted = np.sort(Y_proj)
        
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


def collect_gradient_distributions_shared_encoder(n_samples_per_domain=50, seed=42):
    """
    Collect gradient distributions using shared encoder architecture.
    
    Key: All gradients come from the SAME shared policy head, making them comparable.
    """
    
    env_types = [
        {'env_id': 'deep-sea-treasure-v0', 'name': 'DST', 'group': 0},
        {'env_id': 'four-room-v0', 'name': 'FourRoom', 'group': 1},
        {'env_id': 'minecart-deterministic-v0', 'name': 'Minecart', 'group': 2},
    ]
    
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_instances = 10  # More instances per env type for robustness testing
    
    domains = {}
    gradient_distributions = {}
    true_labels = []
    
    print("=" * 80)
    print("SHARED ENCODER GRADIENT COLLECTION")
    print("=" * 80)
    print(f"Latent dimension: {LATENT_DIM}")
    print(f"Max actions: {MAX_ACTIONS}")
    print(f"Samples per domain: {n_samples_per_domain}")
    print(f"Alpha values: {alpha_values}")
    print()
    
    for env_type in env_types:
        for instance_idx in range(n_instances):
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
            true_labels.append(env_type['group'])
            
            # Collect gradients with multiple policy initializations and average
            gradients = []
            n_policy_inits = 5  # Average over multiple initializations
            samples_per_alpha = n_samples_per_domain // len(alpha_values)
            
            for init_idx in range(n_policy_inits):
                # Different seed per initialization
                init_seed = seed + env_type['group'] * 1000 + instance_idx * 100 + init_idx
                torch.manual_seed(init_seed)
                np.random.seed(init_seed)
                
                # Create SHARED policy head for this initialization
                shared_head = SharedPolicyHead(LATENT_DIM, MAX_ACTIONS)
                
                for alpha in alpha_values:
                    # Create new encoder but REUSE shared head
                    policy = SharedEncoderPolicy(obs_dim, n_actions, LATENT_DIM, shared_head)
                    
                    for _ in range(samples_per_alpha // n_policy_inits):
                        grad = compute_shared_head_gradient(env, policy, alpha, n_episodes=10)
                        if grad is not None:
                            gradients.append(grad)
            
            gradient_distributions[domain_name] = np.array(gradients)
            print(f"  {domain_name}: {len(gradients)} samples, grad_dim={gradients[0].shape[0] if gradients else 0}")
    
    # Verify all gradients have same dimension
    dims = [g.shape[1] for g in gradient_distributions.values()]
    print(f"\nGradient dimensions: {set(dims)} (should be single value)")
    
    return domains, gradient_distributions, np.array(true_labels), env_types


def compute_cosine_similarity_matrix(gradient_distributions, domain_names):
    """Compute cosine similarity matrix using mean gradients"""
    n = len(domain_names)
    
    # Compute mean gradients
    mean_grads = {name: np.mean(grads, axis=0) for name, grads in gradient_distributions.items()}
    
    similarity_matrix = np.zeros((n, n))
    for i, name_i in enumerate(domain_names):
        for j, name_j in enumerate(domain_names):
            g_i = mean_grads[name_i]
            g_j = mean_grads[name_j]
            norm_i = np.linalg.norm(g_i)
            norm_j = np.linalg.norm(g_j)
            if norm_i > 1e-8 and norm_j > 1e-8:
                similarity_matrix[i, j] = np.dot(g_i, g_j) / (norm_i * norm_j)
    
    return similarity_matrix


def run_shared_encoder_experiment():
    """Run the full experiment with shared encoder - comparing SW and Cosine"""
    
    output_dir = 'results/shared_encoder'
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data
    domains, gradient_distributions, true_labels, env_types = \
        collect_gradient_distributions_shared_encoder(n_samples_per_domain=50, seed=42)
    
    domain_names = list(gradient_distributions.keys())
    n_domains = len(domain_names)
    
    # =========================================================================
    # METHOD 1: Sliced Wasserstein
    # =========================================================================
    print("\n" + "="*80)
    print("METHOD 1: SLICED WASSERSTEIN")
    print("="*80)
    print("\nComputing Sliced Wasserstein distances...")
    distance_matrix = np.zeros((n_domains, n_domains))
    
    for i, name_i in enumerate(domain_names):
        for j, name_j in enumerate(domain_names):
            if i < j:
                dist = sliced_wasserstein_distance(
                    gradient_distributions[name_i],
                    gradient_distributions[name_j]
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                print(f"  {name_i} <-> {name_j}: {dist:.4f}")
    
    # Convert to similarity
    max_dist = distance_matrix.max()
    similarity_matrix = 1 - distance_matrix / (max_dist + 1e-8)
    
    # Spectral clustering
    clustering = SpectralClustering(n_clusters=3, affinity='precomputed', random_state=42)
    predicted_labels = clustering.fit_predict(similarity_matrix)
    
    # Metrics
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    sil = silhouette_score(distance_matrix, predicted_labels, metric='precomputed')
    
    # Compute within/cross domain distances
    within_dists = defaultdict(list)
    cross_dists = []
    
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            if true_labels[i] == true_labels[j]:
                within_dists[true_labels[i]].append(distance_matrix[i, j])
            else:
                cross_dists.append(distance_matrix[i, j])
    
    print(f"\n{'='*80}")
    print("RESULTS (SHARED ENCODER)")
    print(f"{'='*80}")
    print(f"\nClustering Metrics:")
    print(f"  ARI: {ari:.4f}")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Silhouette: {sil:.4f}")
    print(f"\nDistance Statistics:")
    print(f"  Cross-domain mean: {np.mean(cross_dists):.4f}")
    for g in np.unique(true_labels):
        if len(within_dists[g]) > 0:
            print(f"  Within {['DST', 'FourRoom', 'Minecart'][g]}: {np.mean(within_dists[g]):.4f}")
    print(f"\nLabels:")
    print(f"  True:      {true_labels}")
    print(f"  Predicted: {predicted_labels}")
    
    # Purity
    purity = 0
    for cluster_id in np.unique(predicted_labels):
        cluster_mask = predicted_labels == cluster_id
        if cluster_mask.sum() > 0:
            true_in_cluster = true_labels[cluster_mask]
            most_common = np.bincount(true_in_cluster).max()
            purity += most_common
    purity = purity / len(true_labels) * 100
    print(f"  Purity: {purity:.1f}%")
    
    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================
    
    # Mean gradients for visualization
    mean_grads = np.array([np.mean(gradient_distributions[name], axis=0) for name in domain_names])
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, n_domains - 1))
    tsne_embeddings = tsne.fit_transform(mean_grads)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_embeddings = pca.fit_transform(mean_grads)
    
    env_names_short = ['DST', 'FourRoom', 'Minecart']
    colors_map = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
    
    # Figure 1: Main results
    fig = plt.figure(figsize=(16, 8))
    
    # (a) Cross-domain distance
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.axhline(y=np.mean(cross_dists), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(cross_dists):.4f}')
    ax1.fill_between([0, 1], [np.mean(cross_dists) - np.std(cross_dists)]*2,
                     [np.mean(cross_dists) + np.std(cross_dists)]*2, alpha=0.3, color='red')
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Normalized Position')
    ax1.set_ylabel('SW Distance')
    ax1.set_title('(a) Cross-Domain SW Distance', fontweight='bold')
    ax1.legend()
    
    # (b) Within vs Cross
    ax2 = fig.add_subplot(2, 3, 2)
    
    bar_data = [np.mean(cross_dists)]
    bar_labels = ['Cross\n(All)']
    bar_colors = ['#8B0000']
    colors_bar = ['#008B8B', '#006400', '#4B0082']
    
    for idx, group_id in enumerate(np.unique(true_labels)):
        if len(within_dists[group_id]) > 0:
            bar_data.append(np.mean(within_dists[group_id]))
            bar_labels.append(f'Within\n{env_names_short[idx]}')
            bar_colors.append(colors_bar[idx])
    
    bars = ax2.bar(range(len(bar_data)), bar_data, color=bar_colors, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(bar_data)))
    ax2.set_xticklabels(bar_labels, fontsize=9)
    ax2.set_ylabel('Mean SW Distance')
    ax2.set_title('(b) Cross vs Within-Domain Distance', fontweight='bold')
    
    for bar, val in zip(bars, bar_data):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # (c) Gradient space (true labels)
    ax3 = fig.add_subplot(2, 3, 3)
    for group_id in np.unique(true_labels):
        mask = true_labels == group_id
        ax3.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1],
                   c=colors_map[group_id], s=150, alpha=0.8, edgecolors='black', linewidths=1.5,
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
                   c=cluster_colors[cluster_id], s=150, alpha=0.8, edgecolors='black', linewidths=1.5,
                   label=f'Cluster {cluster_id}')
    ax4.set_xlabel('t-SNE 1')
    ax4.set_ylabel('t-SNE 2')
    ax4.set_title(f'(d) Discovered Clusters\nPurity: {purity:.1f}%', fontweight='bold')
    ax4.legend(fontsize=9)
    
    # (e) Distance matrix
    ax5 = fig.add_subplot(2, 3, 5)
    
    sort_idx = np.argsort(true_labels)
    sorted_dist = distance_matrix[np.ix_(sort_idx, sort_idx)]
    sorted_names = [domain_names[i].split('_')[0] for i in sort_idx]
    
    sns.heatmap(sorted_dist, cmap='viridis_r', ax=ax5,
                xticklabels=sorted_names, yticklabels=sorted_names,
                cbar_kws={'label': 'SW Distance'}, annot=True, fmt='.3f', annot_kws={'size': 7})
    ax5.set_title('(e) SW Distance Matrix\n(Shared Encoder)', fontweight='bold')
    
    ax5.axhline(y=3, color='white', linewidth=2)
    ax5.axhline(y=6, color='white', linewidth=2)
    ax5.axvline(x=3, color='white', linewidth=2)
    ax5.axvline(x=6, color='white', linewidth=2)
    
    # (f) Results summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Compute separation ratio
    within_mean = np.mean([np.mean(v) for v in within_dists.values() if len(v) > 0])
    sep_ratio = np.mean(cross_dists) / (within_mean + 1e-8)
    
    results_text = f"""SHARED ENCODER RESULTS

Architecture:
• Env-specific encoder → Latent ({LATENT_DIM}D)
• Shared policy head (compared)
• Max actions: {MAX_ACTIONS}

Distance Statistics:
• Cross-domain: {np.mean(cross_dists):.4f} ± {np.std(cross_dists):.4f}
• Within DST: {np.mean(within_dists[0]):.4f}
• Within FourRoom: {np.mean(within_dists[1]):.4f}
• Within Minecart: {np.mean(within_dists[2]):.4f}
• Separation ratio: {sep_ratio:.2f}x

Clustering (Spectral + SW):
• ARI: {ari:.4f}
• NMI: {nmi:.4f}
• Silhouette: {sil:.4f}
• Purity: {purity:.1f}%

Key: Gradients from SHARED head
are directly comparable!
"""
    
    ax6.text(0.05, 0.95, results_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shared_encoder_results.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"\nSaved to {output_dir}/shared_encoder_results.png")
    
    # Also save to publication_figures
    pub_dir = 'results/publication_figures'
    os.makedirs(pub_dir, exist_ok=True)
    plt.savefig(f'{pub_dir}/empirical_gradient_conflict_MO_GYM.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"Saved to {pub_dir}/empirical_gradient_conflict_MO_GYM.png")
    
    # =========================================================================
    # METHOD 2: COSINE SIMILARITY
    # =========================================================================
    print("\n" + "="*80)
    print("METHOD 2: COSINE SIMILARITY")
    print("="*80)
    
    cosine_sim_matrix = compute_cosine_similarity_matrix(gradient_distributions, domain_names)
    
    # Convert to affinity for clustering
    cosine_affinity = (cosine_sim_matrix + 1) / 2  # Map [-1,1] to [0,1]
    
    # Spectral clustering
    cosine_clustering = SpectralClustering(n_clusters=3, affinity='precomputed', random_state=42)
    cosine_predicted = cosine_clustering.fit_predict(cosine_affinity)
    
    # Metrics
    cosine_ari = adjusted_rand_score(true_labels, cosine_predicted)
    cosine_nmi = normalized_mutual_info_score(true_labels, cosine_predicted)
    
    # Convert similarity to distance for silhouette
    cosine_dist = 1 - cosine_sim_matrix
    np.fill_diagonal(cosine_dist, 0)
    cosine_sil = silhouette_score(cosine_dist, cosine_predicted, metric='precomputed')
    
    # Compute within/cross domain similarities
    cosine_within = defaultdict(list)
    cosine_cross = []
    
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            if true_labels[i] == true_labels[j]:
                cosine_within[true_labels[i]].append(cosine_sim_matrix[i, j])
            else:
                cosine_cross.append(cosine_sim_matrix[i, j])
    
    # Purity for cosine
    cosine_purity = 0
    for cluster_id in np.unique(cosine_predicted):
        cluster_mask = cosine_predicted == cluster_id
        if cluster_mask.sum() > 0:
            true_in_cluster = true_labels[cluster_mask]
            most_common = np.bincount(true_in_cluster).max()
            cosine_purity += most_common
    cosine_purity = cosine_purity / len(true_labels) * 100
    
    print(f"\nClustering Metrics (Cosine):")
    print(f"  ARI: {cosine_ari:.4f}")
    print(f"  NMI: {cosine_nmi:.4f}")
    print(f"  Silhouette: {cosine_sil:.4f}")
    print(f"  Purity: {cosine_purity:.1f}%")
    print(f"\nSimilarity Statistics:")
    print(f"  Cross-domain mean: {np.mean(cosine_cross):.4f}")
    for g in np.unique(true_labels):
        if len(cosine_within[g]) > 0:
            print(f"  Within {['DST', 'FourRoom', 'Minecart'][g]}: {np.mean(cosine_within[g]):.4f}")
    print(f"\nLabels (Cosine):")
    print(f"  True:      {true_labels}")
    print(f"  Predicted: {cosine_predicted}")
    
    # =========================================================================
    # COMPARISON FIGURE
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING COMPARISON FIGURE")
    print("="*80)
    
    fig = plt.figure(figsize=(18, 10))
    
    # Row 1: Sliced Wasserstein
    # (a) SW Distance Matrix
    ax1 = fig.add_subplot(2, 4, 1)
    sort_idx = np.argsort(true_labels)
    sorted_dist = distance_matrix[np.ix_(sort_idx, sort_idx)]
    n_per_group = n_domains // 3
    
    sns.heatmap(sorted_dist, cmap='viridis_r', ax=ax1, cbar_kws={'label': 'SW Distance'})
    ax1.set_title('(a) SW Distance Matrix', fontweight='bold')
    ax1.axhline(y=n_per_group, color='white', linewidth=2)
    ax1.axhline(y=2*n_per_group, color='white', linewidth=2)
    ax1.axvline(x=n_per_group, color='white', linewidth=2)
    ax1.axvline(x=2*n_per_group, color='white', linewidth=2)
    ax1.set_xlabel('Domain')
    ax1.set_ylabel('Domain')
    
    # (b) SW Clustering
    ax2 = fig.add_subplot(2, 4, 2)
    for cluster_id in np.unique(predicted_labels):
        mask = predicted_labels == cluster_id
        ax2.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1],
                   c=cluster_colors[cluster_id], s=100, alpha=0.8, edgecolors='black',
                   label=f'Cluster {cluster_id}')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title(f'(b) SW Clustering\nPurity: {purity:.1f}%', fontweight='bold')
    ax2.legend(fontsize=8)
    
    # Row 2: Cosine Similarity
    # (c) Cosine Similarity Matrix
    ax3 = fig.add_subplot(2, 4, 5)
    sorted_cosine = cosine_sim_matrix[np.ix_(sort_idx, sort_idx)]
    
    sns.heatmap(sorted_cosine, cmap='RdBu_r', center=0, ax=ax3, 
                cbar_kws={'label': 'Cosine Sim'}, vmin=-0.5, vmax=1.0)
    ax3.set_title('(c) Cosine Similarity Matrix', fontweight='bold')
    ax3.axhline(y=n_per_group, color='black', linewidth=2)
    ax3.axhline(y=2*n_per_group, color='black', linewidth=2)
    ax3.axvline(x=n_per_group, color='black', linewidth=2)
    ax3.axvline(x=2*n_per_group, color='black', linewidth=2)
    ax3.set_xlabel('Domain')
    ax3.set_ylabel('Domain')
    
    # (d) Cosine Clustering
    ax4 = fig.add_subplot(2, 4, 6)
    for cluster_id in np.unique(cosine_predicted):
        mask = cosine_predicted == cluster_id
        ax4.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1],
                   c=cluster_colors[cluster_id], s=100, alpha=0.8, edgecolors='black',
                   label=f'Cluster {cluster_id}')
    ax4.set_xlabel('t-SNE 1')
    ax4.set_ylabel('t-SNE 2')
    ax4.set_title(f'(d) Cosine Clustering\nPurity: {cosine_purity:.1f}%', fontweight='bold')
    ax4.legend(fontsize=8)
    
    # (e) True Labels
    ax5 = fig.add_subplot(2, 4, 3)
    for group_id in np.unique(true_labels):
        mask = true_labels == group_id
        ax5.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1],
                   c=colors_map[group_id], s=100, alpha=0.8, edgecolors='black',
                   label=env_names_short[group_id])
    ax5.set_xlabel('t-SNE 1')
    ax5.set_ylabel('t-SNE 2')
    ax5.set_title('(e) True Labels', fontweight='bold')
    ax5.legend(fontsize=8)
    
    # (f) Comparison Bar Chart
    ax6 = fig.add_subplot(2, 4, 4)
    
    methods = ['Sliced\nWasserstein', 'Cosine\nSimilarity']
    ari_vals = [ari, cosine_ari]
    nmi_vals = [nmi, cosine_nmi]
    purity_vals = [purity/100, cosine_purity/100]
    
    x = np.arange(len(methods))
    width = 0.25
    
    bars1 = ax6.bar(x - width, ari_vals, width, label='ARI', color='#1f77b4', edgecolor='black')
    bars2 = ax6.bar(x, nmi_vals, width, label='NMI', color='#ff7f0e', edgecolor='black')
    bars3 = ax6.bar(x + width, purity_vals, width, label='Purity', color='#2ca02c', edgecolor='black')
    
    ax6.set_ylabel('Score')
    ax6.set_title('(f) Method Comparison', fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(methods)
    ax6.legend(fontsize=8)
    ax6.set_ylim(0, 1.1)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    
    # (g) Within/Cross Statistics
    ax7 = fig.add_subplot(2, 4, 7)
    
    # SW statistics
    sw_data = [np.mean(cross_dists), np.mean(within_dists[0]), 
               np.mean(within_dists[1]), np.mean(within_dists[2])]
    
    x = np.arange(4)
    bars = ax7.bar(x, sw_data, color=['#8B0000', '#008B8B', '#006400', '#4B0082'], edgecolor='black')
    ax7.set_xticks(x)
    ax7.set_xticklabels(['Cross', 'DST', 'FourRoom', 'Minecart'], fontsize=8)
    ax7.set_ylabel('SW Distance')
    ax7.set_title('(g) SW: Cross vs Within', fontweight='bold')
    
    for bar, val in zip(bars, sw_data):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    
    # (h) Summary
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    
    summary_text = f"""COMPARISON SUMMARY
    
Configuration:
• Domains: {n_domains} ({n_per_group} per env type)
• Shared encoder: {LATENT_DIM}D latent
• Gradient samples: 50 per domain

SLICED WASSERSTEIN:
• ARI: {ari:.4f}
• NMI: {nmi:.4f}
• Purity: {purity:.1f}%
• Cross dist: {np.mean(cross_dists):.4f}
• Within DST: {np.mean(within_dists[0]):.4f}

COSINE SIMILARITY:
• ARI: {cosine_ari:.4f}
• NMI: {cosine_nmi:.4f}
• Purity: {cosine_purity:.1f}%
• Cross sim: {np.mean(cosine_cross):.4f}
• Within DST: {np.mean(cosine_within[0]):.4f}

WINNER: {'SW' if ari > cosine_ari else 'Cosine' if cosine_ari > ari else 'Tie'}
"""
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sw_vs_cosine_comparison.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"\nSaved to {output_dir}/sw_vs_cosine_comparison.png")
    
    # Clean up
    for info in domains.values():
        info['env'].close()
    
    return {
        'sw': {'ari': ari, 'nmi': nmi, 'purity': purity, 'silhouette': sil},
        'cosine': {'ari': cosine_ari, 'nmi': cosine_nmi, 'purity': cosine_purity, 'silhouette': cosine_sil}
    }


if __name__ == '__main__':
    results = run_shared_encoder_experiment()
    
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"\n{'Method':<20} {'ARI':>10} {'NMI':>10} {'Purity':>10}")
    print("-" * 55)
    print(f"{'Sliced Wasserstein':<20} {results['sw']['ari']:>10.4f} {results['sw']['nmi']:>10.4f} {results['sw']['purity']:>10.1f}%")
    print(f"{'Cosine Similarity':<20} {results['cosine']['ari']:>10.4f} {results['cosine']['nmi']:>10.4f} {results['cosine']['purity']:>10.1f}%")
