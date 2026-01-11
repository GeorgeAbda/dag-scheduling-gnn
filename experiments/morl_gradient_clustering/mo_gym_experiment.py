"""
Gradient-based domain clustering on real MO-Gymnasium environments

This script tests the clustering method on actual multi-objective RL environments
where reward components shape behavior differently and dynamics are richer.
"""

import numpy as np
import torch
import torch.nn as nn
import mo_gymnasium as mo_gym
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from collections import defaultdict
import copy
import os
import warnings
warnings.filterwarnings('ignore')


class DiscretePolicy(nn.Module):
    """Policy for discrete action spaces"""
    
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


def compute_gradient_for_env(
    env,
    policy: nn.Module,
    alpha: float,
    n_episodes: int = 20
) -> np.ndarray:
    """
    Compute policy gradient for a given environment and alpha
    """
    policy.train()
    all_gradients = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        log_probs = []
        rewards = []
        
        step_count = 0
        max_steps = 200  # Prevent infinite loops
        
        while not done and step_count < max_steps:
            # Flatten observation if needed
            if isinstance(obs, dict):
                obs = np.concatenate([v.flatten() for v in obs.values()])
            obs = np.array(obs).flatten()
            
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob = policy.sample_action(obs_tensor)
            log_probs.append(log_prob)
            
            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            step_count += 1
            
            # Reward is a vector
            if isinstance(reward, np.ndarray):
                rewards.append(reward)
            else:
                rewards.append(np.array([reward, 0.0]))
        
        if len(rewards) == 0:
            continue
            
        # Compute aggregate return
        rewards = np.array(rewards)
        if len(rewards.shape) == 1:
            rewards = rewards.reshape(-1, 1)
        
        # Handle different reward dimensions
        if rewards.shape[1] >= 2:
            r1 = rewards[:, 0].sum()
            r2 = rewards[:, 1].sum()
        else:
            r1 = rewards[:, 0].sum()
            r2 = 0.0
        
        aggregate_return = alpha * r1 + (1 - alpha) * r2
        
        # Compute policy gradient
        if len(log_probs) > 0:
            policy_loss = -sum(log_probs) * aggregate_return
            
            # Zero gradients and backprop
            for param in policy.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            policy_loss.backward()
            
            # Collect gradients
            grad_vector = []
            for param in policy.parameters():
                if param.grad is not None:
                    grad_vector.append(param.grad.detach().cpu().numpy().flatten())
            
            if len(grad_vector) > 0:
                all_gradients.append(np.concatenate(grad_vector))
    
    # Average gradients
    if len(all_gradients) > 0:
        return np.mean(all_gradients, axis=0)
    else:
        return np.zeros(sum(p.numel() for p in policy.parameters()))


def get_env_dims(env):
    """Get observation and action dimensions from environment"""
    obs, _ = env.reset()
    
    # Handle different observation types
    if isinstance(obs, dict):
        obs_dim = sum(np.array(v).flatten().shape[0] for v in obs.values())
    else:
        obs_dim = np.array(obs).flatten().shape[0]
    
    # Get action dimension
    if hasattr(env.action_space, 'n'):
        n_actions = env.action_space.n
    else:
        n_actions = env.action_space.shape[0]
    
    return obs_dim, n_actions


def run_mo_gym_clustering():
    """
    Run gradient-based clustering on MO-Gymnasium environments
    """
    print("=" * 80)
    print("MO-GYMNASIUM GRADIENT CLUSTERING EXPERIMENT")
    print("=" * 80)
    
    # Configuration
    alpha_values = np.linspace(0, 1, 11).tolist()
    n_episodes_per_alpha = 30
    n_policy_inits = 3
    seed = 42
    
    # Define environment variants to cluster
    # We create "domains" by using different environments or same env with different seeds
    env_configs = [
        # Deep Sea Treasure variants (should cluster together)
        {'name': 'dst_seed0', 'env_id': 'deep-sea-treasure-v0', 'seed': 0, 'group': 0},
        {'name': 'dst_seed1', 'env_id': 'deep-sea-treasure-v0', 'seed': 1, 'group': 0},
        {'name': 'dst_seed2', 'env_id': 'deep-sea-treasure-v0', 'seed': 2, 'group': 0},
        
        # Four Room variants (should cluster together)
        {'name': 'fourroom_seed0', 'env_id': 'four-room-v0', 'seed': 0, 'group': 1},
        {'name': 'fourroom_seed1', 'env_id': 'four-room-v0', 'seed': 1, 'group': 1},
        {'name': 'fourroom_seed2', 'env_id': 'four-room-v0', 'seed': 2, 'group': 1},
        
        # Minecart variants (should cluster together)  
        {'name': 'minecart_seed0', 'env_id': 'minecart-deterministic-v0', 'seed': 0, 'group': 2},
        {'name': 'minecart_seed1', 'env_id': 'minecart-deterministic-v0', 'seed': 1, 'group': 2},
        {'name': 'minecart_seed2', 'env_id': 'minecart-deterministic-v0', 'seed': 2, 'group': 2},
    ]
    
    print(f"\n[1] Loading {len(env_configs)} MO-Gymnasium environments...")
    
    # Load environments and get dimensions
    envs = {}
    env_info = {}
    
    for config in env_configs:
        name = config['name']
        env_id = config['env_id']
        
        try:
            env = mo_gym.make(env_id)
            obs_dim, n_actions = get_env_dims(env)
            
            envs[name] = {
                'env': env,
                'obs_dim': obs_dim,
                'n_actions': n_actions,
                'group': config['group'],
                'seed': config['seed']
            }
            print(f"  ✓ {name}: obs_dim={obs_dim}, n_actions={n_actions}")
            
        except Exception as e:
            print(f"  ✗ {name}: Failed to load - {e}")
    
    if len(envs) == 0:
        print("No environments loaded!")
        return
    
    # Ground truth labels
    true_labels = np.array([envs[name]['group'] for name in envs.keys()])
    domain_names = list(envs.keys())
    
    print(f"\nGround truth groups:")
    for group_id in np.unique(true_labels):
        group_envs = [name for name, info in envs.items() if info['group'] == group_id]
        print(f"  Group {group_id}: {group_envs}")
    
    # Compute gradients
    print(f"\n[2] Computing gradients...")
    print(f"  Alpha values: {len(alpha_values)}")
    print(f"  Episodes per alpha: {n_episodes_per_alpha}")
    print(f"  Policy initializations: {n_policy_inits}")
    
    domain_gradients = {}
    
    for init_idx in range(n_policy_inits):
        print(f"\n--- Policy Initialization {init_idx + 1}/{n_policy_inits} ---")
        
        torch.manual_seed(seed + init_idx)
        np.random.seed(seed + init_idx)
        
        # Create shared base policy (use max dimensions)
        max_obs_dim = max(info['obs_dim'] for info in envs.values())
        max_n_actions = max(info['n_actions'] for info in envs.values())
        
        for name, info in envs.items():
            print(f"  Processing: {name}")
            
            env = info['env']
            obs_dim = info['obs_dim']
            n_actions = info['n_actions']
            
            # Create policy for this environment
            policy = DiscretePolicy(obs_dim, n_actions, hidden_dims=[64, 64])
            
            # Compute gradients for each alpha
            grad_vectors = []
            for alpha in alpha_values:
                grad = compute_gradient_for_env(
                    env, policy, alpha, n_episodes_per_alpha
                )
                grad_vectors.append(grad)
            
            # Concatenate gradients
            full_grad = np.concatenate(grad_vectors)
            
            if name not in domain_gradients:
                domain_gradients[name] = []
            domain_gradients[name].append(full_grad)
    
    # Average across initializations
    print("\n[3] Averaging gradients across initializations...")
    for name in domain_gradients:
        domain_gradients[name] = np.mean(domain_gradients[name], axis=0)
        print(f"  {name}: gradient dim = {len(domain_gradients[name])}")
    
    # Build similarity matrix
    print("\n[4] Building similarity matrix...")
    n_domains = len(domain_names)
    similarity_matrix = np.zeros((n_domains, n_domains))
    
    for i, name_i in enumerate(domain_names):
        for j, name_j in enumerate(domain_names):
            grad_i = domain_gradients[name_i]
            grad_j = domain_gradients[name_j]
            
            # Pad to same length if needed
            max_len = max(len(grad_i), len(grad_j))
            grad_i_padded = np.zeros(max_len)
            grad_j_padded = np.zeros(max_len)
            grad_i_padded[:len(grad_i)] = grad_i
            grad_j_padded[:len(grad_j)] = grad_j
            
            # Cosine similarity
            norm_i = np.linalg.norm(grad_i_padded)
            norm_j = np.linalg.norm(grad_j_padded)
            
            if norm_i > 1e-8 and norm_j > 1e-8:
                similarity = np.dot(grad_i_padded, grad_j_padded) / (norm_i * norm_j)
            else:
                similarity = 0.0
            
            similarity_matrix[i, j] = similarity
    
    print(f"  Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
    
    # Apply spectral clustering
    print("\n[5] Applying spectral clustering...")
    n_clusters = len(np.unique(true_labels))
    
    # Convert similarity to affinity
    affinity_matrix = (similarity_matrix + 1) / 2
    
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=seed
    )
    predicted_labels = clustering.fit_predict(affinity_matrix)
    
    print(f"  Predicted labels: {predicted_labels}")
    print(f"  True labels:      {true_labels}")
    
    # Evaluate
    print("\n[6] Evaluating clustering...")
    
    # Distance matrix for silhouette
    distance_matrix = 1 - (similarity_matrix + 1) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    metrics = {
        'adjusted_rand_index': adjusted_rand_score(true_labels, predicted_labels),
        'normalized_mutual_info': normalized_mutual_info_score(true_labels, predicted_labels),
    }
    
    if len(np.unique(predicted_labels)) > 1:
        metrics['silhouette_score'] = silhouette_score(
            distance_matrix, predicted_labels, metric='precomputed'
        )
    
    print("\nClustering Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Cluster summary
    print("\nCluster Summary:")
    cluster_summary = defaultdict(list)
    for name, label in zip(domain_names, predicted_labels):
        cluster_summary[label].append(name)
    
    for cluster_id, names in sorted(cluster_summary.items()):
        print(f"  Cluster {cluster_id}: {names}")
    
    # Visualize
    print("\n[7] Generating visualization...")
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Similarity matrix
    ax = axes[0]
    sns.heatmap(
        similarity_matrix,
        xticklabels=domain_names,
        yticklabels=domain_names,
        cmap='coolwarm',
        center=0,
        ax=ax,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    ax.set_title('Gradient Similarity Matrix', fontsize=14, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 2: True vs Predicted
    ax = axes[1]
    x_pos = np.arange(len(domain_names))
    width = 0.35
    ax.bar(x_pos - width/2, true_labels, width, label='True', alpha=0.8, color='blue')
    ax.bar(x_pos + width/2, predicted_labels, width, label='Predicted', alpha=0.8, color='orange')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.split('_')[0] for n in domain_names], rotation=45, ha='right')
    ax.set_ylabel('Cluster ID')
    ax.set_title('True vs Predicted Labels', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Metrics
    ax = axes[2]
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    colors = ['green' if v > 0.5 else 'orange' if v > 0.2 else 'red' for v in metric_values]
    bars = ax.barh(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlim(-0.2, 1.0)
    ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Good (>0.5)')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.set_title('Clustering Quality Metrics', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Add value labels
    for bar, val in zip(bars, metric_values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig('results/mo_gym_clustering.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("  Saved to results/mo_gym_clustering.png")
    
    # Clean up
    for info in envs.values():
        info['env'].close()
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    
    return metrics, similarity_matrix, predicted_labels, true_labels


if __name__ == '__main__':
    metrics, sim_matrix, pred_labels, true_labels = run_mo_gym_clustering()
