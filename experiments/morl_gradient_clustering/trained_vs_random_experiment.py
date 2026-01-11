"""
Compare gradient-based clustering using:
1. Random (untrained) policy
2. Partially trained policy
3. Well-trained policy

This answers: Does training the policy improve clustering?
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mo_gymnasium as mo_gym
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from collections import defaultdict
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


def train_policy(env, policy, n_episodes: int = 100, lr: float = 1e-3, alpha: float = 0.5):
    """
    Train policy using REINFORCE on aggregate reward
    
    Args:
        env: Environment
        policy: Policy network
        n_episodes: Number of training episodes
        lr: Learning rate
        alpha: Trade-off parameter for aggregate reward
    
    Returns:
        Trained policy, training stats
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    episode_returns = []
    
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
        
        # Compute aggregate return
        rewards = np.array(rewards)
        if rewards.shape[1] >= 2:
            r1 = rewards[:, 0].sum()
            r2 = rewards[:, 1].sum()
        else:
            r1 = rewards[:, 0].sum()
            r2 = 0.0
        
        aggregate_return = alpha * r1 + (1 - alpha) * r2
        episode_returns.append(aggregate_return)
        
        # REINFORCE update
        if len(log_probs) > 0:
            policy_loss = -sum(log_probs) * aggregate_return
            
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
    
    return policy, episode_returns


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


def run_clustering_experiment(
    env_configs: List[Dict],
    training_episodes: int,
    alpha_values: List[float],
    n_gradient_episodes: int,
    seed: int = 42
):
    """
    Run clustering with specified training level
    
    Args:
        env_configs: Environment configurations
        training_episodes: 0 for random, >0 for trained
        alpha_values: Trade-off parameters
        n_gradient_episodes: Episodes for gradient estimation
        seed: Random seed
    
    Returns:
        metrics, similarity_matrix, predicted_labels, true_labels
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load environments
    envs = {}
    for config in env_configs:
        name = config['name']
        env = mo_gym.make(config['env_id'])
        obs_dim, n_actions = get_env_dims(env)
        envs[name] = {
            'env': env,
            'obs_dim': obs_dim,
            'n_actions': n_actions,
            'group': config['group']
        }
    
    domain_names = list(envs.keys())
    true_labels = np.array([envs[name]['group'] for name in domain_names])
    
    # Compute gradients for each domain
    domain_gradients = {}
    
    for name, info in envs.items():
        env = info['env']
        obs_dim = info['obs_dim']
        n_actions = info['n_actions']
        
        # Create policy
        policy = DiscretePolicy(obs_dim, n_actions, hidden_dims=[64, 64])
        
        # Train if specified
        if training_episodes > 0:
            policy, _ = train_policy(env, policy, n_episodes=training_episodes, alpha=0.5)
        
        # Compute gradients for each alpha
        grad_vectors = []
        for alpha in alpha_values:
            grad = compute_gradient(env, policy, alpha, n_gradient_episodes)
            grad_vectors.append(grad)
        
        domain_gradients[name] = np.concatenate(grad_vectors)
    
    # Build similarity matrix
    n_domains = len(domain_names)
    similarity_matrix = np.zeros((n_domains, n_domains))
    
    for i, name_i in enumerate(domain_names):
        for j, name_j in enumerate(domain_names):
            grad_i = domain_gradients[name_i]
            grad_j = domain_gradients[name_j]
            
            max_len = max(len(grad_i), len(grad_j))
            grad_i_padded = np.zeros(max_len)
            grad_j_padded = np.zeros(max_len)
            grad_i_padded[:len(grad_i)] = grad_i
            grad_j_padded[:len(grad_j)] = grad_j
            
            norm_i = np.linalg.norm(grad_i_padded)
            norm_j = np.linalg.norm(grad_j_padded)
            
            if norm_i > 1e-8 and norm_j > 1e-8:
                similarity = np.dot(grad_i_padded, grad_j_padded) / (norm_i * norm_j)
            else:
                similarity = 0.0
            
            similarity_matrix[i, j] = similarity
    
    # Spectral clustering
    n_clusters = len(np.unique(true_labels))
    affinity_matrix = (similarity_matrix + 1) / 2
    
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=seed
    )
    predicted_labels = clustering.fit_predict(affinity_matrix)
    
    # Evaluate
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
    
    # Clean up
    for info in envs.values():
        info['env'].close()
    
    return metrics, similarity_matrix, predicted_labels, true_labels


def main():
    """Compare random vs trained policies for clustering"""
    
    print("=" * 80)
    print("RANDOM VS TRAINED POLICY COMPARISON")
    print("=" * 80)
    
    # Environment configurations
    env_configs = [
        {'name': 'dst_0', 'env_id': 'deep-sea-treasure-v0', 'group': 0},
        {'name': 'dst_1', 'env_id': 'deep-sea-treasure-v0', 'group': 0},
        {'name': 'dst_2', 'env_id': 'deep-sea-treasure-v0', 'group': 0},
        {'name': 'fourroom_0', 'env_id': 'four-room-v0', 'group': 1},
        {'name': 'fourroom_1', 'env_id': 'four-room-v0', 'group': 1},
        {'name': 'fourroom_2', 'env_id': 'four-room-v0', 'group': 1},
        {'name': 'minecart_0', 'env_id': 'minecart-deterministic-v0', 'group': 2},
        {'name': 'minecart_1', 'env_id': 'minecart-deterministic-v0', 'group': 2},
        {'name': 'minecart_2', 'env_id': 'minecart-deterministic-v0', 'group': 2},
    ]
    
    alpha_values = np.linspace(0, 1, 11).tolist()
    n_gradient_episodes = 30
    
    # Training levels to compare
    training_levels = [
        (0, "Random Policy"),
        (50, "Lightly Trained (50 eps)"),
        (200, "Moderately Trained (200 eps)"),
        (500, "Well Trained (500 eps)")
    ]
    
    results = {}
    
    for train_eps, label in training_levels:
        print(f"\n{'='*80}")
        print(f"Testing: {label}")
        print(f"{'='*80}")
        
        metrics, sim_matrix, pred_labels, true_labels = run_clustering_experiment(
            env_configs,
            training_episodes=train_eps,
            alpha_values=alpha_values,
            n_gradient_episodes=n_gradient_episodes,
            seed=42
        )
        
        results[label] = {
            'metrics': metrics,
            'similarity_matrix': sim_matrix,
            'predicted_labels': pred_labels,
            'true_labels': true_labels
        }
        
        print(f"\nResults for {label}:")
        print(f"  ARI: {metrics['adjusted_rand_index']:.4f}")
        print(f"  NMI: {metrics['normalized_mutual_info']:.4f}")
        print(f"  Silhouette: {metrics.get('silhouette_score', 'N/A')}")
        print(f"  True labels:      {true_labels}")
        print(f"  Predicted labels: {pred_labels}")
    
    # Create comparison visualization
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON VISUALIZATION")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1-4: Similarity matrices for each training level
    for idx, (train_eps, label) in enumerate(training_levels[:4]):
        if idx < 4:
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            sim_matrix = results[label]['similarity_matrix']
            
            sns.heatmap(
                sim_matrix,
                cmap='RdBu_r',
                center=0,
                ax=ax,
                vmin=-0.2,
                vmax=1.0,
                cbar_kws={'label': 'Similarity'}
            )
            
            ari = results[label]['metrics']['adjusted_rand_index']
            nmi = results[label]['metrics']['normalized_mutual_info']
            ax.set_title(f'{label}\nARI={ari:.3f}, NMI={nmi:.3f}', fontsize=11, fontweight='bold')
            
            # Add group boundaries
            ax.axhline(y=3, color='black', linewidth=2)
            ax.axhline(y=6, color='black', linewidth=2)
            ax.axvline(x=3, color='black', linewidth=2)
            ax.axvline(x=6, color='black', linewidth=2)
    
    # Plot 5: ARI comparison
    ax = axes[1, 2]
    labels = [label for _, label in training_levels]
    ari_values = [results[label]['metrics']['adjusted_rand_index'] for label in labels]
    nmi_values = [results[label]['metrics']['normalized_mutual_info'] for label in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ari_values, width, label='ARI', alpha=0.8, color='blue', edgecolor='black')
    bars2 = ax.bar(x + width/2, nmi_values, width, label='NMI', alpha=0.8, color='green', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels([l.split('(')[0].strip() for l in labels], rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Clustering Quality vs Training Level', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars1, ari_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
                ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, nmi_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/trained_vs_random_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved to results/trained_vs_random_comparison.png")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Training Level':<30} {'ARI':>10} {'NMI':>10} {'Silhouette':>12}")
    print("-" * 65)
    
    for _, label in training_levels:
        m = results[label]['metrics']
        sil = m.get('silhouette_score', 0)
        print(f"{label:<30} {m['adjusted_rand_index']:>10.4f} {m['normalized_mutual_info']:>10.4f} {sil:>12.4f}")
    
    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    best_ari_label = max(labels, key=lambda l: results[l]['metrics']['adjusted_rand_index'])
    best_ari = results[best_ari_label]['metrics']['adjusted_rand_index']
    random_ari = results['Random Policy']['metrics']['adjusted_rand_index']
    
    print(f"\nBest ARI: {best_ari_label} ({best_ari:.4f})")
    print(f"Random ARI: {random_ari:.4f}")
    print(f"Improvement: {(best_ari - random_ari):.4f} ({((best_ari - random_ari) / (random_ari + 1e-8)) * 100:.1f}%)")
    
    if best_ari > random_ari + 0.1:
        print("\n→ Training SIGNIFICANTLY improves clustering")
    elif best_ari > random_ari + 0.05:
        print("\n→ Training provides MODERATE improvement")
    elif best_ari > random_ari:
        print("\n→ Training provides SLIGHT improvement")
    else:
        print("\n→ Random policy is sufficient (or even better)")
    
    return results


if __name__ == '__main__':
    results = main()
