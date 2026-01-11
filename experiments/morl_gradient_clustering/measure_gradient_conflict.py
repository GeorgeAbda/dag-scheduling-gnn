"""
Measure intra-domain gradient conflict in MO-Gymnasium environments.

Key insight: Within the SAME environment, different trajectories can produce
CONFLICTING gradients even with the SAME preference weight α.

This script:
1. Samples multiple trajectories from each environment
2. Computes per-trajectory gradients
3. Measures pairwise cosine similarity between trajectory gradients
4. Computes conflict rate (% of pairs with negative cosine similarity)
"""

import numpy as np
import torch
import torch.nn as nn
import mo_gymnasium as mo_gym
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from collections import defaultdict
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


def collect_single_trajectory_gradient(env, policy, alpha: float, max_steps: int = 200) -> Tuple[np.ndarray, dict]:
    """
    Run ONE trajectory and compute its gradient.
    Returns the gradient vector and trajectory info.
    """
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
        return None, {}
    
    # Compute returns
    total_r1 = sum(rewards_obj1)
    total_r2 = sum(rewards_obj2)
    scalarized_return = alpha * total_r1 + (1 - alpha) * total_r2
    
    # Compute gradient
    policy_loss = -sum(log_probs) * scalarized_return
    
    policy.zero_grad()
    policy_loss.backward()
    
    grad_vector = []
    for param in policy.parameters():
        if param.grad is not None:
            grad_vector.append(param.grad.detach().cpu().numpy().flatten())
    
    if len(grad_vector) == 0:
        return None, {}
    
    grad = np.concatenate(grad_vector)
    
    info = {
        'r1': total_r1,
        'r2': total_r2,
        'scalarized': scalarized_return,
        'steps': step_count,
        'terminated': terminated if 'terminated' in dir() else done
    }
    
    return grad, info


def measure_gradient_conflict(env_id: str, env_name: str, alpha: float = 0.5, 
                              n_trajectories: int = 100, seed: int = 42):
    """
    Measure gradient conflict within a single environment.
    
    Returns:
        - Pairwise cosine similarity matrix
        - Conflict rate (% of pairs with negative similarity)
        - Trajectory info
    """
    print(f"\n{'='*60}")
    print(f"Measuring gradient conflict in {env_name}")
    print(f"Alpha = {alpha}, Trajectories = {n_trajectories}")
    print(f"{'='*60}")
    
    env = mo_gym.make(env_id)
    obs_dim, n_actions = get_env_dims(env)
    
    # Use SAME policy for all trajectories (key point!)
    torch.manual_seed(seed)
    policy = SimplePolicy(obs_dim, n_actions)
    
    gradients = []
    infos = []
    
    for i in range(n_trajectories):
        # Reset random seed for trajectory sampling (but keep same policy)
        np.random.seed(seed + i)
        
        grad, info = collect_single_trajectory_gradient(env, policy, alpha)
        
        if grad is not None:
            gradients.append(grad)
            infos.append(info)
        
        if (i + 1) % 20 == 0:
            print(f"  Collected {i+1}/{n_trajectories} trajectories")
    
    env.close()
    
    if len(gradients) < 2:
        print("  Not enough valid gradients!")
        return None, None, None
    
    gradients = np.array(gradients)
    n = len(gradients)
    
    # Compute pairwise cosine similarities
    print(f"\n  Computing pairwise cosine similarities ({n}x{n} = {n*n} pairs)...")
    
    # Normalize gradients
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1e-8)
    normalized = gradients / norms
    
    # Cosine similarity matrix
    cosine_matrix = normalized @ normalized.T
    
    # Extract upper triangle (excluding diagonal)
    upper_tri_indices = np.triu_indices(n, k=1)
    pairwise_cosines = cosine_matrix[upper_tri_indices]
    
    # Compute statistics
    conflict_rate = np.mean(pairwise_cosines < 0) * 100
    mean_cosine = np.mean(pairwise_cosines)
    std_cosine = np.std(pairwise_cosines)
    
    print(f"\n  Results for {env_name}:")
    print(f"    Conflict rate: {conflict_rate:.1f}%")
    print(f"    Mean cosine: {mean_cosine:.4f}")
    print(f"    Std cosine: {std_cosine:.4f}")
    print(f"    Min cosine: {np.min(pairwise_cosines):.4f}")
    print(f"    Max cosine: {np.max(pairwise_cosines):.4f}")
    
    return cosine_matrix, pairwise_cosines, infos


def run_conflict_analysis():
    """Run gradient conflict analysis on all environments"""
    
    output_dir = 'results/gradient_conflict'
    os.makedirs(output_dir, exist_ok=True)
    
    environments = [
        ('deep-sea-treasure-v0', 'Deep Sea Treasure'),
        ('four-room-v0', 'Four Room'),
        ('minecart-deterministic-v0', 'Minecart'),
    ]
    
    alpha = 0.5  # Balanced preference
    n_trajectories = 100
    
    results = {}
    
    for env_id, env_name in environments:
        cosine_matrix, pairwise_cosines, infos = measure_gradient_conflict(
            env_id, env_name, alpha, n_trajectories
        )
        
        if pairwise_cosines is not None:
            results[env_name] = {
                'cosine_matrix': cosine_matrix,
                'pairwise_cosines': pairwise_cosines,
                'infos': infos,
                'conflict_rate': np.mean(pairwise_cosines < 0) * 100,
                'mean_cosine': np.mean(pairwise_cosines),
                'std_cosine': np.std(pairwise_cosines)
            }
    
    # =========================================================================
    # Generate Figure
    # =========================================================================
    
    print("\n" + "="*60)
    print("Generating visualization...")
    print("="*60)
    
    fig = plt.figure(figsize=(16, 12))
    
    colors = {'Deep Sea Treasure': '#1f77b4', 'Four Room': '#ff7f0e', 'Minecart': '#2ca02c'}
    
    # Row 1: Cosine similarity distributions
    for idx, (env_name, data) in enumerate(results.items()):
        ax = fig.add_subplot(3, 3, idx + 1)
        
        pairwise = data['pairwise_cosines']
        conflict_rate = data['conflict_rate']
        
        # Histogram
        ax.hist(pairwise, bins=50, density=True, alpha=0.7, color=colors[env_name],
                edgecolor='black', linewidth=0.5)
        
        # Vertical line at 0 (conflict threshold)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Conflict threshold')
        
        # Mean line
        ax.axvline(x=data['mean_cosine'], color='black', linestyle='-', linewidth=2,
                  label=f"Mean: {data['mean_cosine']:.3f}")
        
        # Shade conflict region
        ax.axvspan(-1, 0, alpha=0.2, color='red', label=f'Conflict: {conflict_rate:.1f}%')
        
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Density')
        ax.set_title(f'({"abc"[idx]}) {env_name}\nConflict Rate: {conflict_rate:.1f}%', fontweight='bold')
        ax.set_xlim(-1, 1)
        ax.legend(fontsize=7, loc='upper left')
    
    # Row 2: Cosine similarity matrices (subset)
    for idx, (env_name, data) in enumerate(results.items()):
        ax = fig.add_subplot(3, 3, idx + 4)
        
        # Show first 30 trajectories for visibility
        n_show = min(30, data['cosine_matrix'].shape[0])
        matrix_subset = data['cosine_matrix'][:n_show, :n_show]
        
        sns.heatmap(matrix_subset, cmap='RdBu_r', center=0, ax=ax,
                   vmin=-1, vmax=1, cbar_kws={'label': 'Cosine Sim'})
        
        ax.set_title(f'({"def"[idx]}) {env_name}\nPairwise Gradient Similarity', fontweight='bold')
        ax.set_xlabel('Trajectory')
        ax.set_ylabel('Trajectory')
    
    # Row 3: Comparison and explanation
    
    # (g) Bar chart comparison
    ax = fig.add_subplot(3, 3, 7)
    
    env_names = list(results.keys())
    conflict_rates = [results[e]['conflict_rate'] for e in env_names]
    mean_cosines = [results[e]['mean_cosine'] for e in env_names]
    
    x = np.arange(len(env_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, conflict_rates, width, label='Conflict Rate (%)',
                   color=[colors[e] for e in env_names], edgecolor='black')
    
    ax.set_ylabel('Conflict Rate (%)')
    ax.set_title('(g) Gradient Conflict Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace(' ', '\n') for e in env_names], fontsize=9)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.legend(fontsize=8)
    
    for bar, val in zip(bars1, conflict_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # (h) Trajectory outcomes scatter
    ax = fig.add_subplot(3, 3, 8)
    
    for env_name, data in results.items():
        infos = data['infos']
        r1_vals = [info['r1'] for info in infos]
        r2_vals = [info['r2'] for info in infos]
        
        ax.scatter(r1_vals, r2_vals, alpha=0.5, s=30, c=colors[env_name],
                  label=env_name, edgecolors='black', linewidths=0.3)
    
    ax.set_xlabel('Objective 1 Return')
    ax.set_ylabel('Objective 2 Return')
    ax.set_title('(h) Trajectory Outcomes\n(Same α, Different Results)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (i) Summary text
    ax = fig.add_subplot(3, 3, 9)
    ax.axis('off')
    
    summary = f"""INTRA-DOMAIN GRADIENT CONFLICT

Key Finding:
Within the SAME environment, different
trajectories produce CONFLICTING gradients
even with the SAME preference weight α.

Results (α = 0.5):
"""
    
    for env_name, data in results.items():
        summary += f"\n{env_name}:"
        summary += f"\n  Conflict rate: {data['conflict_rate']:.1f}%"
        summary += f"\n  Mean cosine: {data['mean_cosine']:.3f} ± {data['std_cosine']:.3f}"
    
    summary += """

Why This Happens:
• Different trajectories reach different
  Pareto-optimal solutions
• Stochastic policy explores different paths
• Sparse rewards create high variance

Implication:
Mean gradients are unreliable!
→ Use distributional comparison (OT)
→ Sample many trajectories
"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/intra_domain_gradient_conflict.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"\nSaved to {output_dir}/intra_domain_gradient_conflict.png")
    
    # Also save to paper figures
    paper_dir = 'paper/figures'
    os.makedirs(paper_dir, exist_ok=True)
    plt.savefig(f'{paper_dir}/gradient_conflict.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"Saved to {paper_dir}/gradient_conflict.png")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY: Intra-Domain Gradient Conflict")
    print("="*60)
    print(f"\n{'Environment':<25} {'Conflict Rate':>15} {'Mean Cosine':>15}")
    print("-" * 60)
    for env_name, data in results.items():
        print(f"{env_name:<25} {data['conflict_rate']:>14.1f}% {data['mean_cosine']:>15.4f}")
    
    return results


if __name__ == '__main__':
    results = run_conflict_analysis()
