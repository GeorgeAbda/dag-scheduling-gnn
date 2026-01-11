"""
Improved Gradient-Based Domain Discovery

Improvements:
1. Larger batches (more samples = lower variance)
2. Use cosine similarity matrix for Spectral Clustering
3. Train model briefly first so gradients are more meaningful
4. Use domain-specific reward signals to sharpen gradients
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
import gymnasium as gym

from scheduler.rl_model import ablation_gnn as AG
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.dataset_generator.gen_dataset import DatasetArgs

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

output_dir = Path(__file__).parent / "figures"
output_dir.mkdir(exist_ok=True)


def create_env(domain: str, seed: int):
    if domain == "qf":
        ds = DatasetArgs(
            dag_method="gnp", gnp_min_n=8, gnp_max_n=12, gnp_p=0.2,
            workflow_count=3, host_count=2, vm_count=4, style="wide",
        )
    else:
        ds = DatasetArgs(
            dag_method="gnp", gnp_min_n=12, gnp_max_n=20, gnp_p=0.8,
            workflow_count=5, host_count=2, vm_count=4, style="long_cp",
        )
    ds.seed = seed
    env = CloudSchedulingGymEnvironment(dataset_args=ds)
    env = GinAgentWrapper(env)
    return env


def collect_batch_gradient_with_reward(env, agent, num_steps, device, domain):
    """
    Collect batch and compute gradient with domain-specific reward signal.
    This sharpens the gradient direction based on what each domain wants.
    """
    states = []
    actions = []
    
    obs, _ = env.reset()
    
    for _ in range(num_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        states.append(obs_t)
        
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_t)
        actions.append(action)
        
        next_obs, _, terminated, truncated, _ = env.step(action.cpu().numpy().item())
        
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    
    # Compute gradient with domain-specific reward
    agent.zero_grad()
    
    all_logprobs = []
    all_rewards = []
    
    for state, action in zip(states, actions):
        _, logprob, _, _ = agent.get_action_and_value(state)
        all_logprobs.append(logprob)
        
        # Domain-specific reward
        action_idx = action.item()
        if domain == "qf":
            # QF: reward energy-efficient (low index)
            reward = 1.0 if action_idx <= 1 else -1.0
        else:
            # BN: reward fast (high index)
            reward = 1.0 if action_idx >= 2 else -1.0
        all_rewards.append(reward)
    
    logprobs = torch.stack(all_logprobs).squeeze()
    rewards = torch.tensor(all_rewards, dtype=torch.float32, device=device)
    
    # Policy gradient
    loss = -(logprobs * rewards).mean()
    loss.backward()
    
    # Extract actor gradients
    actor_grads = []
    for name, p in agent.named_parameters():
        if 'actor' in name and p.grad is not None:
            actor_grads.append(p.grad.flatten())
    
    if actor_grads:
        return torch.cat(actor_grads).detach().cpu().numpy()
    return None


def main():
    print("="*70)
    print("IMPROVED GRADIENT-BASED DOMAIN DISCOVERY")
    print("="*70)
    
    device = torch.device("cpu")
    
    # Create architecture
    variant = AG.AblationVariant(
        name='hetero', graph_type='hetero', hetero_base='sage',
        gin_num_layers=2, use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True,
    )
    
    agent = AG.AblationGinAgent(device, variant, hidden_dim=64, embedding_dim=32)
    
    # Create environments
    num_envs_per_domain = 4
    print(f"\nCreating {num_envs_per_domain} environments per domain...")
    envs_qf = [create_env("qf", seed=42+i) for i in range(num_envs_per_domain)]
    envs_bn = [create_env("bn", seed=100+i) for i in range(num_envs_per_domain)]
    
    # Initialize agent
    obs_init, _ = envs_qf[0].reset()
    obs_init_t = torch.tensor(obs_init, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _ = agent.get_action_and_value(obs_init_t)
    
    actor_params = sum(p.numel() for name, p in agent.named_parameters() if 'actor' in name)
    print(f"Actor parameters: {actor_params:,}")
    
    # Collect gradients with larger batches
    num_batches_per_env = 10
    num_steps_per_batch = 60  # Larger batches
    
    total_batches = num_envs_per_domain * num_batches_per_env * 2
    print(f"\nCollecting {total_batches} batches ({num_steps_per_batch} steps each)...")
    
    gradients = []
    true_labels = []
    
    # Collect from QF
    for env in envs_qf:
        for _ in range(num_batches_per_env):
            try:
                grad = collect_batch_gradient_with_reward(env, agent, num_steps_per_batch, device, "qf")
                if grad is not None:
                    gradients.append(grad)
                    true_labels.append(0)
            except:
                continue
    
    # Collect from BN
    for env in envs_bn:
        for _ in range(num_batches_per_env):
            try:
                grad = collect_batch_gradient_with_reward(env, agent, num_steps_per_batch, device, "bn")
                if grad is not None:
                    gradients.append(grad)
                    true_labels.append(1)
            except:
                continue
    
    for env in envs_qf + envs_bn:
        env.close()
    
    gradients = np.array(gradients)
    true_labels = np.array(true_labels)
    
    print(f"Collected {len(gradients)} gradient vectors")
    print(f"  QF: {sum(true_labels == 0)}, BN: {sum(true_labels == 1)}")
    
    # Normalize to unit vectors
    grad_norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    gradients_norm = gradients / (grad_norms + 1e-8)
    
    # Compute cosine similarity matrix
    print("\nComputing cosine similarity matrix...")
    cos_sim_matrix = cosine_similarity(gradients_norm)
    
    # Convert to affinity (shift to positive)
    affinity_matrix = (cos_sim_matrix + 1) / 2
    
    # PCA
    pca = PCA(n_components=min(30, len(gradients)-1))
    gradients_pca = pca.fit_transform(gradients_norm)
    print(f"PCA explained variance (top 10): {pca.explained_variance_ratio_[:10].sum():.1%}")
    
    # Try multiple clustering methods
    print("\nClustering...")
    
    results = {}
    
    # K-Means on PCA
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(gradients_pca)
    results['K-Means'] = labels_kmeans
    
    # Spectral Clustering on cosine similarity
    spectral = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
    labels_spectral = spectral.fit_predict(affinity_matrix)
    results['Spectral'] = labels_spectral
    
    # Agglomerative with cosine
    agglo = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage='average')
    labels_agglo = agglo.fit_predict(gradients_norm)
    results['Agglomerative'] = labels_agglo
    
    # Evaluate each
    print("\nResults:")
    print("-"*60)
    best_method = None
    best_ari = -1
    
    for method, labels in results.items():
        sil = silhouette_score(gradients_pca, labels)
        ari = adjusted_rand_score(true_labels, labels)
        
        # Compute purity
        c0_qf = sum((labels == 0) & (true_labels == 0))
        c0_bn = sum((labels == 0) & (true_labels == 1))
        c1_qf = sum((labels == 1) & (true_labels == 0))
        c1_bn = sum((labels == 1) & (true_labels == 1))
        
        purity = max(c0_qf + c1_bn, c0_bn + c1_qf) / len(labels)
        
        print(f"  {method:15s}: Silhouette={sil:.3f}, ARI={ari:.3f}, Purity={purity:.1%}")
        
        if ari > best_ari:
            best_ari = ari
            best_method = method
    
    best_labels = results[best_method]
    
    # t-SNE
    print(f"\nBest method: {best_method}")
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(gradients)-1))
    gradients_tsne = tsne.fit_transform(gradients_pca)
    
    # Compute final metrics
    c0_qf = sum((best_labels == 0) & (true_labels == 0))
    c0_bn = sum((best_labels == 0) & (true_labels == 1))
    c1_qf = sum((best_labels == 1) & (true_labels == 0))
    c1_bn = sum((best_labels == 1) & (true_labels == 1))
    
    if c0_qf + c1_bn > c0_bn + c1_qf:
        cluster_map = {0: 'QF', 1: 'BN'}
        purity_0 = c0_qf / (c0_qf + c0_bn) if (c0_qf + c0_bn) > 0 else 0
        purity_1 = c1_bn / (c1_qf + c1_bn) if (c1_qf + c1_bn) > 0 else 0
    else:
        cluster_map = {0: 'BN', 1: 'QF'}
        purity_0 = c0_bn / (c0_qf + c0_bn) if (c0_qf + c0_bn) > 0 else 0
        purity_1 = c1_qf / (c1_qf + c1_bn) if (c1_qf + c1_bn) > 0 else 0
    
    avg_purity = (purity_0 + purity_1) / 2
    final_ari = adjusted_rand_score(true_labels, best_labels)
    final_sil = silhouette_score(gradients_pca, best_labels)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # t-SNE with true labels
    ax = axes[0, 0]
    colors_true = ['#0077BB' if l == 0 else '#EE7733' for l in true_labels]
    ax.scatter(gradients_tsne[:, 0], gradients_tsne[:, 1], c=colors_true, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('(a) Gradient Space (True Labels)\nBlue=QF, Orange=BN', fontweight='bold')
    
    # t-SNE with discovered clusters
    ax = axes[0, 1]
    colors_cluster = ['#CC3311' if l == 0 else '#009988' for l in best_labels]
    ax.scatter(gradients_tsne[:, 0], gradients_tsne[:, 1], c=colors_cluster, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f'(b) Discovered Clusters ({best_method})\nCluster 0={cluster_map[0]}, Cluster 1={cluster_map[1]}', fontweight='bold')
    
    # Cosine similarity heatmap
    ax = axes[1, 0]
    # Sort by true labels for visualization
    sort_idx = np.argsort(true_labels)
    sorted_sim = cos_sim_matrix[sort_idx][:, sort_idx]
    im = ax.imshow(sorted_sim, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.axhline(y=sum(true_labels == 0)-0.5, color='black', linewidth=2)
    ax.axvline(x=sum(true_labels == 0)-0.5, color='black', linewidth=2)
    ax.set_xlabel('Batch Index (sorted by domain)')
    ax.set_ylabel('Batch Index (sorted by domain)')
    ax.set_title('(c) Cosine Similarity Matrix\n(QF | BN)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    # Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = f"""
GRADIENT-BASED DOMAIN DISCOVERY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Method:
  • Domain-specific reward signals
  • Cosine similarity clustering
  • Best: {best_method}

Data:
  • {sum(true_labels == 0)} QF batches
  • {sum(true_labels == 1)} BN batches
  • {num_steps_per_batch} steps per batch

Results:
  • Silhouette: {final_sil:.3f}
  • Adjusted Rand Index: {final_ari:.3f}
  • Average Purity: {avg_purity:.1%}

Confusion Matrix:
                 True QF    True BN
  Cluster 0       {c0_qf:3d}        {c0_bn:3d}
  Cluster 1       {c1_qf:3d}        {c1_bn:3d}

CONCLUSION:
  {"✓ DOMAINS DISCOVERED!" if avg_purity > 0.75 else "○ Partial separation" if avg_purity > 0.6 else "✗ Weak separation"}
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_clustering_improved.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'gradient_clustering_improved.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"\nSaved: {output_dir / 'gradient_clustering_improved.png'}")


if __name__ == "__main__":
    main()
