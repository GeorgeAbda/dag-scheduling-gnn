"""
Gradient-Based Domain Discovery via Clustering

Idea: Collect gradients from many batches, then cluster them.
Batches from the same domain should have similar gradient directions,
while batches from different domains should cluster separately.

This discovers latent domains WITHOUT knowing the labels beforehand.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
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


def collect_batch_gradient(env, agent, num_steps, device):
    """
    Collect a batch of transitions and compute the actor gradient.
    Returns the gradient vector for this batch.
    """
    states = []
    
    obs, _ = env.reset()
    
    for _ in range(num_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        states.append(obs_t)
        
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_t)
        
        next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy().item())
        
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    
    # Compute gradient for this batch
    agent.zero_grad()
    
    all_logprobs = []
    for state in states:
        _, logprob, _, _ = agent.get_action_and_value(state)
        all_logprobs.append(logprob)
    
    logprobs = torch.stack(all_logprobs).squeeze()
    loss = -logprobs.mean()
    loss.backward()
    
    # Extract actor gradients
    actor_grads = []
    for name, p in agent.named_parameters():
        if 'actor' in name and p.grad is not None:
            actor_grads.append(p.grad.flatten())
    
    if actor_grads:
        return torch.cat(actor_grads).detach().cpu().numpy()
    else:
        return torch.cat([p.grad.flatten() for p in agent.parameters() if p.grad is not None]).detach().cpu().numpy()


def main():
    print("="*70)
    print("GRADIENT-BASED DOMAIN DISCOVERY")
    print("Clustering batches by gradient direction")
    print("="*70)
    
    device = torch.device("cpu")
    
    # Create architecture
    variant = AG.AblationVariant(
        name='hetero', graph_type='hetero', hetero_base='sage',
        gin_num_layers=2, use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True,
    )
    
    agent = AG.AblationGinAgent(device, variant, hidden_dim=64, embedding_dim=32)
    
    # Create environments for both domains
    num_envs_per_domain = 5
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
    
    # Collect gradients from many batches
    num_batches_per_env = 8
    num_steps_per_batch = 30
    
    print(f"\nCollecting gradients from {num_envs_per_domain * num_batches_per_env * 2} batches...")
    
    gradients = []
    true_labels = []  # 0 = QF, 1 = BN
    
    # Collect from QF environments
    for env in envs_qf:
        for _ in range(num_batches_per_env):
            try:
                grad = collect_batch_gradient(env, agent, num_steps_per_batch, device)
                gradients.append(grad)
                true_labels.append(0)  # QF
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
    # Collect from BN environments
    for env in envs_bn:
        for _ in range(num_batches_per_env):
            try:
                grad = collect_batch_gradient(env, agent, num_steps_per_batch, device)
                gradients.append(grad)
                true_labels.append(1)  # BN
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
    # Close environments
    for env in envs_qf + envs_bn:
        env.close()
    
    gradients = np.array(gradients)
    true_labels = np.array(true_labels)
    
    print(f"Collected {len(gradients)} gradient vectors")
    print(f"  QF batches: {sum(true_labels == 0)}")
    print(f"  BN batches: {sum(true_labels == 1)}")
    
    # Normalize gradients (use direction only)
    grad_norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    gradients_normalized = gradients / (grad_norms + 1e-8)
    
    # PCA for dimensionality reduction
    print("\nApplying PCA...")
    pca = PCA(n_components=min(50, len(gradients)-1))
    gradients_pca = pca.fit_transform(gradients_normalized)
    print(f"  Explained variance (first 10): {pca.explained_variance_ratio_[:10].sum():.2%}")
    
    # Clustering with K-Means
    print("\nClustering with K-Means (K=2)...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(gradients_pca)
    
    # Evaluate clustering
    silhouette = silhouette_score(gradients_pca, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    
    # Compute cluster purity
    cluster_0_qf = sum((cluster_labels == 0) & (true_labels == 0))
    cluster_0_bn = sum((cluster_labels == 0) & (true_labels == 1))
    cluster_1_qf = sum((cluster_labels == 1) & (true_labels == 0))
    cluster_1_bn = sum((cluster_labels == 1) & (true_labels == 1))
    
    # Determine which cluster corresponds to which domain
    if cluster_0_qf + cluster_1_bn > cluster_0_bn + cluster_1_qf:
        # Cluster 0 = QF, Cluster 1 = BN
        purity_0 = cluster_0_qf / (cluster_0_qf + cluster_0_bn) if (cluster_0_qf + cluster_0_bn) > 0 else 0
        purity_1 = cluster_1_bn / (cluster_1_qf + cluster_1_bn) if (cluster_1_qf + cluster_1_bn) > 0 else 0
        cluster_mapping = {0: 'QF', 1: 'BN'}
    else:
        # Cluster 0 = BN, Cluster 1 = QF
        purity_0 = cluster_0_bn / (cluster_0_qf + cluster_0_bn) if (cluster_0_qf + cluster_0_bn) > 0 else 0
        purity_1 = cluster_1_qf / (cluster_1_qf + cluster_1_bn) if (cluster_1_qf + cluster_1_bn) > 0 else 0
        cluster_mapping = {0: 'BN', 1: 'QF'}
    
    avg_purity = (purity_0 + purity_1) / 2
    
    print(f"\nClustering Results:")
    print(f"  Silhouette Score: {silhouette:.3f}")
    print(f"  Adjusted Rand Index: {ari:.3f}")
    print(f"  Average Purity: {avg_purity:.1%}")
    print(f"  Cluster 0 ({cluster_mapping[0]}): {purity_0:.1%} pure")
    print(f"  Cluster 1 ({cluster_mapping[1]}): {purity_1:.1%} pure")
    
    # t-SNE for visualization
    print("\nApplying t-SNE for visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(gradients)-1))
    gradients_tsne = tsne.fit_transform(gradients_pca)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # t-SNE colored by TRUE labels
    ax = axes[0, 0]
    colors_true = ['#0077BB' if l == 0 else '#EE7733' for l in true_labels]
    ax.scatter(gradients_tsne[:, 0], gradients_tsne[:, 1], c=colors_true, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('(a) Gradient Space (True Labels)\nBlue=QF, Orange=BN', fontweight='bold')
    
    # t-SNE colored by CLUSTER labels
    ax = axes[0, 1]
    colors_cluster = ['#CC3311' if l == 0 else '#009988' for l in cluster_labels]
    ax.scatter(gradients_tsne[:, 0], gradients_tsne[:, 1], c=colors_cluster, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f'(b) Discovered Clusters (K-Means)\nCluster 0={cluster_mapping[0]}, Cluster 1={cluster_mapping[1]}', fontweight='bold')
    
    # PCA projection
    ax = axes[1, 0]
    ax.scatter(gradients_pca[:, 0], gradients_pca[:, 1], c=colors_true, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('(c) PCA Projection (True Labels)', fontweight='bold')
    
    # Confusion matrix style
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create confusion matrix
    conf_matrix = np.array([
        [cluster_0_qf, cluster_0_bn],
        [cluster_1_qf, cluster_1_bn]
    ])
    
    summary = f"""
GRADIENT-BASED DOMAIN DISCOVERY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Method:
  • Collect actor gradients from batches
  • Normalize to unit vectors (direction only)
  • PCA → K-Means clustering

Data:
  • {sum(true_labels == 0)} QF batches
  • {sum(true_labels == 1)} BN batches
  • {actor_params:,} gradient dimensions

Clustering Quality:
  • Silhouette Score: {silhouette:.3f}
  • Adjusted Rand Index: {ari:.3f}
  • Average Purity: {avg_purity:.1%}

Confusion Matrix:
                 True QF    True BN
  Cluster 0       {cluster_0_qf:3d}        {cluster_0_bn:3d}
  Cluster 1       {cluster_1_qf:3d}        {cluster_1_bn:3d}

CONCLUSION:
  {"✓ Domains discovered!" if avg_purity > 0.7 else "○ Partial separation" if avg_purity > 0.55 else "✗ No clear separation"}
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_clustering.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'gradient_clustering.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"\nSaved: {output_dir / 'gradient_clustering.png'}")
    
    # Also try different K values
    print("\n" + "="*70)
    print("Testing different K values...")
    print("="*70)
    
    for k in [2, 3, 4]:
        kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_k = kmeans_k.fit_predict(gradients_pca)
        sil_k = silhouette_score(gradients_pca, labels_k)
        print(f"  K={k}: Silhouette={sil_k:.3f}")


if __name__ == "__main__":
    main()
