"""
Gradient Subspace Discovery

Unsupervised method to discover natural gradient directions and cluster MDPs.

Method:
1. Sample random objective combinations (α_makespan, α_energy)
2. Collect gradients for all MDPs under these random objectives
3. Use PCA to discover principal gradient directions
4. Project each MDP's gradients onto these directions
5. Cluster MDPs by their gradient projections

This discovers domains WITHOUT knowing which MDPs are "wide" or "long_cp"!
"""

import os
import json
import torch
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE

from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.dataset_generator.gen_dataset import DatasetArgs


def create_mdp_from_params(params, host_specs_file):
    """Create MDP from parameters."""
    edge_prob, min_tasks, max_tasks, min_length, max_length = params
    
    style = "long_cp" if edge_prob > 0.5 else "wide"
    
    args = DatasetArgs(
        host_count=10,
        vm_count=10,
        workflow_count=1,
        style=style,
        gnp_p=float(edge_prob),
        gnp_min_n=int(min_tasks),
        gnp_max_n=int(max_tasks),
        min_task_length=int(min_length),
        max_task_length=int(max_length),
    )
    
    os.environ['HOST_SPECS_PATH'] = host_specs_file
    
    env = GinAgentWrapper(CloudSchedulingGymEnvironment(
        dataset_args=args, collect_timelines=False, compute_metrics=True
    ))
    
    return env, args


def compute_gradient_for_objective_weights(
    agent, env, alpha_makespan: float, alpha_energy: float,
    num_steps=256, device=torch.device("cpu"), seed=None
):
    """Compute policy gradient for weighted objective."""
    obs_list = []
    action_list = []
    rewards = []
    
    if seed is not None:
        obs, _ = env.reset(seed=int(seed))
    else:
        obs, _ = env.reset()
    
    for _ in range(num_steps):
        obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(1, -1)).to(device)
        
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        
        obs_list.append(obs_tensor)
        action_list.append(action)
        
        obs, reward, terminated, truncated, info = env.step(int(action.item()))
        
        if terminated or truncated:
            makespan = info.get('makespan', 0.0)
            energy = info.get('total_energy_active', 0.0)
            energy_normalized = energy / 1e8
            
            # Weighted objective
            episode_reward = -alpha_makespan * makespan - alpha_energy * energy_normalized
            rewards.append(episode_reward)
            obs, _ = env.reset()
        else:
            rewards.append(0.0)
    
    # REINFORCE gradient
    agent.zero_grad()
    
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    
    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    loss = 0.0
    for obs_t, action_t, ret in zip(obs_list, action_list, returns):
        _, log_prob, _, _ = agent.get_action_and_value(obs_t, action_t)
        loss = loss - log_prob * ret
    
    loss = loss / len(obs_list)
    loss.backward()
    
    # Extract gradients
    grad_parts = []
    for p in agent.actor.parameters():
        if p.grad is not None:
            grad_parts.append(p.grad.view(-1).detach().clone())
    
    agent.zero_grad()
    
    if not grad_parts:
        return None
    
    return torch.cat(grad_parts).cpu().numpy()


def discover_gradient_subspace(
    agent, mdp_configs: List[dict],
    host_specs_file: str,
    device: torch.device,
    num_random_objectives: int = 20,
    num_replicates: int = 3,
    n_components: int = 2
):
    """
    Discover principal gradient directions via PCA.
    
    Returns:
    - pca: Fitted PCA model
    - all_gradients: All collected gradients
    - gradient_matrix: [n_mdps, n_objectives, gradient_dim]
    """
    print("\n[1/3] Collecting gradients for random objectives...")
    
    # Sample random objective weights
    random_objectives = []
    for _ in range(num_random_objectives):
        alpha_makespan = random.uniform(0, 1)
        alpha_energy = 1 - alpha_makespan
        random_objectives.append((alpha_makespan, alpha_energy))
    
    print(f"  Sampled {num_random_objectives} random objective combinations")
    
    # Collect gradients for all (MDP, objective) pairs
    all_gradients = []
    gradient_matrix = []  # [n_mdps, n_objectives, gradient_dim]
    
    pbar = tqdm(mdp_configs, desc="  MDPs", ncols=100)
    
    for mdp_idx, mdp_config in enumerate(pbar):
        params = [
            mdp_config['edge_prob'],
            mdp_config['min_tasks'],
            mdp_config['max_tasks'],
            mdp_config['min_length'],
            mdp_config['max_length']
        ]
        env, _ = create_mdp_from_params(params, host_specs_file)
        
        mdp_gradients = []
        
        for obj_idx, (alpha_m, alpha_e) in enumerate(random_objectives):
            # Average over replicates
            grads = []
            for r in range(num_replicates):
                g = compute_gradient_for_objective_weights(
                    agent, env, alpha_m, alpha_e,
                    num_steps=256, device=device,
                    seed=12345 + mdp_idx * 1000 + obj_idx * 10 + r
                )
                if g is not None:
                    grads.append(g)
            
            if grads:
                avg_grad = np.mean(np.stack(grads, axis=0), axis=0)
                mdp_gradients.append(avg_grad)
                all_gradients.append(avg_grad)
        
        env.close()
        
        if mdp_gradients:
            gradient_matrix.append(np.stack(mdp_gradients, axis=0))
    
    gradient_matrix = np.array(gradient_matrix)  # [n_mdps, n_objectives, grad_dim]
    all_gradients = np.array(all_gradients)  # [n_mdps * n_objectives, grad_dim]
    
    print(f"  ✓ Collected {len(all_gradients)} gradients")
    print(f"    Shape: {gradient_matrix.shape}")
    
    # Discover principal gradient directions
    print("\n[2/3] Discovering principal gradient directions (PCA)...")
    
    pca = PCA(n_components=n_components)
    pca.fit(all_gradients)
    
    print(f"  ✓ Discovered {n_components} principal directions")
    print(f"    Explained variance: {pca.explained_variance_ratio_}")
    print(f"    Cumulative variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    return pca, all_gradients, gradient_matrix, random_objectives


def cluster_mdps_by_gradient_projections(
    pca, gradient_matrix, random_objectives,
    n_clusters: int = 2
):
    """
    Cluster MDPs by their gradient projections onto principal directions.
    """
    print("\n[3/3] Clustering MDPs by gradient projections...")
    
    n_mdps, n_objectives, grad_dim = gradient_matrix.shape
    
    # For each MDP, compute features based on gradient projections
    mdp_features = []
    
    for mdp_idx in range(n_mdps):
        # Get all gradients for this MDP
        mdp_grads = gradient_matrix[mdp_idx]  # [n_objectives, grad_dim]
        
        # Project onto principal components
        projections = pca.transform(mdp_grads)  # [n_objectives, n_components]
        
        # Aggregate projections (mean and std along objectives)
        proj_mean = projections.mean(axis=0)
        proj_std = projections.std(axis=0)
        
        # Combine into feature vector
        features = np.concatenate([proj_mean, proj_std])
        mdp_features.append(features)
    
    mdp_features = np.array(mdp_features)
    
    print(f"  MDP feature shape: {mdp_features.shape}")
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(mdp_features)
    
    # Compute silhouette score
    if len(np.unique(clusters)) > 1:
        sil_score = silhouette_score(mdp_features, clusters)
    else:
        sil_score = 0.0
    
    print(f"  ✓ Clustered into {n_clusters} domains")
    print(f"    Cluster sizes: {np.bincount(clusters)}")
    print(f"    Silhouette score: {sil_score:.3f}")
    
    return clusters, mdp_features, sil_score


def gradient_subspace_discovery(
    host_specs_file: str,
    output_dir: str = "logs/gradient_subspace",
    num_mdp_samples: int = 30,
    num_random_objectives: int = 20,
    use_trained_agent: bool = False,
    trained_agent_path: str = None
):
    """
    Main gradient subspace discovery pipeline.
    """
    print("="*80)
    print("GRADIENT SUBSPACE DISCOVERY")
    print("="*80)
    print("\nUnsupervised domain discovery via principal gradient directions")
    print()
    
    device = torch.device("cpu")
    
    # Global seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Sample diverse MDPs (balanced: half wide, half long_cp)
    print(f"[Setup] Sampling {num_mdp_samples} diverse MDP configurations...")
    mdp_configs = []
    true_labels = []
    
    for i in range(num_mdp_samples // 2):
        # Wide MDPs
        edge_prob = 0.05 + (i / (num_mdp_samples // 2)) * 0.35
        min_tasks = random.randint(20, 40)
        max_tasks = random.randint(min_tasks, 50)
        min_length = random.randint(100, 2000)
        max_length = random.randint(max(min_length, 5000), 50000)
        
        mdp_configs.append({
            'edge_prob': edge_prob,
            'min_tasks': min_tasks,
            'max_tasks': max_tasks,
            'min_length': min_length,
            'max_length': max_length
        })
        true_labels.append(0)
    
    for i in range(num_mdp_samples // 2):
        # Long CP MDPs
        edge_prob = 0.60 + (i / (num_mdp_samples // 2)) * 0.35
        min_tasks = random.randint(5, 15)
        max_tasks = random.randint(min_tasks, 20)
        min_length = random.randint(50000, 100000)
        max_length = random.randint(max(min_length, 100000), 200000)
        
        mdp_configs.append({
            'edge_prob': edge_prob,
            'min_tasks': min_tasks,
            'max_tasks': max_tasks,
            'min_length': min_length,
            'max_length': max_length
        })
        true_labels.append(1)
    
    true_labels = np.array(true_labels)
    
    print(f"  ✓ Sampled {len(mdp_configs)} MDPs")
    print(f"    Wide (label=0): {np.sum(true_labels == 0)} MDPs")
    print(f"    Long CP (label=1): {np.sum(true_labels == 1)} MDPs")
    
    # Initialize agent
    print(f"\n[Setup] Initializing agent...")
    
    if use_trained_agent and trained_agent_path:
        print(f"  Loading trained agent from {trained_agent_path}")
        variant = AblationVariant(
            name="hetero", graph_type="hetero", gin_num_layers=2,
            use_batchnorm=True, use_task_dependencies=True,
            use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
        )
        agent = AblationGinAgent(device=device, variant=variant, hidden_dim=128, embedding_dim=64)
        checkpoint = torch.load(trained_agent_path, map_location=device)
        agent.load_state_dict(checkpoint)
        agent.eval()
        print("  ✓ Loaded trained agent")
    else:
        print("  Using random (untrained) agent")
        variant = AblationVariant(
            name="hetero", graph_type="hetero", gin_num_layers=2,
            use_batchnorm=True, use_task_dependencies=True,
            use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
        )
        agent = AblationGinAgent(device=device, variant=variant, hidden_dim=128, embedding_dim=64)
        
        # Initialize with dummy forward pass
        dummy_env, _ = create_mdp_from_params([0.5, 20, 30, 1000, 5000], host_specs_file)
        dummy_obs, _ = dummy_env.reset(seed=42)
        dummy_obs_tensor = torch.from_numpy(np.asarray(dummy_obs, dtype=np.float32).reshape(1, -1)).to(device)
        with torch.no_grad():
            _ = agent.get_action_and_value(dummy_obs_tensor)
        dummy_env.close()
        print("  ✓ Initialized random agent")
    
    # Discover gradient subspace
    pca, all_gradients, gradient_matrix, random_objectives = discover_gradient_subspace(
        agent, mdp_configs, host_specs_file, device,
        num_random_objectives=num_random_objectives,
        num_replicates=3,
        n_components=2
    )
    
    # Cluster MDPs
    clusters, mdp_features, sil_score = cluster_mdps_by_gradient_projections(
        pca, gradient_matrix, random_objectives, n_clusters=2
    )
    
    # Evaluate clustering quality
    nmi = normalized_mutual_info_score(true_labels, clusters)
    ari = adjusted_rand_score(true_labels, clusters)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nClustering Quality:")
    print(f"  NMI: {nmi:.3f}")
    print(f"  ARI: {ari:.3f}")
    print(f"  Silhouette: {sil_score:.3f}")
    
    if nmi > 0.4:
        print("\n  ✓ GOOD: Discovered domains match true structure!")
    elif nmi > 0.2:
        print("\n  ~ MODERATE: Some structure discovered")
    else:
        print("\n  ✗ POOR: Clustering doesn't match true structure")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'mdp_configs': mdp_configs,
        'true_labels': true_labels.tolist(),
        'clusters': clusters.tolist(),
        'nmi': float(nmi),
        'ari': float(ari),
        'silhouette': float(sil_score),
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'num_random_objectives': num_random_objectives,
        'use_trained_agent': use_trained_agent
    }
    
    with open(output_path / "subspace_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualizations
    create_visualizations(
        pca, all_gradients, gradient_matrix, mdp_features,
        mdp_configs, true_labels, clusters, random_objectives,
        output_path
    )
    
    print(f"\n  ✓ Saved results to {output_dir}")
    
    return results


def create_visualizations(
    pca, all_gradients, gradient_matrix, mdp_features,
    mdp_configs, true_labels, clusters, random_objectives,
    output_path
):
    """Create comprehensive visualizations."""
    
    edge_probs = [cfg['edge_prob'] for cfg in mdp_configs]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: All gradients in PCA space
    ax = axes[0, 0]
    projections = pca.transform(all_gradients)
    ax.scatter(projections[:, 0], projections[:, 1], alpha=0.3, s=20, c='gray')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_title('All Gradients in PCA Space', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: MDP features (true labels)
    ax = axes[0, 1]
    colors_true = ['red' if l == 0 else 'blue' for l in true_labels]
    ax.scatter(mdp_features[:, 0], mdp_features[:, 1], c=colors_true, alpha=0.6, s=100)
    ax.set_xlabel('Mean PC1 Projection', fontsize=11)
    ax.set_ylabel('Mean PC2 Projection', fontsize=11)
    ax.set_title('MDP Features (True Labels)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: MDP features (predicted clusters)
    ax = axes[0, 2]
    colors_pred = ['red' if c == 0 else 'blue' for c in clusters]
    ax.scatter(mdp_features[:, 0], mdp_features[:, 1], c=colors_pred, alpha=0.6, s=100)
    ax.set_xlabel('Mean PC1 Projection', fontsize=11)
    ax.set_ylabel('Mean PC2 Projection', fontsize=11)
    ax.set_title('MDP Features (Predicted Clusters)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Clusters vs edge probability
    ax = axes[1, 0]
    for cluster_id in [0, 1]:
        mask = clusters == cluster_id
        ax.scatter(
            [edge_probs[i] for i in range(len(edge_probs)) if mask[i]],
            [cluster_id] * np.sum(mask),
            alpha=0.6, s=100, label=f'Cluster {cluster_id}'
        )
    ax.set_xlabel('Edge Probability', fontsize=11)
    ax.set_ylabel('Cluster', fontsize=11)
    ax.set_title('Clusters vs Edge Probability', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Confusion matrix
    ax = axes[1, 1]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, clusters)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Cluster', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Plot 6: Objective weight distribution
    ax = axes[1, 2]
    alphas_m = [obj[0] for obj in random_objectives]
    alphas_e = [obj[1] for obj in random_objectives]
    ax.scatter(alphas_m, alphas_e, alpha=0.6, s=100)
    ax.set_xlabel('α_makespan', fontsize=11)
    ax.set_ylabel('α_energy', fontsize=11)
    ax.set_title('Random Objective Weights', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "subspace_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import sys
    
    host_specs = sys.argv[1] if len(sys.argv) > 1 else "data/host_specs.json"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    num_objectives = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    # Option to use trained agent
    use_trained = len(sys.argv) > 4 and sys.argv[4].lower() == 'trained'
    trained_path = sys.argv[5] if len(sys.argv) > 5 else None
    
    results = gradient_subspace_discovery(
        host_specs,
        num_mdp_samples=num_samples,
        num_random_objectives=num_objectives,
        use_trained_agent=use_trained,
        trained_agent_path=trained_path
    )
