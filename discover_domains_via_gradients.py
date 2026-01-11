"""
Domain Discovery via Multi-Objective Gradient Clustering

Problem: Given a single MDP with multi-objective rewards (makespan + energy),
can we discover latent domains by clustering gradient directions?

Hypothesis: Different batches from the same MDP push the agent toward different
objectives. Some batches prioritize makespan, others prioritize energy.

Method:
1. Collect multiple batches from same MDP
2. Compute gradients for each batch (using multi-objective reward)
3. Cluster gradients by direction
4. Analyze: Do clusters correspond to different objective preferences?
"""

import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.dataset_generator.gen_dataset import DatasetArgs


def load_mdp_config(config_path: str) -> tuple[DatasetArgs, int]:
    """Load MDP configuration from JSON."""
    import json
    
    with open(config_path) as f:
        cfg = json.load(f)
    
    ds_cfg = cfg["dataset"]
    args = DatasetArgs(
        host_count=ds_cfg["hosts"],
        vm_count=ds_cfg["vms"],
        workflow_count=ds_cfg["workflow_count"],
        style=ds_cfg["style"],
        gnp_p=ds_cfg.get("edge_probability"),
        gnp_min_n=ds_cfg.get("min_tasks", 10),
        gnp_max_n=ds_cfg.get("max_tasks", 10),
        min_task_length=ds_cfg.get("task_length", {}).get("min", 500),
        max_task_length=ds_cfg.get("task_length", {}).get("max", 100000),
    )
    
    seed = cfg["training_seeds"][0]
    return args, seed


def collect_batch_with_objectives(env, agent, num_steps, device):
    """Collect batch and track makespan/energy separately."""
    obs_list = []
    action_list = []
    makespan_rewards = []
    energy_rewards = []
    
    obs, _ = env.reset()
    
    for _ in range(num_steps):
        obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(1, -1)).to(device)
        
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        
        obs_list.append(obs_tensor)
        action_list.append(action)
        
        obs, reward, terminated, truncated, info = env.step(int(action.item()))
        
        # Extract separate objectives
        makespan_reward = reward  # Negative makespan
        energy_reward = -info.get("total_energy_active", 0.0)  # Negative energy
        
        makespan_rewards.append(makespan_reward)
        energy_rewards.append(energy_reward)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    return obs_list, action_list, makespan_rewards, energy_rewards


def compute_gradient_for_objective(agent, obs_list, action_list, rewards, alpha_makespan, alpha_energy):
    """Compute gradient with specific alpha weights."""
    agent.zero_grad()
    
    # REINFORCE
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
    
    return torch.cat(grad_parts)


def discover_domains(
    mdp1_config_path: str,
    mdp2_config_path: str,
    host_specs_file: str,
    num_batches: int = 50,
    batch_size: int = 64,
    num_clusters: int = 2,
    output_dir: str = "logs/domain_discovery"
):
    """Discover domains via gradient clustering from two MDPs."""
    
    print("="*70)
    print("Domain Discovery via Multi-Objective Gradient Clustering")
    print("="*70)
    print(f"MDP1: {mdp1_config_path}")
    print(f"MDP2: {mdp2_config_path}")
    print(f"Host specs: {host_specs_file}")
    print(f"Batches per MDP: {num_batches}")
    print(f"Clusters: {num_clusters}")
    print()
    
    # Setup
    os.environ['HOST_SPECS_PATH'] = host_specs_file
    device = torch.device("cpu")
    
    # Load MDPs
    print("[1/6] Loading MDPs...")
    mdp1_args, mdp1_seed = load_mdp_config(mdp1_config_path)
    mdp2_args, mdp2_seed = load_mdp_config(mdp2_config_path)
    print(f"  MDP1: {mdp1_args.style}, p={mdp1_args.gnp_p}")
    print(f"  MDP2: {mdp2_args.style}, p={mdp2_args.gnp_p}")
    
    # Initialize agent
    print("\n[2/6] Initializing agent...")
    variant = AblationVariant(
        name="hetero", graph_type="hetero", gin_num_layers=2,
        use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
    )
    agent = AblationGinAgent(device=device, variant=variant, hidden_dim=128, embedding_dim=64)
    print("  ✓ Single agent for both MDPs")
    
    # Create environments
    print("\n[3/6] Creating environments...")
    env1 = GinAgentWrapper(CloudSchedulingGymEnvironment(
        dataset_args=mdp1_args, collect_timelines=False, compute_metrics=False
    ))
    env1.reset(seed=mdp1_seed)
    
    env2 = GinAgentWrapper(CloudSchedulingGymEnvironment(
        dataset_args=mdp2_args, collect_timelines=False, compute_metrics=False
    ))
    env2.reset(seed=mdp2_seed)
    print("  ✓ Environments created")
    
    # Collect batches from BOTH MDPs
    print(f"\n[4/6] Collecting {num_batches} batches from each MDP...")
    
    gradients = []
    batch_stats = []
    true_labels = []  # Ground truth: 0=MDP1, 1=MDP2
    
    # Collect from MDP1
    print(f"  Collecting from MDP1 ({mdp1_args.style})...")
    for i in range(num_batches):
        if i % 10 == 0:
            print(f"    Batch {i}/{num_batches}...")
        
        obs, act, mk_rew, en_rew = collect_batch_with_objectives(env1, agent, batch_size, device)
        combined_rew = [mk + en for mk, en in zip(mk_rew, en_rew)]
        grad = compute_gradient_for_objective(agent, obs, act, combined_rew, 1.0, 1.0)
        
        if grad is not None:
            gradients.append(grad.cpu().numpy())
            batch_stats.append({
                'batch_id': i,
                'mdp': 'MDP1',
                'style': mdp1_args.style,
                'mean_makespan': np.mean(mk_rew),
                'mean_energy': np.mean(en_rew),
            })
            true_labels.append(0)
    
    # Collect from MDP2
    print(f"  Collecting from MDP2 ({mdp2_args.style})...")
    for i in range(num_batches):
        if i % 10 == 0:
            print(f"    Batch {i}/{num_batches}...")
        
        obs, act, mk_rew, en_rew = collect_batch_with_objectives(env2, agent, batch_size, device)
        combined_rew = [mk + en for mk, en in zip(mk_rew, en_rew)]
        grad = compute_gradient_for_objective(agent, obs, act, combined_rew, 1.0, 1.0)
        
        if grad is not None:
            gradients.append(grad.cpu().numpy())
            batch_stats.append({
                'batch_id': i + num_batches,
                'mdp': 'MDP2',
                'style': mdp2_args.style,
                'mean_makespan': np.mean(mk_rew),
                'mean_energy': np.mean(en_rew),
            })
            true_labels.append(1)
    
    env1.close()
    env2.close()
    
    print(f"  ✓ Collected {len(gradients)} gradient vectors ({num_batches} per MDP)")
    true_labels = np.array(true_labels)
    
    # Cluster gradients
    print(f"\n[5/6] Clustering gradients into {num_clusters} domains...")
    
    # Normalize gradients (direction only)
    gradients_normalized = np.array([g / (np.linalg.norm(g) + 1e-9) for g in gradients])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(gradients_normalized)
    
    print(f"  ✓ Clustering complete")
    
    # Compute clustering accuracy
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    
    # Compute purity
    contingency_matrix = np.zeros((num_clusters, 2))
    for i in range(len(cluster_labels)):
        contingency_matrix[cluster_labels[i], true_labels[i]] += 1
    
    purity = np.sum(np.max(contingency_matrix, axis=1)) / len(cluster_labels)
    
    print(f"\n  Clustering Quality:")
    print(f"    Adjusted Rand Index: {ari:.4f}")
    print(f"    Normalized Mutual Info: {nmi:.4f}")
    print(f"    Purity: {purity:.4f}")
    
    # Analyze clusters
    print("\n" + "="*70)
    print("CLUSTER ANALYSIS")
    print("="*70)
    
    for cluster_id in range(num_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_batches = [batch_stats[i] for i in range(len(batch_stats)) if cluster_mask[i]]
        
        if not cluster_batches:
            continue
        
        # Count MDP composition
        mdp1_count = sum(1 for b in cluster_batches if b['mdp'] == 'MDP1')
        mdp2_count = sum(1 for b in cluster_batches if b['mdp'] == 'MDP2')
        
        # Compute cluster statistics
        mean_makespan = np.mean([b['mean_makespan'] for b in cluster_batches])
        mean_energy = np.mean([b['mean_energy'] for b in cluster_batches])
        
        print(f"\nCluster {cluster_id}: {len(cluster_batches)} batches")
        print(f"  Composition:")
        print(f"    MDP1 ({mdp1_args.style}): {mdp1_count} batches ({100*mdp1_count/len(cluster_batches):.1f}%)")
        print(f"    MDP2 ({mdp2_args.style}): {mdp2_count} batches ({100*mdp2_count/len(cluster_batches):.1f}%)")
        print(f"  Mean makespan reward: {mean_makespan:.4f}")
        print(f"  Mean energy reward:   {mean_energy:.4f}")
        
        # Determine dominant MDP
        if mdp1_count > mdp2_count * 1.5:
            print(f"  → DOMINATED by MDP1 ({mdp1_args.style})")
        elif mdp2_count > mdp1_count * 1.5:
            print(f"  → DOMINATED by MDP2 ({mdp2_args.style})")
        else:
            print(f"  → MIXED cluster")
    
    # Compute inter-cluster conflict
    print("\n" + "="*70)
    print("INTER-CLUSTER GRADIENT CONFLICT")
    print("="*70)
    
    cluster_centroids = kmeans.cluster_centers_
    
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            centroid_i = cluster_centroids[i]
            centroid_j = cluster_centroids[j]
            
            # Cosine similarity
            cosine_sim = np.dot(centroid_i, centroid_j) / (
                np.linalg.norm(centroid_i) * np.linalg.norm(centroid_j) + 1e-9
            )
            
            print(f"\nCluster {i} vs Cluster {j}:")
            print(f"  Cosine similarity: {cosine_sim:+.4f}")
            
            if cosine_sim < -0.5:
                print(f"  → HIGH CONFLICT: Clusters push in opposite directions")
            elif cosine_sim < 0:
                print(f"  → MODERATE CONFLICT: Some opposition")
            elif cosine_sim < 0.5:
                print(f"  → WEAK CONFLICT: Nearly orthogonal")
            else:
                print(f"  → ALIGNED: Similar gradient directions")
    
    # Visualize (PCA)
    print("\n" + "="*70)
    print("VISUALIZATION")
    print("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # PCA to 2D
    pca = PCA(n_components=2)
    gradients_2d = pca.fit_transform(gradients_normalized)
    
    # Create two plots: predicted clusters and true labels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Predicted clusters
    scatter1 = ax1.scatter(
        gradients_2d[:, 0], gradients_2d[:, 1],
        c=cluster_labels, cmap='viridis', alpha=0.6, s=100, edgecolors='black', linewidth=0.5
    )
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title(f'Predicted Clusters (Purity: {purity:.2f})')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Predicted Cluster')
    
    # Plot 2: True labels
    scatter2 = ax2.scatter(
        gradients_2d[:, 0], gradients_2d[:, 1],
        c=true_labels, cmap='coolwarm', alpha=0.6, s=100, edgecolors='black', linewidth=0.5
    )
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.set_title(f'True Labels (MDP1 vs MDP2)')
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter2, ax=ax2, ticks=[0, 1])
    cbar.set_ticklabels([f'MDP1 ({mdp1_args.style})', f'MDP2 ({mdp2_args.style})'])
    
    plt.tight_layout()
    plot_file = output_path / "gradient_clusters.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved plot: {plot_file}")
    
    # Save cluster assignments
    import csv
    csv_file = output_path / "cluster_assignments.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['batch_id', 'cluster', 'mean_makespan', 'mean_energy'])
        for i, (stats, label) in enumerate(zip(batch_stats, cluster_labels)):
            writer.writerow([
                stats['batch_id'], label,
                stats['mean_makespan'], stats['mean_energy']
            ])
    
    print(f"  ✓ Saved data: {csv_file}")
    print()
    
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    print()
    print("If clusters show different objective preferences:")
    print("  → Domain discovery successful!")
    print("  → Different batches push toward different objectives")
    print("  → Need separate policies or task conditioning")
    print()
    print("If clusters are similar:")
    print("  → No clear domain structure")
    print("  → Single policy may suffice")
    print()


if __name__ == "__main__":
    import sys
    
    mdp1_config = sys.argv[1] if len(sys.argv) > 1 else "data/rl_configs/extreme_longcp_p095_bottleneck.json"
    mdp2_config = sys.argv[2] if len(sys.argv) > 2 else "data/rl_configs/extreme_wide_p005_bottleneck.json"
    host_specs = sys.argv[3] if len(sys.argv) > 3 else "data/host_specs.json"
    
    discover_domains(
        mdp1_config_path=mdp1_config,
        mdp2_config_path=mdp2_config,
        host_specs_file=host_specs,
        num_batches=50,
        batch_size=64,
        num_clusters=2,
        output_dir="logs/domain_discovery"
    )
