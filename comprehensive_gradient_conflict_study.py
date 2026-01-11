"""
Comprehensive Gradient Conflict Study

Systematic exploration of:
1. Architecture complexity: Simple MLP → Tiny GNN → Large GNN
2. Distance metrics: Cosine similarity, Wasserstein distance, Euclidean
3. Cluster quality: NMI, ARI, Silhouette score
4. Gradient visualization: PCA, t-SNE
5. Task characteristics: Edge prob, task count, task length
"""

import os
import json
import torch
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine, euclidean

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


def compute_gradient_for_objective(
    agent, env, objective_type: str,
    num_steps=256, device=torch.device("cpu"), seed=None
):
    """Compute policy gradient for a specific objective."""
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
            
            if objective_type == "makespan":
                episode_reward = -makespan
            elif objective_type == "energy":
                episode_reward = -energy_normalized
            else:
                raise ValueError(f"Unknown objective: {objective_type}")
            
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


def compute_conflict_cosine(g_makespan, g_energy):
    """Compute conflict using cosine similarity."""
    cos_sim = np.dot(g_makespan, g_energy) / (
        np.linalg.norm(g_makespan) * np.linalg.norm(g_energy) + 1e-9
    )
    return cos_sim


def compute_conflict_wasserstein(g_makespan, g_energy, num_bins=100):
    """Compute conflict using Wasserstein distance."""
    # Normalize gradients to [0, 1] for histogram
    g_m_norm = (g_makespan - g_makespan.min()) / (g_makespan.max() - g_makespan.min() + 1e-9)
    g_e_norm = (g_energy - g_energy.min()) / (g_energy.max() - g_energy.min() + 1e-9)
    
    # Compute Wasserstein distance
    w_dist = wasserstein_distance(g_m_norm, g_e_norm)
    
    # Normalize to [-1, 1] range (higher distance = more conflict)
    # Negate so that high conflict is negative (consistent with cosine)
    return -w_dist


def compute_conflict_euclidean(g_makespan, g_energy):
    """Compute conflict using normalized Euclidean distance."""
    # Normalize gradients
    g_m_norm = g_makespan / (np.linalg.norm(g_makespan) + 1e-9)
    g_e_norm = g_energy / (np.linalg.norm(g_energy) + 1e-9)
    
    # Euclidean distance
    euc_dist = np.linalg.norm(g_m_norm - g_e_norm)
    
    # Normalize to [-1, 1] range (negate for consistency)
    return -euc_dist / np.sqrt(2)


def collect_gradients_for_architecture(
    agent, mdp_configs: List[dict],
    host_specs_file: str,
    device: torch.device,
    num_replicates: int = 3,
    arch_name: str = ""
):
    """Collect makespan and energy gradients for all MDPs."""
    makespan_gradients = []
    energy_gradients = []
    
    pbar = tqdm(mdp_configs, desc=f"  {arch_name}", ncols=100, leave=False)
    
    for mdp_idx, mdp_config in enumerate(pbar):
        params = [
            mdp_config['edge_prob'],
            mdp_config['min_tasks'],
            mdp_config['max_tasks'],
            mdp_config['min_length'],
            mdp_config['max_length']
        ]
        env, _ = create_mdp_from_params(params, host_specs_file)
        
        makespan_grads = []
        energy_grads = []
        
        for r in range(num_replicates):
            g_m = compute_gradient_for_objective(
                agent, env, "makespan",
                num_steps=256, device=device, seed=12345 + mdp_idx * 100 + r
            )
            g_e = compute_gradient_for_objective(
                agent, env, "energy",
                num_steps=256, device=device, seed=12345 + mdp_idx * 100 + r + 50
            )
            
            if g_m is not None:
                makespan_grads.append(g_m)
            if g_e is not None:
                energy_grads.append(g_e)
        
        env.close()
        
        # Average gradients
        if makespan_grads and energy_grads:
            makespan_gradients.append(np.mean(np.stack(makespan_grads, axis=0), axis=0))
            energy_gradients.append(np.mean(np.stack(energy_grads, axis=0), axis=0))
        else:
            # Fallback to zeros
            if makespan_grads:
                makespan_gradients.append(makespan_grads[0])
                energy_gradients.append(np.zeros_like(makespan_grads[0]))
            else:
                makespan_gradients.append(np.zeros(100))
                energy_gradients.append(np.zeros(100))
    
    return np.array(makespan_gradients), np.array(energy_gradients)


def compute_conflicts_with_metric(makespan_grads, energy_grads, metric='cosine'):
    """Compute conflicts using specified metric."""
    num_mdps = len(makespan_grads)
    conflicts = []
    
    for i in range(num_mdps):
        if metric == 'cosine':
            conflict = compute_conflict_cosine(makespan_grads[i], energy_grads[i])
        elif metric == 'wasserstein':
            conflict = compute_conflict_wasserstein(makespan_grads[i], energy_grads[i])
        elif metric == 'euclidean':
            conflict = compute_conflict_euclidean(makespan_grads[i], energy_grads[i])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        conflicts.append(conflict)
    
    return np.array(conflicts)


def evaluate_clustering(conflicts, true_labels, n_clusters=2):
    """Evaluate clustering quality."""
    # Cluster based on conflicts
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pred_labels = kmeans.fit_predict(conflicts.reshape(-1, 1))
    
    # Compute metrics
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    # Silhouette score (if enough samples)
    if len(conflicts) >= n_clusters:
        sil = silhouette_score(conflicts.reshape(-1, 1), pred_labels)
    else:
        sil = 0.0
    
    return {
        'nmi': nmi,
        'ari': ari,
        'silhouette': sil,
        'pred_labels': pred_labels
    }


def comprehensive_study(
    host_specs_file: str,
    output_dir: str = "logs/comprehensive_study",
    num_mdp_samples: int = 30
):
    """Run comprehensive gradient conflict study."""
    print("="*80)
    print("COMPREHENSIVE GRADIENT CONFLICT STUDY")
    print("="*80)
    print("\nSystematic exploration of:")
    print("  1. Architecture complexity: MLP → Tiny GNN → Large GNN")
    print("  2. Distance metrics: Cosine, Wasserstein, Euclidean")
    print("  3. Cluster quality: NMI, ARI, Silhouette")
    print("  4. Gradient visualization: PCA, t-SNE")
    print()
    
    device = torch.device("cpu")
    
    # Global seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Define architecture progression: Simple → Complex
    architectures = [
        {
            'name': 'Tiny GNN (1 layer, 16 dim)',
            'variant': AblationVariant(
                name="tiny", graph_type="hetero", gin_num_layers=1,
                use_batchnorm=False, use_task_dependencies=True,
                use_actor_global_embedding=False, mlp_only=False, gat_heads=1,
            ),
            'hidden_dim': 16,
            'embedding_dim': 8,
            'complexity': 1
        },
        {
            'name': 'Small GNN (2 layers, 32 dim)',
            'variant': AblationVariant(
                name="small", graph_type="hetero", gin_num_layers=2,
                use_batchnorm=False, use_task_dependencies=True,
                use_actor_global_embedding=True, mlp_only=False, gat_heads=2,
            ),
            'hidden_dim': 32,
            'embedding_dim': 16,
            'complexity': 2
        },
        {
            'name': 'Medium GNN (2 layers, 64 dim)',
            'variant': AblationVariant(
                name="medium", graph_type="hetero", gin_num_layers=2,
                use_batchnorm=True, use_task_dependencies=True,
                use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
            ),
            'hidden_dim': 64,
            'embedding_dim': 32,
            'complexity': 3
        },
        {
            'name': 'Large GNN (2 layers, 128 dim)',
            'variant': AblationVariant(
                name="large", graph_type="hetero", gin_num_layers=2,
                use_batchnorm=True, use_task_dependencies=True,
                use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
            ),
            'hidden_dim': 128,
            'embedding_dim': 64,
            'complexity': 4
        },
        {
            'name': 'Deep GNN (3 layers, 128 dim)',
            'variant': AblationVariant(
                name="deep", graph_type="hetero", gin_num_layers=3,
                use_batchnorm=True, use_task_dependencies=True,
                use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
            ),
            'hidden_dim': 128,
            'embedding_dim': 64,
            'complexity': 5
        },
    ]
    
    # Sample diverse MDPs with clear structure
    print(f"[1/4] Sampling {num_mdp_samples} diverse MDP configurations...")
    mdp_configs = []
    true_labels = []  # 0 = wide (low edge_prob), 1 = long_cp (high edge_prob)
    
    # Create balanced dataset: half wide, half long_cp
    for i in range(num_mdp_samples // 2):
        # Wide MDPs (low edge probability)
        edge_prob = 0.05 + (i / (num_mdp_samples // 2)) * 0.35  # 0.05 to 0.40
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
        # Long CP MDPs (high edge probability)
        edge_prob = 0.60 + (i / (num_mdp_samples // 2)) * 0.35  # 0.60 to 0.95
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
    
    # Distance metrics to test
    metrics = ['cosine', 'wasserstein', 'euclidean']
    
    # Collect gradients for each architecture
    print(f"\n[2/4] Collecting gradients for {len(architectures)} architectures...")
    
    all_results = {}
    
    for arch_idx, arch_config in enumerate(architectures):
        print(f"\n  [{arch_idx+1}/{len(architectures)}] {arch_config['name']}")
        
        # Initialize agent
        agent = AblationGinAgent(
            device=device,
            variant=arch_config['variant'],
            hidden_dim=arch_config['hidden_dim'],
            embedding_dim=arch_config['embedding_dim']
        )
        
        # Initialize lazy modules with a dummy forward pass
        # Create a dummy environment to get observation shape
        dummy_env, _ = create_mdp_from_params(
            [0.5, 20, 30, 1000, 5000], host_specs_file
        )
        dummy_obs, _ = dummy_env.reset(seed=42)
        dummy_obs_tensor = torch.from_numpy(np.asarray(dummy_obs, dtype=np.float32).reshape(1, -1)).to(device)
        
        with torch.no_grad():
            _ = agent.get_action_and_value(dummy_obs_tensor)
        
        dummy_env.close()
        
        # Count parameters (now initialized)
        num_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
        print(f"    Parameters: {num_params:,}")
        
        # Collect gradients
        makespan_grads, energy_grads = collect_gradients_for_architecture(
            agent, mdp_configs, host_specs_file, device,
            num_replicates=3, arch_name=arch_config['name']
        )
        
        # Compute conflicts with different metrics
        arch_results = {
            'num_params': num_params,
            'complexity': arch_config['complexity'],
            'makespan_grads': makespan_grads,
            'energy_grads': energy_grads,
            'metrics': {}
        }
        
        for metric in metrics:
            conflicts = compute_conflicts_with_metric(makespan_grads, energy_grads, metric)
            clustering = evaluate_clustering(conflicts, true_labels)
            
            arch_results['metrics'][metric] = {
                'conflicts': conflicts.tolist(),
                'nmi': clustering['nmi'],
                'ari': clustering['ari'],
                'silhouette': clustering['silhouette'],
                'pred_labels': clustering['pred_labels'].tolist(),
                'mean': float(conflicts.mean()),
                'std': float(conflicts.std()),
                'min': float(conflicts.min()),
                'max': float(conflicts.max())
            }
            
            print(f"    {metric:>12}: NMI={clustering['nmi']:.3f}, ARI={clustering['ari']:.3f}, Sil={clustering['silhouette']:.3f}")
        
        all_results[arch_config['name']] = arch_results
    
    # Analyze results
    print(f"\n[3/4] Analyzing results...")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    results_json = {}
    for arch_name, arch_data in all_results.items():
        results_json[arch_name] = {
            'num_params': arch_data['num_params'],
            'complexity': arch_data['complexity'],
            'metrics': arch_data['metrics']
        }
    
    results_json['mdp_configs'] = mdp_configs
    results_json['true_labels'] = true_labels.tolist()
    
    with open(output_path / "comprehensive_results.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"  ✓ Saved results to {output_dir}")
    
    # Visualizations
    print(f"\n[4/4] Creating visualizations...")
    
    create_comprehensive_visualizations(
        all_results, mdp_configs, true_labels, metrics, architectures, output_path
    )
    
    print(f"  ✓ Saved visualizations to {output_dir}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n## Best Architecture-Metric Combinations (by NMI):\n")
    
    best_combos = []
    for arch_name, arch_data in all_results.items():
        for metric in metrics:
            nmi = arch_data['metrics'][metric]['nmi']
            ari = arch_data['metrics'][metric]['ari']
            best_combos.append((arch_name, metric, nmi, ari))
    
    best_combos.sort(key=lambda x: x[2], reverse=True)
    
    for i, (arch, metric, nmi, ari) in enumerate(best_combos[:10]):
        print(f"  {i+1}. {arch:<30} + {metric:<12} → NMI={nmi:.3f}, ARI={ari:.3f}")
    
    return all_results


def create_comprehensive_visualizations(
    all_results, mdp_configs, true_labels, metrics, architectures, output_path
):
    """Create comprehensive visualizations."""
    
    arch_names = [arch['name'] for arch in architectures]
    edge_probs = [cfg['edge_prob'] for cfg in mdp_configs]
    
    # Figure 1: Clustering Quality Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    quality_metrics = ['nmi', 'ari', 'silhouette']
    
    for idx, quality_metric in enumerate(quality_metrics):
        ax = axes[idx]
        
        # Create matrix: architectures x distance_metrics
        matrix = np.zeros((len(arch_names), len(metrics)))
        
        for i, arch_name in enumerate(arch_names):
            for j, metric in enumerate(metrics):
                matrix[i, j] = all_results[arch_name]['metrics'][metric][quality_metric]
        
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1 if quality_metric != 'silhouette' else 0.5)
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(arch_names)))
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_yticklabels([name[:25] for name in arch_names], fontsize=9)
        ax.set_title(f'{quality_metric.upper()}', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(arch_names)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_path / "clustering_quality_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Conflict Distributions
    fig, axes = plt.subplots(len(metrics), len(arch_names), figsize=(20, 12))
    
    for i, metric in enumerate(metrics):
        for j, arch_name in enumerate(arch_names):
            ax = axes[i, j]
            
            conflicts = np.array(all_results[arch_name]['metrics'][metric]['conflicts'])
            pred_labels = np.array(all_results[arch_name]['metrics'][metric]['pred_labels'])
            
            # Plot conflicts colored by true labels
            colors = ['red' if label == 0 else 'blue' for label in true_labels]
            ax.scatter(edge_probs, conflicts, c=colors, alpha=0.6, s=50)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_ylim(-1, 1)
            
            if i == 0:
                ax.set_title(arch_name[:20], fontsize=9, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{metric}\nConflict', fontsize=9)
            if i == len(metrics) - 1:
                ax.set_xlabel('Edge Prob', fontsize=8)
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path / "conflict_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Gradient PCA Visualization (for largest architecture)
    largest_arch = arch_names[-1]
    makespan_grads = all_results[largest_arch]['makespan_grads']
    energy_grads = all_results[largest_arch]['energy_grads']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA for makespan gradients
    pca = PCA(n_components=2)
    makespan_pca = pca.fit_transform(makespan_grads)
    
    ax = axes[0]
    colors = ['red' if label == 0 else 'blue' for label in true_labels]
    ax.scatter(makespan_pca[:, 0], makespan_pca[:, 1], c=colors, alpha=0.6, s=100)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_title(f'Makespan Gradients PCA\n{largest_arch}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # PCA for energy gradients
    pca = PCA(n_components=2)
    energy_pca = pca.fit_transform(energy_grads)
    
    ax = axes[1]
    ax.scatter(energy_pca[:, 0], energy_pca[:, 1], c=colors, alpha=0.6, s=100)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_title(f'Energy Gradients PCA\n{largest_arch}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Wide (low edge_prob)'),
        Patch(facecolor='blue', alpha=0.6, label='Long CP (high edge_prob)')
    ]
    axes[0].legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / "gradient_pca.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Architecture Complexity vs Performance
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    complexities = [all_results[arch]['complexity'] for arch in arch_names]
    num_params = [all_results[arch]['num_params'] for arch in arch_names]
    
    for idx, quality_metric in enumerate(['nmi', 'ari', 'silhouette']):
        ax = axes[idx]
        
        for metric in metrics:
            scores = [all_results[arch]['metrics'][metric][quality_metric] for arch in arch_names]
            ax.plot(complexities, scores, marker='o', linewidth=2, markersize=8, label=metric)
        
        ax.set_xlabel('Architecture Complexity', fontsize=11)
        ax.set_ylabel(quality_metric.upper(), fontsize=11)
        ax.set_title(f'{quality_metric.upper()} vs Complexity', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(complexities)
    
    plt.tight_layout()
    plt.savefig(output_path / "complexity_vs_performance.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Task Characteristics by Cluster
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Use best architecture-metric combo
    best_arch = arch_names[-1]  # Largest architecture
    best_metric = 'cosine'
    
    pred_labels = np.array(all_results[best_arch]['metrics'][best_metric]['pred_labels'])
    
    task_features = {
        'edge_prob': [cfg['edge_prob'] for cfg in mdp_configs],
        'min_tasks': [cfg['min_tasks'] for cfg in mdp_configs],
        'max_tasks': [cfg['max_tasks'] for cfg in mdp_configs],
        'min_length': [cfg['min_length'] for cfg in mdp_configs],
        'max_length': [cfg['max_length'] for cfg in mdp_configs],
        'task_range': [cfg['max_tasks'] - cfg['min_tasks'] for cfg in mdp_configs]
    }
    
    for idx, (feature_name, feature_values) in enumerate(task_features.items()):
        ax = axes[idx // 3, idx % 3]
        
        # Box plot by cluster
        cluster_0_values = [feature_values[i] for i in range(len(feature_values)) if pred_labels[i] == 0]
        cluster_1_values = [feature_values[i] for i in range(len(feature_values)) if pred_labels[i] == 1]
        
        bp = ax.boxplot([cluster_0_values, cluster_1_values], labels=['Cluster 0', 'Cluster 1'])
        ax.set_ylabel(feature_name.replace('_', ' ').title(), fontsize=10)
        ax.set_title(f'{feature_name.replace("_", " ").title()} by Cluster', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / "task_characteristics_by_cluster.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import sys
    
    host_specs = sys.argv[1] if len(sys.argv) > 1 else "data/host_specs.json"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    results = comprehensive_study(host_specs, num_mdp_samples=num_samples)
