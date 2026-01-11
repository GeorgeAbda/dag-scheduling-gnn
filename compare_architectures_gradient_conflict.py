"""
Compare gradient-based task affinity across different agent architectures.

Tests whether discovered strategy domains are:
- Architecture-invariant (same domains across all architectures)
- Architecture-specific (different architectures find different domains)
"""

import os
import json
import torch
import numpy as np
import random
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

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
            elif objective_type == "balanced":
                episode_reward = -makespan - energy_normalized
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


def compute_conflicts_for_architecture(
    agent, mdp_configs: List[dict],
    host_specs_file: str,
    device: torch.device,
    num_replicates: int = 3,
    arch_name: str = ""
):
    """Compute gradient conflicts for a single architecture."""
    objectives = ["makespan", "energy"]
    num_mdps = len(mdp_configs)
    conflicts = []
    
    pbar = tqdm(mdp_configs, desc=f"    {arch_name}", ncols=100)
    
    for mdp_idx, mdp_config in enumerate(pbar):
        # Create environment
        params = [
            mdp_config['edge_prob'],
            mdp_config['min_tasks'],
            mdp_config['max_tasks'],
            mdp_config['min_length'],
            mdp_config['max_length']
        ]
        env, _ = create_mdp_from_params(params, host_specs_file)
        
        # Collect gradients for each objective
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
        
        # Compute conflict
        if makespan_grads and energy_grads:
            g_m_mean = np.mean(np.stack(makespan_grads, axis=0), axis=0)
            g_e_mean = np.mean(np.stack(energy_grads, axis=0), axis=0)
            
            cos_sim = np.dot(g_m_mean, g_e_mean) / (
                np.linalg.norm(g_m_mean) * np.linalg.norm(g_e_mean) + 1e-9
            )
            conflicts.append(cos_sim)
        else:
            conflicts.append(0.0)
        
        # Update progress bar with current stats
        if conflicts:
            pbar.set_postfix({
                'conflict': f'{conflicts[-1]:.3f}',
                'mean': f'{np.mean(conflicts):.3f}'
            })
    
    return np.array(conflicts)


def compare_architectures(
    host_specs_file: str,
    output_dir: str = "logs/architecture_comparison",
    num_mdp_samples: int = 20
):
    """Compare gradient-based task affinity across different architectures."""
    print("="*70)
    print("ARCHITECTURE COMPARISON: GRADIENT-BASED TASK AFFINITY")
    print("="*70)
    print("\nObjective: Test if discovered domains are architecture-invariant")
    print()
    
    device = torch.device("cpu")
    
    # Global seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Define architectures to test
    architectures = [
        {
            'name': 'Hetero GNN (2 layers)',
            'variant': AblationVariant(
                name="hetero", graph_type="hetero", gin_num_layers=2,
                use_batchnorm=True, use_task_dependencies=True,
                use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
            ),
            'hidden_dim': 128,
            'embedding_dim': 64
        },
        {
            'name': 'Hetero GNN (3 layers)',
            'variant': AblationVariant(
                name="hetero_deep", graph_type="hetero", gin_num_layers=3,
                use_batchnorm=True, use_task_dependencies=True,
                use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
            ),
            'hidden_dim': 128,
            'embedding_dim': 64
        },
        {
            'name': 'Hetero GNN (small)',
            'variant': AblationVariant(
                name="hetero_small", graph_type="hetero", gin_num_layers=2,
                use_batchnorm=True, use_task_dependencies=True,
                use_actor_global_embedding=True, mlp_only=False, gat_heads=2,
            ),
            'hidden_dim': 64,
            'embedding_dim': 32
        },
        {
            'name': 'Hetero GNN (no BatchNorm)',
            'variant': AblationVariant(
                name="hetero_nobn", graph_type="hetero", gin_num_layers=2,
                use_batchnorm=False, use_task_dependencies=True,
                use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
            ),
            'hidden_dim': 128,
            'embedding_dim': 64
        },
        {
            'name': 'Homo GNN',
            'variant': AblationVariant(
                name="homo", graph_type="homo", gin_num_layers=2,
                use_batchnorm=True, use_task_dependencies=True,
                use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
            ),
            'hidden_dim': 128,
            'embedding_dim': 64
        },
    ]
    
    # Sample diverse MDP configurations (same as before)
    print(f"[1/3] Sampling {num_mdp_samples} diverse MDP configurations...")
    mdp_configs = []
    
    for i in range(num_mdp_samples):
        edge_prob = 0.05 + (i / num_mdp_samples) * 0.90
        min_tasks = random.randint(10, 30)
        max_tasks = random.randint(min_tasks, 40)
        min_length = random.randint(100, 5000)
        max_length = random.randint(max(min_length, 10000), 150000)
        
        mdp_configs.append({
            'edge_prob': edge_prob,
            'min_tasks': min_tasks,
            'max_tasks': max_tasks,
            'min_length': min_length,
            'max_length': max_length
        })
    
    print(f"  ✓ Sampled {len(mdp_configs)} MDP configurations")
    
    # Compute conflicts for each architecture
    print(f"\n[2/3] Computing gradient conflicts for each architecture...")
    
    results = {}
    
    for arch_idx, arch_config in enumerate(architectures):
        print(f"\n  [{arch_idx+1}/{len(architectures)}] {arch_config['name']}")
        
        # Initialize agent
        agent = AblationGinAgent(
            device=device,
            variant=arch_config['variant'],
            hidden_dim=arch_config['hidden_dim'],
            embedding_dim=arch_config['embedding_dim']
        )
        
        # Compute conflicts with progress bar
        conflicts = compute_conflicts_for_architecture(
            agent, mdp_configs, host_specs_file, device, 
            num_replicates=3, arch_name=arch_config['name']
        )
        
        print(f"    ✓ Conflict: mean={conflicts.mean():.4f}, std={conflicts.std():.4f}, range=[{conflicts.min():.4f}, {conflicts.max():.4f}]")
        
        # Cluster
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(conflicts.reshape(-1, 1))
        
        results[arch_config['name']] = {
            'conflicts': conflicts.tolist(),
            'clusters': clusters.tolist(),
            'stats': {
                'mean': float(conflicts.mean()),
                'std': float(conflicts.std()),
                'min': float(conflicts.min()),
                'max': float(conflicts.max())
            }
        }
    
    # Analyze cross-architecture consistency
    print(f"\n[3/3] Analyzing cross-architecture consistency...")
    
    # Compute pairwise cluster agreement
    arch_names = list(results.keys())
    num_archs = len(arch_names)
    
    agreement_matrix = np.zeros((num_archs, num_archs))
    
    for i, arch_i in enumerate(arch_names):
        for j, arch_j in enumerate(arch_names):
            clusters_i = np.array(results[arch_i]['clusters'])
            clusters_j = np.array(results[arch_j]['clusters'])
            
            # Compute agreement (handle label permutation)
            agreement_direct = np.mean(clusters_i == clusters_j)
            agreement_flipped = np.mean(clusters_i == (1 - clusters_j))
            agreement = max(agreement_direct, agreement_flipped)
            
            agreement_matrix[i, j] = agreement
    
    print(f"\n  Cluster Agreement Matrix:")
    print(f"  {'Architecture':<30}", end="")
    for arch in arch_names:
        print(f"{arch[:10]:<12}", end="")
    print()
    
    for i, arch_i in enumerate(arch_names):
        print(f"  {arch_i:<30}", end="")
        for j in range(num_archs):
            print(f"{agreement_matrix[i, j]:.2f}        ", end="")
        print()
    
    mean_agreement = np.mean(agreement_matrix[np.triu_indices(num_archs, k=1)])
    print(f"\n  Mean pairwise agreement: {mean_agreement:.4f}")
    
    if mean_agreement > 0.8:
        print("  ✓ HIGH AGREEMENT: Domains are architecture-invariant")
    elif mean_agreement > 0.6:
        print("  ~ MODERATE AGREEMENT: Some architecture dependence")
    else:
        print("  ✗ LOW AGREEMENT: Domains are architecture-specific")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results['mdp_configs'] = mdp_configs
    results['agreement_matrix'] = agreement_matrix.tolist()
    results['mean_agreement'] = float(mean_agreement)
    
    with open(output_path / "architecture_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    edge_probs = [cfg['edge_prob'] for cfg in mdp_configs]
    
    for idx, arch_name in enumerate(arch_names):
        ax = axes[idx]
        conflicts = np.array(results[arch_name]['conflicts'])
        clusters = np.array(results[arch_name]['clusters'])
        
        colors = ['red' if c == 0 else 'blue' for c in clusters]
        ax.scatter(edge_probs, conflicts, c=colors, alpha=0.6, s=80)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Edge Probability', fontsize=10)
        ax.set_ylabel('Gradient Conflict', fontsize=10)
        ax.set_title(arch_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.6, 0.6)
    
    # Last subplot: agreement matrix heatmap
    ax = axes[-1]
    im = ax.imshow(agreement_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(num_archs))
    ax.set_yticks(range(num_archs))
    ax.set_xticklabels([name[:15] for name in arch_names], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([name[:15] for name in arch_names], fontsize=8)
    ax.set_title('Cluster Agreement Matrix', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(num_archs):
        for j in range(num_archs):
            text = ax.text(j, i, f'{agreement_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_path / "architecture_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved results to {output_dir}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTested {len(architectures)} architectures on {num_mdp_samples} MDPs")
    print(f"Mean cluster agreement: {mean_agreement:.4f}")
    print("\nConflict statistics by architecture:")
    for arch_name in arch_names:
        stats = results[arch_name]['stats']
        print(f"  {arch_name:<30} mean={stats['mean']:>7.4f}, std={stats['std']:>6.4f}, range=[{stats['min']:>7.4f}, {stats['max']:>6.4f}]")
    
    return results


if __name__ == "__main__":
    import sys
    
    host_specs = sys.argv[1] if len(sys.argv) > 1 else "data/host_specs.json"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    results = compare_architectures(host_specs, num_mdp_samples=num_samples)
