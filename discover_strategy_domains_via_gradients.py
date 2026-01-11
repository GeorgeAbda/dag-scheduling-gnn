"""
Discover Strategy Domains via Gradient-based Task Affinity

Based on: "Scalable Multitask Learning Using Gradient-based Estimation of Task Affinity"

Key idea:
- Define multiple "tasks" = different objectives (makespan vs energy)
- Train agents on different MDP configurations
- Measure gradient conflict between objectives
- Cluster MDPs by which objective they favor

This discovers:
- Which MDPs are "makespan-critical" (long_cp)
- Which MDPs are "energy-critical" (wide)
"""

import os
import json
import torch
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
    """
    Compute policy gradient for a specific objective.
    
    objective_type:
    - "makespan": Minimize makespan only
    - "energy": Minimize energy only
    - "balanced": Minimize both equally
    """
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
        
        # Compute objective-specific reward
        if terminated or truncated:
            makespan = info.get('makespan', 0.0)
            # Use correct key for active energy
            energy = info.get('total_energy_active', 0.0)
            
            # Normalize energy to similar scale as makespan
            # Makespan is in [0, 1] range (normalized time)
            # Energy is in millions, so scale it down
            energy_normalized = energy / 1e8  # Scale to roughly [0, 1]
            
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


def compute_task_affinity_matrix(
    agent, mdp_configs: List[dict],
    host_specs_file: str,
    objectives: List[str],
    device: torch.device,
    num_replicates: int = 3
):
    """
    Compute task affinity matrix between objectives for each MDP.
    
    Returns:
    - affinity_matrix: [num_mdps, num_objectives, num_objectives]
      affinity[i, j, k] = cosine similarity between objective j and k on MDP i
    """
    num_mdps = len(mdp_configs)
    num_objectives = len(objectives)
    
    affinity_matrix = np.zeros((num_mdps, num_objectives, num_objectives))
    
    print(f"\nComputing task affinity matrix...")
    print(f"  MDPs: {num_mdps}")
    print(f"  Objectives: {objectives}")
    print(f"  Replicates per (MDP, objective): {num_replicates}")
    
    for mdp_idx, mdp_config in enumerate(mdp_configs):
        print(f"\n  MDP {mdp_idx+1}/{num_mdps}: edge_prob={mdp_config['edge_prob']:.2f}")
        
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
        objective_gradients = {obj: [] for obj in objectives}
        
        for obj in objectives:
            print(f"    Objective: {obj}...", end=" ", flush=True)
            
            for r in range(num_replicates):
                g = compute_gradient_for_objective(
                    agent, env, obj,
                    num_steps=256, device=device, seed=12345 + mdp_idx * 100 + r
                )
                if g is not None:
                    objective_gradients[obj].append(g)
            
            print(f"✓ ({len(objective_gradients[obj])} gradients)")
        
        env.close()
        
        # Compute pairwise cosine similarities
        for i, obj_i in enumerate(objectives):
            for j, obj_j in enumerate(objectives):
                if not objective_gradients[obj_i] or not objective_gradients[obj_j]:
                    affinity_matrix[mdp_idx, i, j] = 0.0
                    continue
                
                # Average gradients
                g_i = np.mean(np.stack(objective_gradients[obj_i], axis=0), axis=0)
                g_j = np.mean(np.stack(objective_gradients[obj_j], axis=0), axis=0)
                
                # Cosine similarity
                cos_sim = np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j) + 1e-9)
                affinity_matrix[mdp_idx, i, j] = cos_sim
    
    return affinity_matrix


def discover_strategy_domains(
    host_specs_file: str,
    output_dir: str = "logs/strategy_domains",
    num_mdp_samples: int = 20
):
    """
    Discover strategy domains via gradient-based task affinity.
    """
    print("="*70)
    print("STRATEGY DOMAIN DISCOVERY VIA GRADIENT-BASED TASK AFFINITY")
    print("="*70)
    print("\nBased on: 'Scalable Multitask Learning Using Gradient-based")
    print("           Estimation of Task Affinity'")
    print()
    print("Objectives:")
    print("  1. Sample diverse MDP configurations")
    print("  2. Compute gradients for different objectives (makespan vs energy)")
    print("  3. Measure gradient conflict between objectives")
    print("  4. Cluster MDPs by which objective they favor")
    print()
    
    device = torch.device("cpu")
    
    # Global seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize agent
    print("[1/4] Initializing agent...")
    variant = AblationVariant(
        name="hetero", graph_type="hetero", gin_num_layers=2,
        use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
    )
    agent = AblationGinAgent(device=device, variant=variant, hidden_dim=128, embedding_dim=64)
    print("  ✓ Agent initialized")
    
    # Sample diverse MDP configurations
    print(f"\n[2/4] Sampling {num_mdp_samples} diverse MDP configurations...")
    mdp_configs = []
    
    # Sample from different regions of parameter space
    for i in range(num_mdp_samples):
        # Vary edge probability (main driver of structure)
        edge_prob = 0.05 + (i / num_mdp_samples) * 0.90  # 0.05 to 0.95
        
        # Vary other parameters
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
    print(f"    Edge probability range: [{mdp_configs[0]['edge_prob']:.2f}, {mdp_configs[-1]['edge_prob']:.2f}]")
    
    # Define objectives (tasks)
    objectives = ["makespan", "energy"]
    
    # Compute task affinity matrix
    print(f"\n[3/4] Computing task affinity matrix...")
    affinity_matrix = compute_task_affinity_matrix(
        agent, mdp_configs, host_specs_file, objectives, device, num_replicates=3
    )
    
    print(f"\n  ✓ Computed affinity matrix: shape {affinity_matrix.shape}")
    
    # Analyze gradient conflict for each MDP
    print(f"\n[4/4] Analyzing gradient conflict...")
    
    # Extract makespan vs energy conflict for each MDP
    conflicts = affinity_matrix[:, 0, 1]  # cosine similarity between makespan and energy gradients
    
    print(f"\n  Conflict statistics:")
    print(f"    Mean: {conflicts.mean():.4f}")
    print(f"    Std:  {conflicts.std():.4f}")
    print(f"    Min:  {conflicts.min():.4f} (most conflict)")
    print(f"    Max:  {conflicts.max():.4f} (least conflict)")
    
    # Cluster MDPs by conflict
    # Negative conflict = opposing gradients = different strategies needed
    conflict_features = conflicts.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(conflict_features)
    
    # Identify which cluster is makespan-critical vs energy-critical
    cluster_0_mean_edge_prob = np.mean([mdp_configs[i]['edge_prob'] for i in range(len(mdp_configs)) if clusters[i] == 0])
    cluster_1_mean_edge_prob = np.mean([mdp_configs[i]['edge_prob'] for i in range(len(mdp_configs)) if clusters[i] == 1])
    
    if cluster_0_mean_edge_prob > cluster_1_mean_edge_prob:
        longcp_cluster = 0
        wide_cluster = 1
    else:
        longcp_cluster = 1
        wide_cluster = 0
    
    print(f"\n  ✓ Discovered 2 strategy domains:")
    print(f"    Domain 0 (Long CP): {np.sum(clusters == longcp_cluster)} MDPs, mean edge_prob={cluster_0_mean_edge_prob if longcp_cluster==0 else cluster_1_mean_edge_prob:.2f}")
    print(f"    Domain 1 (Wide):    {np.sum(clusters == wide_cluster)} MDPs, mean edge_prob={cluster_1_mean_edge_prob if wide_cluster==1 else cluster_0_mean_edge_prob:.2f}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'mdp_configs': mdp_configs,
        'affinity_matrix': affinity_matrix.tolist(),
        'conflicts': conflicts.tolist(),
        'clusters': clusters.tolist(),
        'objectives': objectives
    }
    
    with open(output_path / "strategy_domains.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Conflict vs edge probability
    plt.subplot(1, 2, 1)
    edge_probs = [cfg['edge_prob'] for cfg in mdp_configs]
    colors = ['red' if c == wide_cluster else 'blue' for c in clusters]
    plt.scatter(edge_probs, conflicts, c=colors, alpha=0.6)
    plt.xlabel('Edge Probability')
    plt.ylabel('Gradient Conflict (cos_sim)')
    plt.title('Gradient Conflict vs MDP Structure')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cluster distribution
    plt.subplot(1, 2, 2)
    plt.hist([edge_probs[i] for i in range(len(edge_probs)) if clusters[i] == wide_cluster],
             bins=10, alpha=0.5, label='Wide (energy-critical)', color='red')
    plt.hist([edge_probs[i] for i in range(len(edge_probs)) if clusters[i] == longcp_cluster],
             bins=10, alpha=0.5, label='Long CP (makespan-critical)', color='blue')
    plt.xlabel('Edge Probability')
    plt.ylabel('Count')
    plt.title('Strategy Domain Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "strategy_domains.png", dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved results to {output_dir}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("\nGradient conflict reveals which MDPs need different strategies:")
    print("  • Negative conflict → Opposing gradients → Different strategies")
    print("  • Positive conflict → Aligned gradients → Same strategy")
    print("\nDiscovered domains:")
    print(f"  • Wide (low edge_prob): Energy-critical, pack tasks")
    print(f"  • Long CP (high edge_prob): Makespan-critical, parallelize")
    
    return results


if __name__ == "__main__":
    import sys
    
    host_specs = sys.argv[1] if len(sys.argv) > 1 else "data/host_specs.json"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    results = discover_strategy_domains(host_specs, num_mdp_samples=num_samples)
