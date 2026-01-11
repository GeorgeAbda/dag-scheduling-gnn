"""
Inverse Domain Discovery: Train to FIND MDP configurations that cause gradient conflict.

Instead of training agents on fixed MDPs, we search for MDP configurations that:
1. Maximize gradient conflict (adversarial pairs)
2. Form natural clusters (domain prototypes)
3. Span the space of possible scheduling problems

This is "reasoning by absurdity" - we assume domains exist and search for them.
"""

import os
import json
import torch
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import differential_evolution
from sklearn.cluster import AgglomerativeClustering

from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.dataset_generator.gen_dataset import DatasetArgs


def create_mdp_from_params(params, host_specs_file):
    """
    Create MDP environment from continuous parameters.
    
    params: [edge_prob, min_tasks, max_tasks, min_length, max_length]
    """
    edge_prob = np.clip(params[0], 0.01, 0.99)
    min_tasks = int(np.clip(params[1], 5, 50))
    max_tasks = int(np.clip(params[2], min_tasks, 50))
    min_length = int(np.clip(params[3], 100, 10000))
    max_length = int(np.clip(params[4], min_length, 200000))
    
    # Determine style based on edge probability
    style = "long_cp" if edge_prob > 0.5 else "wide"
    
    args = DatasetArgs(
        host_count=10,
        vm_count=10,
        workflow_count=1,
        style=style,
        gnp_p=edge_prob,
        gnp_min_n=min_tasks,
        gnp_max_n=max_tasks,
        min_task_length=min_length,
        max_task_length=max_length,
    )
    
    os.environ['HOST_SPECS_PATH'] = host_specs_file
    
    env = GinAgentWrapper(CloudSchedulingGymEnvironment(
        dataset_args=args, collect_timelines=False, compute_metrics=False
    ))
    
    return env, args


def compute_gradient(agent, env, num_steps=256, device=torch.device("cpu"), seed=None):
    """Compute policy gradient for given environment."""
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
        rewards.append(reward)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
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


def objective_maximize_conflict(params_flat, agent, host_specs_file, device):
    """
    Objective function: Find two MDP configs that maximize gradient conflict.
    
    params_flat: [mdp1_params (5), mdp2_params (5)] = 10 total
    Returns: cosine similarity (minimize to maximize conflict)
    """
    params1 = params_flat[:5]
    params2 = params_flat[5:]
    
    try:
        # Create MDPs
        env1, args1 = create_mdp_from_params(params1, host_specs_file)
        env2, args2 = create_mdp_from_params(params2, host_specs_file)
        
        # Compute gradients with seeding and replication for stability
        K = 3
        grads1 = []
        grads2 = []
        base_seed1 = 12345
        base_seed2 = 67890
        for r in range(K):
            g1 = compute_gradient(agent, env1, num_steps=256, device=device, seed=base_seed1 + r)
            g2 = compute_gradient(agent, env2, num_steps=256, device=device, seed=base_seed2 + r)
            if g1 is not None:
                grads1.append(g1)
            if g2 is not None:
                grads2.append(g2)
        env1.close()
        env2.close()
        
        if not grads1 or not grads2:
            return 0.0
        
        g1_mean = np.mean(np.stack(grads1, axis=0), axis=0)
        g2_mean = np.mean(np.stack(grads2, axis=0), axis=0)
        
        # Cosine similarity
        cos_sim = np.dot(g1_mean, g2_mean) / (np.linalg.norm(g1_mean) * np.linalg.norm(g2_mean) + 1e-9)
        
        # Minimize cosine similarity to maximize conflict (target ≈ -1)
        return cos_sim
    
    except Exception as e:
        print(f"Error in objective: {e}")
        return 0.0


def discover_adversarial_mdp_pair(
    host_specs_file: str,
    output_dir: str = "logs/mdp_discovery",
    max_iterations: int = 100
):
    """
    Use optimization to discover MDP configurations that maximize gradient conflict.
    """
    print("="*70)
    print("ADVERSARIAL MDP DISCOVERY")
    print("="*70)
    print("Goal: Find two MDP configurations that cause maximum gradient conflict")
    print()
    
    device = torch.device("cpu")
    # Global seeds for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Initialize agent
    print("[1/3] Initializing agent...")
    variant = AblationVariant(
        name="hetero", graph_type="hetero", gin_num_layers=2,
        use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
    )
    agent = AblationGinAgent(device=device, variant=variant, hidden_dim=128, embedding_dim=64)
    print("  ✓ Agent initialized")
    
    # Define search bounds
    # [edge_prob, min_tasks, max_tasks, min_length, max_length] x 2 MDPs
    bounds = [
        (0.01, 0.99),  # MDP1 edge_prob
        (5, 50),       # MDP1 min_tasks
        (5, 50),       # MDP1 max_tasks
        (100, 10000),  # MDP1 min_length
        (1000, 200000),# MDP1 max_length
        (0.01, 0.99),  # MDP2 edge_prob
        (5, 50),       # MDP2 min_tasks
        (5, 50),       # MDP2 max_tasks
        (100, 10000),  # MDP2 min_length
        (1000, 200000),# MDP2 max_length
    ]
    
    # Initialization: seed population with known extreme pairs (long_cp vs wide)
    popsize = 5
    dim = len(bounds)
    pop_n = popsize * dim
    
    seeds = []
    # Seed 1: extreme long vs extreme wide
    seeds.append(np.array([0.95, 10, 10, 750, 150000, 0.05, 10, 10, 750, 150000], dtype=float))
    # Seed 2: swapped edge probs with slightly different task ranges
    seeds.append(np.array([0.90, 12, 24, 1000, 180000, 0.10, 12, 24, 1000, 180000], dtype=float))
    # Seed 3: moderate long vs very wide
    seeds.append(np.array([0.75, 15, 30, 500, 120000, 0.02, 15, 30, 500, 120000], dtype=float))
    # Seed 4: very long vs moderate wide
    seeds.append(np.array([0.99, 8, 16, 2000, 200000, 0.25, 8, 16, 2000, 200000], dtype=float))
    # Seed 5: near threshold split
    seeds.append(np.array([0.60, 10, 20, 1000, 150000, 0.40, 10, 20, 1000, 150000], dtype=float))
    
    initial_pop = np.zeros((pop_n, dim), dtype=float)
    # Place seeds first
    for i, s in enumerate(seeds):
        initial_pop[i, :] = s
    # Fill the rest with uniform random within bounds
    rng = np.random.default_rng(42)
    for i in range(len(seeds), pop_n):
        for j, (lo, hi) in enumerate(bounds):
            initial_pop[i, j] = rng.uniform(lo, hi)
    
    print(f"\n[2/3] Searching for adversarial MDP pair (max {max_iterations} iterations)...")
    print("  This will take several minutes...")
    
    # Use differential evolution to search
    result = differential_evolution(
        objective_maximize_conflict,
        bounds,
        args=(agent, host_specs_file, device),
        maxiter=max_iterations,
        popsize=popsize,
        init=initial_pop,
        seed=42,
        workers=1,
        updating='deferred',
        disp=True
    )
    
    # Extract best parameters
    best_params = result.x
    best_conflict = result.fun  # Already the cosine similarity (minimized)
    
    params1 = best_params[:5]
    params2 = best_params[5:]
    
    print(f"\n[3/3] Best MDP pair found!")
    print(f"  Gradient conflict (cosine similarity): {best_conflict:.4f}")
    
    # Create final configs
    env1, args1 = create_mdp_from_params(params1, host_specs_file)
    env2, args2 = create_mdp_from_params(params2, host_specs_file)
    
    config1 = {
        "training_seeds": [101001],
        "all_seeds": [101001],
        "dataset": {
            "style": args1.style,
            "edge_probability": float(args1.gnp_p),
            "min_tasks": int(args1.gnp_min_n),
            "max_tasks": int(args1.gnp_max_n),
            "hosts": 10,
            "vms": 10,
            "workflow_count": 1,
            "task_length": {
                "distribution": "normal",
                "min": int(args1.min_task_length),
                "max": int(args1.max_task_length)
            }
        },
        "comment": f"Discovered via adversarial search (conflict={best_conflict:.4f})"
    }
    
    config2 = {
        "training_seeds": [202002],
        "all_seeds": [202002],
        "dataset": {
            "style": args2.style,
            "edge_probability": float(args2.gnp_p),
            "min_tasks": int(args2.gnp_min_n),
            "max_tasks": int(args2.gnp_max_n),
            "hosts": 10,
            "vms": 10,
            "workflow_count": 1,
            "task_length": {
                "distribution": "normal",
                "min": int(args2.min_task_length),
                "max": int(args2.max_task_length)
            }
        },
        "comment": f"Discovered via adversarial search (conflict={best_conflict:.4f})"
    }
    
    # Save configs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config1_file = output_path / "discovered_mdp1.json"
    config2_file = output_path / "discovered_mdp2.json"
    
    with open(config1_file, 'w') as f:
        json.dump(config1, f, indent=2)
    
    with open(config2_file, 'w') as f:
        json.dump(config2, f, indent=2)
    
    print(f"\n  ✓ Saved MDP1 config: {config1_file}")
    print(f"  ✓ Saved MDP2 config: {config2_file}")
    
    print("\n" + "="*70)
    print("DISCOVERED MDP CONFIGURATIONS")
    print("="*70)
    
    print(f"\nMDP 1:")
    print(f"  Style: {args1.style}")
    print(f"  Edge Probability: {args1.gnp_p:.3f}")
    print(f"  Tasks: {args1.gnp_min_n}-{args1.gnp_max_n}")
    print(f"  Task Length: {args1.min_task_length}-{args1.max_task_length}")
    
    print(f"\nMDP 2:")
    print(f"  Style: {args2.style}")
    print(f"  Edge Probability: {args2.gnp_p:.3f}")
    print(f"  Tasks: {args2.gnp_min_n}-{args2.gnp_max_n}")
    print(f"  Task Length: {args2.min_task_length}-{args2.max_task_length}")
    
    print(f"\nGradient Conflict: {best_conflict:.4f}")
    
    if best_conflict < -0.8:
        print("  → SEVERE CONFLICT: These MDPs require separate policies")
    elif best_conflict < -0.3:
        print("  → MODERATE CONFLICT: Consider task-conditioned policy")
    else:
        print("  → LOW CONFLICT: Single policy may work")
    
    env1.close()
    env2.close()
    
    return config1, config2, best_conflict


if __name__ == "__main__":
    import sys
    
    host_specs = sys.argv[1] if len(sys.argv) > 1 else "data/host_specs.json"
    
    config1, config2, conflict = discover_adversarial_mdp_pair(
        host_specs_file=host_specs,
        output_dir="logs/mdp_discovery",
        max_iterations=10  # Reduce for faster testing
    )
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Use discovered configs for training:")
    print("   python discover_domains_improved.py \\")
    print("     logs/mdp_discovery/discovered_mdp1.json \\")
    print("     logs/mdp_discovery/discovered_mdp2.json \\")
    print(f"     {host_specs}")
    print("\n2. Compare with manually designed configs (long_cp vs wide)")
    print("\n3. Iterate: Run discovery multiple times to find diverse prototypes")
