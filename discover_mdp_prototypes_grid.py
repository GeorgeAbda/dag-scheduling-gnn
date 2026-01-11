"""
Simplified Adversarial MDP Discovery: Grid Search + Random Sampling

Instead of differential evolution, use a simpler approach:
1. Grid search over edge probability (main driver of conflict)
2. Random sampling for other parameters
3. Evaluate all pairs and pick the most conflicting one
"""

import os
import json
import torch
import numpy as np
import random
from pathlib import Path
from itertools import product

from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.dataset_generator.gen_dataset import DatasetArgs


def create_mdp_from_params(params, host_specs_file):
    """
    Create MDP environment from parameters.
    
    params: dict with keys: edge_prob, min_tasks, max_tasks, min_length, max_length
    """
    edge_prob = params['edge_prob']
    min_tasks = params['min_tasks']
    max_tasks = params['max_tasks']
    min_length = params['min_length']
    max_length = params['max_length']
    
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


def compute_conflict(agent, params1, params2, host_specs_file, device, num_replicates=3):
    """
    Compute gradient conflict between two MDP configurations.
    Returns cosine similarity (more negative = more conflict).
    """
    try:
        # Create MDPs
        env1, args1 = create_mdp_from_params(params1, host_specs_file)
        env2, args2 = create_mdp_from_params(params2, host_specs_file)
        
        # Compute gradients with replication for stability
        grads1 = []
        grads2 = []
        base_seed1 = 12345
        base_seed2 = 67890
        
        for r in range(num_replicates):
            g1 = compute_gradient(agent, env1, num_steps=256, device=device, seed=base_seed1 + r)
            g2 = compute_gradient(agent, env2, num_steps=256, device=device, seed=base_seed2 + r)
            if g1 is not None:
                grads1.append(g1)
            if g2 is not None:
                grads2.append(g2)
        
        env1.close()
        env2.close()
        
        if not grads1 or not grads2:
            return 0.0, args1, args2
        
        g1_mean = np.mean(np.stack(grads1, axis=0), axis=0)
        g2_mean = np.mean(np.stack(grads2, axis=0), axis=0)
        
        # Cosine similarity
        cos_sim = np.dot(g1_mean, g2_mean) / (np.linalg.norm(g1_mean) * np.linalg.norm(g2_mean) + 1e-9)
        
        return cos_sim, args1, args2
    
    except Exception as e:
        print(f"Error computing conflict: {e}")
        return 0.0, None, None


def discover_adversarial_mdp_pair_grid(
    host_specs_file: str,
    output_dir: str = "logs/mdp_discovery",
    num_candidates: int = 20
):
    """
    Use grid search + random sampling to discover adversarial MDP pairs.
    
    Strategy:
    1. Grid over edge probabilities (main conflict driver)
    2. Sample random task/length configs
    3. Evaluate all pairs, keep most conflicting
    """
    print("="*70)
    print("ADVERSARIAL MDP DISCOVERY (Grid Search)")
    print("="*70)
    print("Goal: Find two MDP configurations that cause maximum gradient conflict")
    print(f"Candidates: {num_candidates} per edge probability")
    print()
    
    device = torch.device("cpu")
    
    # Global seeds for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Initialize agent
    print("[1/4] Initializing agent...")
    variant = AblationVariant(
        name="hetero", graph_type="hetero", gin_num_layers=2,
        use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
    )
    agent = AblationGinAgent(device=device, variant=variant, hidden_dim=128, embedding_dim=64)
    print("  ✓ Agent initialized")
    
    # Define edge probability grid (main conflict driver)
    edge_probs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    
    print(f"\n[2/4] Generating {num_candidates} candidate configs per edge probability...")
    
    # Generate candidate configs
    candidates = []
    for edge_prob in edge_probs:
        for _ in range(num_candidates):
            # Random sampling for other parameters
            min_tasks = np.random.randint(8, 20)
            max_tasks = np.random.randint(min_tasks, 30)
            min_length = np.random.randint(500, 5000)
            max_length = np.random.randint(max(min_length, 10000), 200000)
            
            config = {
                'edge_prob': edge_prob,
                'min_tasks': min_tasks,
                'max_tasks': max_tasks,
                'min_length': min_length,
                'max_length': max_length
            }
            candidates.append(config)
    
    print(f"  ✓ Generated {len(candidates)} candidate configs")
    
    # Evaluate pairs focusing on extreme edge probabilities
    print(f"\n[3/4] Evaluating pairs (focusing on extreme edge probs)...")
    
    # Strategy: pair low edge_prob with high edge_prob
    low_edge_candidates = [c for c in candidates if c['edge_prob'] <= 0.25]
    high_edge_candidates = [c for c in candidates if c['edge_prob'] >= 0.75]
    
    print(f"  Low edge prob configs: {len(low_edge_candidates)}")
    print(f"  High edge prob configs: {len(high_edge_candidates)}")
    
    best_conflict = 1.0  # Start with max (no conflict)
    best_pair = None
    best_args = (None, None)
    
    # Sample pairs to evaluate (not all combinations, too expensive)
    num_pairs_to_eval = min(50, len(low_edge_candidates) * len(high_edge_candidates))
    print(f"  Evaluating {num_pairs_to_eval} pairs...")
    
    evaluated = 0
    for i in range(num_pairs_to_eval):
        # Sample random pair
        c1 = random.choice(low_edge_candidates)
        c2 = random.choice(high_edge_candidates)
        
        print(f"    Pair {i+1}/{num_pairs_to_eval}: edge_prob {c1['edge_prob']:.2f} vs {c2['edge_prob']:.2f}...", end=" ")
        
        conflict, args1, args2 = compute_conflict(agent, c1, c2, host_specs_file, device, num_replicates=2)
        
        print(f"conflict={conflict:+.4f}")
        
        if conflict < best_conflict:
            best_conflict = conflict
            best_pair = (c1, c2)
            best_args = (args1, args2)
            print(f"      → New best! conflict={best_conflict:+.4f}")
        
        evaluated += 1
    
    print(f"\n[4/4] Best pair found after {evaluated} evaluations!")
    print(f"  Gradient conflict (cosine similarity): {best_conflict:.4f}")
    
    if best_pair is None:
        print("  ✗ No valid pair found")
        return None, None, None
    
    params1, params2 = best_pair
    args1, args2 = best_args
    
    # Create final configs
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
        "comment": f"Discovered via grid search (conflict={best_conflict:.4f})"
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
        "comment": f"Discovered via grid search (conflict={best_conflict:.4f})"
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
    
    print(f"\nMDP 1 ({args1.style}):")
    print(f"  Edge Probability: {args1.gnp_p:.3f}")
    print(f"  Tasks: {args1.gnp_min_n}-{args1.gnp_max_n}")
    print(f"  Task Length: {args1.min_task_length}-{args1.max_task_length}")
    
    print(f"\nMDP 2 ({args2.style}):")
    print(f"  Edge Probability: {args2.gnp_p:.3f}")
    print(f"  Tasks: {args2.gnp_min_n}-{args2.gnp_max_n}")
    print(f"  Task Length: {args2.min_task_length}-{args2.max_task_length}")
    
    print(f"\nGradient Conflict (cosine similarity): {best_conflict:.4f}")
    
    if best_conflict < -0.8:
        print("  → SEVERE CONFLICT: These MDPs require separate policies")
    elif best_conflict < -0.3:
        print("  → MODERATE CONFLICT: Consider task-conditioned policy")
    else:
        print("  → LOW CONFLICT: Single policy may work")
    
    return config1, config2, best_conflict


if __name__ == "__main__":
    import sys
    
    host_specs = sys.argv[1] if len(sys.argv) > 1 else "data/host_specs.json"
    
    config1, config2, conflict = discover_adversarial_mdp_pair_grid(
        host_specs_file=host_specs,
        output_dir="logs/mdp_discovery",
        num_candidates=5  # 5 candidates per edge_prob = 35 total configs
    )
    
    if config1 is not None:
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Use discovered configs for training:")
        print("   python discover_domains_improved.py \\")
        print("     logs/mdp_discovery/discovered_mdp1.json \\")
        print("     logs/mdp_discovery/discovered_mdp2.json \\")
        print(f"     {host_specs}")
        print("\n2. Compare with manually designed configs (long_cp vs wide)")
        print("\n3. Run multiple times to find diverse prototypes")
