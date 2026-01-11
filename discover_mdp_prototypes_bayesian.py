"""
Bayesian Optimization for Adversarial MDP Discovery

Instead of training a neural network surrogate, use Bayesian Optimization (BO)
to directly optimize the true gradient conflict function.

BO is sample-efficient and handles noisy objectives well.
"""

import os
import json
import torch
import numpy as np
import random
from pathlib import Path

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
except ImportError:
    print("ERROR: scikit-optimize not installed. Install with:")
    print("  pip install scikit-optimize")
    exit(1)

from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.dataset_generator.gen_dataset import DatasetArgs


def create_mdp_from_params(params, host_specs_file):
    """Create MDP environment from parameters."""
    edge_prob = float(params[0])
    min_tasks = int(params[1])
    max_tasks = int(params[2])
    min_length = int(params[3])
    max_length = int(params[4])
    
    # Ensure constraints
    max_tasks = max(min_tasks, max_tasks)
    max_length = max(min_length, max_length)
    
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


def compute_conflict(agent, params_flat, host_specs_file, device, num_replicates=5):
    """
    Compute gradient conflict between two MDP configurations.
    Returns cosine similarity (more negative = more conflict).
    
    params_flat: [edge1, min_t1, max_t1, min_l1, max_l1, edge2, min_t2, max_t2, min_l2, max_l2]
    """
    params1 = params_flat[:5]
    params2 = params_flat[5:]
    
    try:
        env1, _ = create_mdp_from_params(params1, host_specs_file)
        env2, _ = create_mdp_from_params(params2, host_specs_file)
        
        grads1 = []
        grads2 = []
        base_seed1 = 12345
        base_seed2 = 67890
        
        for r in range(num_replicates):
            g1 = compute_gradient(agent, env1, num_steps=512, device=device, seed=base_seed1 + r)
            g2 = compute_gradient(agent, env2, num_steps=512, device=device, seed=base_seed2 + r)
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
        
        cos_sim = np.dot(g1_mean, g2_mean) / (np.linalg.norm(g1_mean) * np.linalg.norm(g2_mean) + 1e-9)
        
        return cos_sim
    
    except Exception as e:
        print(f"Error: {e}")
        return 0.0


def discover_adversarial_mdp_pair_bayesian(
    host_specs_file: str,
    output_dir: str = "logs/mdp_discovery",
    n_calls: int = 100
):
    """
    Use Bayesian Optimization to discover adversarial MDP pairs.
    
    BO is sample-efficient: finds good solutions with ~100 evaluations.
    """
    print("="*70)
    print("ADVERSARIAL MDP DISCOVERY (Bayesian Optimization)")
    print("="*70)
    print("Strategy:")
    print("  1. Use Gaussian Process to model conflict landscape")
    print("  2. Acquisition function balances exploration vs exploitation")
    print("  3. Directly optimize true conflict (no surrogate needed)")
    print(f"  4. Budget: {n_calls} evaluations")
    print()
    
    device = torch.device("cpu")
    
    # Global seeds
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
    
    # Define search space
    print("\n[2/3] Setting up Bayesian Optimization...")
    space = [
        Real(0.01, 0.99, name='edge_prob1'),
        Integer(5, 50, name='min_tasks1'),
        Integer(5, 50, name='max_tasks1'),
        Integer(100, 10000, name='min_length1'),
        Integer(1000, 200000, name='max_length1'),
        Real(0.01, 0.99, name='edge_prob2'),
        Integer(5, 50, name='min_tasks2'),
        Integer(5, 50, name='max_tasks2'),
        Integer(100, 10000, name='min_length2'),
        Integer(1000, 200000, name='max_length2'),
    ]
    
    # Objective function
    @use_named_args(space)
    def objective(**params):
        params_flat = [
            params['edge_prob1'], params['min_tasks1'], params['max_tasks1'],
            params['min_length1'], params['max_length1'],
            params['edge_prob2'], params['min_tasks2'], params['max_tasks2'],
            params['min_length2'], params['max_length2']
        ]
        
        conflict = compute_conflict(agent, params_flat, host_specs_file, device, num_replicates=3)
        
        # Minimize cosine similarity (maximize conflict)
        return conflict
    
    # Initial points: theoretically-motivated seeds
    # Theory: Gradient conflict maximized when:
    # 1. Edge probability extremes (wide vs long_cp)
    # 2. Task count affects parallelism vs sequentiality
    # 3. Task length affects makespan vs energy trade-off
    # 4. Combinations that stress different scheduling objectives
    # IMPORTANT: Ensure min <= max for tasks and lengths
    
    x0 = [
        # Seed 1: Extreme wide (max parallelism) vs extreme long_cp (max sequentiality)
        # Wide: low edge, many tasks, short tasks → energy-critical
        # Long: high edge, few tasks, long tasks → makespan-critical
        [0.01, 30, 50, 100, 10000, 0.99, 5, 10, 8000, 200000],
        
        # Seed 2: Wide with long tasks vs long_cp with short tasks
        # Tests if task length reversal creates conflict
        [0.05, 25, 40, 8000, 200000, 0.95, 8, 15, 1000, 10000],
        
        # Seed 3: Moderate wide (high parallelism) vs moderate long (high sequentiality)
        # More realistic scenarios
        [0.10, 20, 35, 1000, 100000, 0.90, 10, 20, 5000, 150000],
        
        # Seed 4: Very sparse vs very dense
        # Extreme edge probability with balanced other params
        [0.02, 15, 25, 5000, 120000, 0.98, 15, 25, 5000, 120000],
        
        # Seed 5: Wide with few long tasks vs long_cp with many short tasks
        # Tests task count vs length interaction
        [0.05, 5, 10, 9000, 200000, 0.95, 30, 50, 1000, 10000],
        
        # Seed 6: Asymmetric task counts
        # Wide with many tasks vs long_cp with few tasks
        [0.08, 35, 50, 1000, 80000, 0.92, 5, 12, 9000, 180000],
        
        # Seed 7: Asymmetric task lengths
        # Wide with short tasks vs long_cp with very long tasks
        [0.03, 20, 30, 100, 5000, 0.97, 15, 25, 9500, 200000],
        
        # Seed 8: Balanced but opposite
        # Medium edge probs but opposite sides
        [0.15, 18, 28, 2000, 100000, 0.85, 18, 28, 2000, 100000],
        
        # Seed 9: Extreme task count difference
        # Wide with max tasks vs long_cp with min tasks
        [0.06, 40, 50, 5000, 120000, 0.94, 5, 8, 5000, 120000],
        
        # Seed 10: Extreme task length difference
        # Wide with min length vs long_cp with max length
        [0.04, 20, 30, 100, 20000, 0.96, 20, 30, 9800, 200000],
    ]
    
    print(f"  Search space: 10 dimensions")
    print(f"  Initial points: {len(x0)} theory-guided seeds")
    print(f"    - Extreme edge probability contrasts (0.01 vs 0.99)")
    print(f"    - Task count asymmetries (5 vs 50)")
    print(f"    - Task length asymmetries (100 vs 200000)")
    print(f"    - Makespan vs energy trade-off stressors")
    print(f"  Total budget: {n_calls} evaluations")
    print(f"  Expected time: ~{n_calls * 10 / 60:.0f} minutes")
    
    # Run Bayesian Optimization
    print(f"\n[3/3] Running Bayesian Optimization...")
    
    # Use Matern kernel (better for non-smooth functions than RBF)
    from skopt.learning import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),  # nu=2.5 is twice differentiable, good for optimization
        normalize_y=True,
        noise="gaussian",
        n_restarts_optimizer=5
    )
    
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        x0=x0,
        base_estimator=gp,
        random_state=0,
        verbose=True,
        n_jobs=1,
        acq_func='EI',  # Expected Improvement
        n_initial_points=5,  # Reduced random exploration (we have good seeds)
        acq_optimizer='sampling',  # More thorough acquisition optimization
        n_points=10000,  # More candidate points for acquisition
        xi=0.01,  # Exploitation vs exploration (lower = more exploitation)
        kappa=1.96  # For UCB acquisition (not used with EI, but good to set)
    )
    
    # Extract best result
    best_params = result.x
    best_conflict = result.fun
    
    print(f"\n  ✓ Optimization complete!")
    print(f"  Best conflict found: {best_conflict:.4f}")
    print(f"  Total evaluations: {len(result.func_vals)}")
    
    params1 = best_params[:5]
    params2 = best_params[5:]
    
    # Create configs
    _, args1 = create_mdp_from_params(params1, host_specs_file)
    _, args2 = create_mdp_from_params(params2, host_specs_file)
    
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
        "comment": f"Discovered via Bayesian Optimization (conflict={best_conflict:.4f})"
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
        "comment": f"Discovered via Bayesian Optimization (conflict={best_conflict:.4f})"
    }
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config1_file = output_path / "discovered_mdp1.json"
    config2_file = output_path / "discovered_mdp2.json"
    
    with open(config1_file, 'w') as f:
        json.dump(config1, f, indent=2)
    
    with open(config2_file, 'w') as f:
        json.dump(config2, f, indent=2)
    
    print(f"\n  ✓ Saved configs to {output_dir}")
    
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
    
    print(f"\nGradient Conflict: {best_conflict:.4f}")
    
    if best_conflict < -0.8:
        print("  → SEVERE CONFLICT: Separate policies required")
    elif best_conflict < -0.3:
        print("  → MODERATE CONFLICT: Task-conditioned policy recommended")
    else:
        print("  → LOW CONFLICT: Single policy may work")
    
    # Plot convergence
    print("\n" + "="*70)
    print("OPTIMIZATION CONVERGENCE")
    print("="*70)
    print(f"\nBest conflict by iteration:")
    best_so_far = []
    current_best = 1.0
    for i, val in enumerate(result.func_vals):
        if val < current_best:
            current_best = val
        best_so_far.append(current_best)
        if (i + 1) % 10 == 0:
            print(f"  Iter {i+1:3d}: {current_best:.4f}")
    
    return config1, config2, best_conflict


if __name__ == "__main__":
    import sys
    
    host_specs = sys.argv[1] if len(sys.argv) > 1 else "data/host_specs.json"
    
    config1, config2, conflict = discover_adversarial_mdp_pair_bayesian(
        host_specs_file=host_specs,
        output_dir="logs/mdp_discovery",
        n_calls=150  # 150 evaluations for better convergence
    )
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Use discovered configs:")
    print("   python discover_domains_improved.py \\")
    print("     logs/mdp_discovery/discovered_mdp1.json \\")
    print("     logs/mdp_discovery/discovered_mdp2.json \\")
    print(f"     {host_specs}")
    print("\n2. Increase n_calls for better results (e.g., 200-500)")
