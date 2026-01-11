"""
Train a single agent on two MDPs simultaneously and track gradient conflict.

Goal: Prove that gradient conflict prevents learning.

Experiment:
1. Train single agent with mixed batches from MDP1 and MDP2
2. At each iteration, measure gradient conflict between the two batches
3. Track learning progress (returns) for both MDPs
4. Show: High conflict → No learning improvement

Expected result: Agent cannot learn because gradients from MDP1 and MDP2 contradict each other.
"""

import os
import torch
import numpy as np
import csv
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


def collect_batch(env, agent, num_steps, device):
    """Collect a batch of transitions."""
    obs_list = []
    action_list = []
    reward_list = []
    
    obs, _ = env.reset()
    episode_rewards = []
    episode_reward = 0
    
    for _ in range(num_steps):
        obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(1, -1)).to(device)
        
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        
        obs_list.append(obs_tensor)
        action_list.append(action)
        
        obs, reward, terminated, truncated, info = env.step(int(action.item()))
        reward_list.append(reward)
        episode_reward += reward
        
        if terminated or truncated:
            episode_rewards.append(episode_reward)
            episode_reward = 0
            obs, _ = env.reset()
    
    return obs_list, action_list, reward_list, episode_rewards


def compute_gradient(agent, obs_list, action_list, reward_list):
    """Compute policy gradient from batch."""
    agent.zero_grad()
    
    # REINFORCE
    returns = []
    G = 0
    for r in reversed(reward_list):
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
    
    if not grad_parts:
        return None
    
    return torch.cat(grad_parts)


def update_agent(agent, optimizer, obs_list, action_list, reward_list):
    """Update agent with batch."""
    optimizer.zero_grad()
    
    # REINFORCE
    returns = []
    G = 0
    for r in reversed(reward_list):
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
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
    
    optimizer.step()
    
    return loss.item()


def train_with_conflict_tracking(
    host_specs_file: str,
    num_iterations: int = 100,
    batch_size: int = 128,
    output_dir: str = "logs/conflict_tracking"
):
    """Train single agent on two MDPs and track gradient conflict."""
    
    print("="*70)
    print("Training with Gradient Conflict Tracking")
    print("="*70)
    print(f"Host specs: {host_specs_file}")
    print(f"Iterations: {num_iterations}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Set up
    os.environ['HOST_SPECS_PATH'] = host_specs_file
    device = torch.device("cpu")
    
    # Load MDPs
    print("[1/5] Loading MDPs...")
    mdp1_args, mdp1_seed = load_mdp_config("data/rl_configs/extreme_longcp_p095_bottleneck.json")
    mdp2_args, mdp2_seed = load_mdp_config("data/rl_configs/extreme_wide_p005_bottleneck.json")
    print(f"  MDP1: {mdp1_args.style}, p={mdp1_args.gnp_p}")
    print(f"  MDP2: {mdp2_args.style}, p={mdp2_args.gnp_p}")
    
    # Initialize agent
    print("\n[2/5] Initializing agent...")
    variant = AblationVariant(
        name="hetero",
        graph_type="hetero",
        gin_num_layers=2,
        use_batchnorm=True,
        use_task_dependencies=True,
        use_actor_global_embedding=True,
        mlp_only=False,
        gat_heads=4,
    )
    
    agent = AblationGinAgent(device=device, variant=variant, hidden_dim=128, embedding_dim=64)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.0003)
    print("  ✓ Agent initialized")
    
    # Create environments
    print("\n[3/5] Creating environments...")
    env1 = GinAgentWrapper(CloudSchedulingGymEnvironment(
        dataset_args=mdp1_args, collect_timelines=False, compute_metrics=False
    ))
    env1.reset(seed=mdp1_seed)
    
    env2 = GinAgentWrapper(CloudSchedulingGymEnvironment(
        dataset_args=mdp2_args, collect_timelines=False, compute_metrics=False
    ))
    env2.reset(seed=mdp2_seed)
    print("  ✓ Environments created")
    
    # Prepare logging
    print("\n[4/5] Setting up logging...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_file = output_path / "training_log.csv"
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'iteration', 
            'mdp1_mean_return', 'mdp2_mean_return',
            'mdp1_loss', 'mdp2_loss',
            'cosine_similarity', 
            'grad1_norm', 'grad2_norm'
        ])
    
    print(f"  ✓ Logging to {log_file}")
    
    # Training loop
    print("\n[5/5] Starting training...")
    print()
    print(f"{'Iter':>6} | {'MDP1 Ret':>10} | {'MDP2 Ret':>10} | {'Conflict':>10} | {'Status':>20}")
    print("-" * 70)
    
    for iteration in range(num_iterations):
        # Collect batches from both MDPs
        obs1, act1, rew1, ep_rew1 = collect_batch(env1, agent, batch_size, device)
        obs2, act2, rew2, ep_rew2 = collect_batch(env2, agent, batch_size, device)
        
        # Compute gradients BEFORE update (to measure conflict)
        grad1 = compute_gradient(agent, obs1, act1, rew1)
        grad2 = compute_gradient(agent, obs2, act2, rew2)
        
        # Measure conflict
        if grad1 is not None and grad2 is not None:
            norm1 = grad1.norm().item()
            norm2 = grad2.norm().item()
            cosine_sim = torch.dot(grad1, grad2).item() / (norm1 * norm2 + 1e-9)
        else:
            norm1 = norm2 = cosine_sim = 0.0
        
        # Update agent with BOTH batches (mixed training)
        loss1 = update_agent(agent, optimizer, obs1, act1, rew1)
        loss2 = update_agent(agent, optimizer, obs2, act2, rew2)
        
        # Compute mean returns
        mdp1_return = np.mean(ep_rew1) if ep_rew1 else np.mean(rew1)
        mdp2_return = np.mean(ep_rew2) if ep_rew2 else np.mean(rew2)
        
        # Log
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                mdp1_return, mdp2_return,
                loss1, loss2,
                cosine_sim,
                norm1, norm2
            ])
        
        # Print progress
        if iteration % 10 == 0:
            if cosine_sim < -0.5:
                status = "HIGH CONFLICT"
            elif cosine_sim < 0:
                status = "MODERATE CONFLICT"
            else:
                status = "LOW CONFLICT"
            
            print(f"{iteration:6d} | {mdp1_return:10.4f} | {mdp2_return:10.4f} | {cosine_sim:10.4f} | {status:>20}")
    
    env1.close()
    env2.close()
    
    print()
    print("="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Results saved to: {log_file}")
    print()
    print("Analysis:")
    print("  1. Check if returns improve over time")
    print("  2. Check if conflict remains high")
    print("  3. If conflict is high AND returns don't improve → conflict prevents learning")
    print()


if __name__ == "__main__":
    import sys
    
    host_specs = sys.argv[1] if len(sys.argv) > 1 else "data/host_specs.json"
    
    print("Running experiment to prove gradient conflict prevents learning...")
    print()
    
    train_with_conflict_tracking(
        host_specs_file=host_specs,
        num_iterations=100,
        batch_size=128,
        output_dir=f"logs/conflict_proof_{Path(host_specs).stem}"
    )
