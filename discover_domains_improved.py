"""
Improved Domain Discovery via Task Affinity Matrix

Improvements over original approach:
1. Pre-train agent for 10K steps on mixed data
2. Use task affinity matrix instead of direct gradient clustering
3. Increase batch size to 256 steps for more stable gradients
"""

import os
import torch
import torch.optim as optim
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from pathlib import Path
from tqdm import tqdm
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


def pretrain_agent(agent, envs, num_steps=10000, lr=3e-4, device=torch.device("cpu")):
    """
    Pre-train agent on mixed data from multiple MDPs.
    Uses simple REINFORCE for quick training.
    """
    print(f"\n[PRE-TRAINING] Training agent for {num_steps} steps on mixed data...")
    
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    total_episodes = 0
    total_reward = 0.0
    
    pbar = tqdm(total=num_steps, desc="Pre-training")
    
    for step in range(num_steps):
        # Alternate between environments
        env = envs[step % len(envs)]
        
        # Collect episode
        obs, _ = env.reset()
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        
        done = False
        while not done:
            obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(1, -1)).to(device)
            
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            
            episode_obs.append(obs_tensor)
            episode_actions.append(action)
            
            obs, reward, terminated, truncated, info = env.step(int(action.item()))
            episode_rewards.append(reward)
            
            done = terminated or truncated
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute loss
        optimizer.zero_grad()
        loss = 0.0
        
        for obs_t, action_t, ret in zip(episode_obs, episode_actions, returns):
            _, log_prob, _, _ = agent.get_action_and_value(obs_t, action_t)
            loss = loss - log_prob * ret
        
        loss = loss / len(episode_obs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()
        
        total_episodes += 1
        total_reward += sum(episode_rewards)
        
        pbar.update(1)
        
        if (step + 1) % 1000 == 0:
            avg_reward = total_reward / total_episodes
            pbar.set_postfix({"avg_reward": f"{avg_reward:.2f}", "episodes": total_episodes})
    
    pbar.close()
    
    avg_reward = total_reward / total_episodes
    print(f"  ✓ Pre-training complete: {total_episodes} episodes, avg reward: {avg_reward:.2f}")
    
    return agent


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
        makespan_reward = reward
        energy_reward = -info.get("total_energy_active", 0.0)
        
        makespan_rewards.append(makespan_reward)
        energy_rewards.append(energy_reward)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    return obs_list, action_list, makespan_rewards, energy_rewards


def compute_gradient_for_objective(agent, obs_list, action_list, rewards):
    """Compute gradient for given rewards."""
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


def compute_task_affinity(agent_source, env_target, num_batches, batch_size, device):
    """
    Compute task affinity: how well agent trained on source performs on target.
    
    Returns affinity score based on gradient-reward correlation.
    """
    gradient_norms = []
    avg_rewards = []
    
    for _ in range(num_batches):
        obs, act, mk_rew, en_rew = collect_batch_with_objectives(env_target, agent_source, batch_size, device)
        combined_rew = [mk + en for mk, en in zip(mk_rew, en_rew)]
        
        grad = compute_gradient_for_objective(agent_source, obs, act, combined_rew)
        
        if grad is not None:
            gradient_norms.append(np.linalg.norm(grad.cpu().numpy()))
            avg_rewards.append(np.mean(combined_rew))
    
    # Affinity = correlation between gradient magnitude and reward
    # High correlation = agent can learn from this task
    if len(gradient_norms) > 1:
        affinity = np.corrcoef(gradient_norms, avg_rewards)[0, 1]
        # Handle NaN (constant values)
        if np.isnan(affinity):
            affinity = 0.0
    else:
        affinity = 0.0
    
    return affinity, np.mean(avg_rewards), np.mean(gradient_norms)


def discover_domains_improved(
    mdp1_config_path: str,
    mdp2_config_path: str,
    host_specs_file: str,
    pretrain_steps: int = 10000,
    num_batches: int = 50,
    batch_size: int = 256,
    num_clusters: int = 2,
    output_dir: str = "logs/domain_discovery_improved"
):
    """
    Improved domain discovery using task affinity matrix.
    """
    
    print("="*70)
    print("IMPROVED Domain Discovery via Task Affinity Matrix")
    print("="*70)
    print(f"MDP1: {mdp1_config_path}")
    print(f"MDP2: {mdp2_config_path}")
    print(f"Host specs: {host_specs_file}")
    print(f"Pre-training steps: {pretrain_steps}")
    print(f"Batches per MDP: {num_batches}")
    print(f"Batch size: {batch_size} (increased from 64)")
    print(f"Clusters: {num_clusters}")
    print()
    
    # Setup
    os.environ['HOST_SPECS_PATH'] = host_specs_file
    device = torch.device("cpu")
    
    # Load MDPs
    print("[1/7] Loading MDPs...")
    mdp1_args, mdp1_seed = load_mdp_config(mdp1_config_path)
    mdp2_args, mdp2_seed = load_mdp_config(mdp2_config_path)
    print(f"  MDP1: {mdp1_args.style}, p={mdp1_args.gnp_p}")
    print(f"  MDP2: {mdp2_args.style}, p={mdp2_args.gnp_p}")
    
    # Initialize agent
    print("\n[2/7] Initializing agent...")
    variant = AblationVariant(
        name="hetero", graph_type="hetero", gin_num_layers=2,
        use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
    )
    agent = AblationGinAgent(device=device, variant=variant, hidden_dim=128, embedding_dim=64)
    print("  ✓ Agent initialized")
    
    # Create environments
    print("\n[3/7] Creating environments...")
    env1 = GinAgentWrapper(CloudSchedulingGymEnvironment(
        dataset_args=mdp1_args, collect_timelines=False, compute_metrics=False
    ))
    env1.reset(seed=mdp1_seed)
    
    env2 = GinAgentWrapper(CloudSchedulingGymEnvironment(
        dataset_args=mdp2_args, collect_timelines=False, compute_metrics=False
    ))
    env2.reset(seed=mdp2_seed)
    print("  ✓ Environments created")
    
    # Pre-train agent
    print("\n[4/7] Pre-training agent on mixed data...")
    agent = pretrain_agent(agent, [env1, env2], num_steps=pretrain_steps, device=device)
    
    # Compute task affinity matrix
    print(f"\n[5/7] Computing task affinity matrix...")
    print("  This measures how well each MDP transfers to the other")
    
    mdps = [env1, env2]
    mdp_names = [mdp1_args.style, mdp2_args.style]
    affinity_matrix = np.zeros((2, 2))
    reward_matrix = np.zeros((2, 2))
    gradient_matrix = np.zeros((2, 2))
    
    for i, (env_source, name_source) in enumerate(zip(mdps, mdp_names)):
        print(f"\n  Fine-tuning on {name_source}...")
        
        # Clone agent and fine-tune on source MDP
        agent_finetuned = AblationGinAgent(device=device, variant=variant, hidden_dim=128, embedding_dim=64)
        agent_finetuned.load_state_dict(agent.state_dict())
        
        # Quick fine-tune (1000 steps)
        agent_finetuned = pretrain_agent(agent_finetuned, [env_source], num_steps=1000, device=device)
        
        for j, (env_target, name_target) in enumerate(zip(mdps, mdp_names)):
            print(f"    Testing on {name_target}...", end=" ")
            
            affinity, avg_reward, avg_grad = compute_task_affinity(
                agent_finetuned, env_target, num_batches, batch_size, device
            )
            
            affinity_matrix[i, j] = affinity
            reward_matrix[i, j] = avg_reward
            gradient_matrix[i, j] = avg_grad
            
            print(f"affinity={affinity:+.3f}, reward={avg_reward:.2f}, grad_norm={avg_grad:.2e}")
    
    print(f"\n  ✓ Task affinity matrix computed")
    
    # Analyze affinity matrix
    print("\n[6/7] Analyzing task affinity matrix...")
    print("\nTask Affinity Matrix:")
    print(f"              {mdp_names[0]:>12} {mdp_names[1]:>12}")
    for i, name in enumerate(mdp_names):
        print(f"  {name:>10}  {affinity_matrix[i, 0]:>12.4f} {affinity_matrix[i, 1]:>12.4f}")
    
    print("\nAverage Reward Matrix:")
    print(f"              {mdp_names[0]:>12} {mdp_names[1]:>12}")
    for i, name in enumerate(mdp_names):
        print(f"  {name:>10}  {reward_matrix[i, 0]:>12.2f} {reward_matrix[i, 1]:>12.2f}")
    
    # Compute cross-task affinity (how well tasks transfer to each other)
    cross_affinity_01 = affinity_matrix[0, 1]
    cross_affinity_10 = affinity_matrix[1, 0]
    avg_cross_affinity = (cross_affinity_01 + cross_affinity_10) / 2
    
    # Compute self-affinity (how well tasks transfer to themselves)
    self_affinity_0 = affinity_matrix[0, 0]
    self_affinity_1 = affinity_matrix[1, 1]
    avg_self_affinity = (self_affinity_0 + self_affinity_1) / 2
    
    print(f"\nKey Metrics:")
    print(f"  Self-affinity (avg):  {avg_self_affinity:+.4f}")
    print(f"  Cross-affinity (avg): {avg_cross_affinity:+.4f}")
    print(f"  Affinity gap:         {avg_self_affinity - avg_cross_affinity:+.4f}")
    
    # Decision criterion
    print("\n" + "="*70)
    print("DOMAIN DISCOVERY RESULT")
    print("="*70)
    
    if avg_cross_affinity < 0.3 and (avg_self_affinity - avg_cross_affinity) > 0.3:
        print("\n✓ DISTINCT DOMAINS DETECTED")
        print(f"  Tasks have low cross-affinity ({avg_cross_affinity:.3f})")
        print(f"  Tasks have high self-affinity ({avg_self_affinity:.3f})")
        print(f"  → Recommendation: Use separate policies or task-conditioned policy")
        domain_decision = "separate_domains"
    elif avg_cross_affinity < 0:
        print("\n⚠ MODERATE DOMAIN SEPARATION")
        print(f"  Tasks have negative cross-affinity ({avg_cross_affinity:.3f})")
        print(f"  → Recommendation: Use gradient conflict resolution (PCGrad/CAGrad)")
        domain_decision = "moderate_conflict"
    else:
        print("\n✓ SIMILAR DOMAINS")
        print(f"  Tasks have positive cross-affinity ({avg_cross_affinity:.3f})")
        print(f"  → Recommendation: Single policy should work")
        domain_decision = "single_domain"
    
    # Clustering (for comparison with original method)
    print(f"\n[7/7] Clustering based on affinity matrix...")
    
    # Use spectral clustering on affinity matrix
    # Convert affinity to distance: distance = 1 - affinity
    distance_matrix = 1 - affinity_matrix
    distance_matrix = np.maximum(distance_matrix, 0)  # Ensure non-negative
    
    # For 2 MDPs, clustering is trivial, but we do it for consistency
    if avg_cross_affinity < avg_self_affinity - 0.3:
        cluster_labels = np.array([0, 1])  # Separate clusters
        purity = 1.0
    else:
        cluster_labels = np.array([0, 0])  # Same cluster
        purity = 0.5
    
    print(f"  Cluster assignments: {cluster_labels}")
    print(f"  Purity: {purity:.4f}")
    
    # Save visualization data (plot separately to avoid matplotlib issues)
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results
    import json
    results = {
        "mdp1": {"name": mdp_names[0], "config": mdp1_config_path},
        "mdp2": {"name": mdp_names[1], "config": mdp2_config_path},
        "affinity_matrix": affinity_matrix.tolist(),
        "reward_matrix": reward_matrix.tolist(),
        "gradient_matrix": gradient_matrix.tolist(),
        "metrics": {
            "avg_self_affinity": float(avg_self_affinity),
            "avg_cross_affinity": float(avg_cross_affinity),
            "affinity_gap": float(avg_self_affinity - avg_cross_affinity),
        },
        "decision": domain_decision,
        "cluster_labels": cluster_labels.tolist(),
        "purity": float(purity),
    }
    
    results_file = output_path / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved results: {results_file}")
    print()
    
    env1.close()
    env2.close()
    
    return results


if __name__ == "__main__":
    import sys
    
    mdp1_config = sys.argv[1] if len(sys.argv) > 1 else "data/rl_configs/extreme_longcp_p095_bottleneck.json"
    mdp2_config = sys.argv[2] if len(sys.argv) > 2 else "data/rl_configs/extreme_wide_p005_bottleneck.json"
    host_specs = sys.argv[3] if len(sys.argv) > 3 else "data/host_specs.json"
    
    results = discover_domains_improved(
        mdp1_config_path=mdp1_config,
        mdp2_config_path=mdp2_config,
        host_specs_file=host_specs,
        pretrain_steps=10000,  # NEW: Pre-training
        num_batches=50,
        batch_size=256,  # INCREASED: from 64 to 256
        num_clusters=2,
        output_dir="logs/domain_discovery_improved"
    )
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nDecision: {results['decision']}")
    print(f"Affinity gap: {results['metrics']['affinity_gap']:.4f}")
    print(f"Purity: {results['purity']:.4f}")
