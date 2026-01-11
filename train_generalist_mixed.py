"""
Train a Generalist Agent on Mixed Data

Trains a single agent on a balanced mixture of:
- Wide MDPs (low edge_prob, many tasks, short)
- Long CP MDPs (high edge_prob, few tasks, long)

This agent will have balanced experience and can be used for
gradient subspace discovery.
"""

import os
import sys
import torch
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.dataset_generator.gen_dataset import DatasetArgs


def create_mixed_environment(host_specs_file: str, seed: int = None):
    """
    Create environment that samples from mixed distribution.
    50% wide, 50% long_cp.
    """
    # Randomly choose MDP type
    if random.random() < 0.5:
        # Wide MDP
        style = "wide"
        edge_prob = random.uniform(0.05, 0.40)
        min_tasks = random.randint(20, 40)
        max_tasks = random.randint(min_tasks, 50)
        min_length = random.randint(100, 2000)
        max_length = random.randint(max(min_length, 5000), 50000)
    else:
        # Long CP MDP
        style = "long_cp"
        edge_prob = random.uniform(0.60, 0.95)
        min_tasks = random.randint(5, 15)
        max_tasks = random.randint(min_tasks, 20)
        min_length = random.randint(50000, 100000)
        max_length = random.randint(max(min_length, 100000), 200000)
    
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
        dataset_args=args, collect_timelines=False, compute_metrics=True
    ))
    
    return env


def train_generalist_agent(
    host_specs_file: str,
    output_dir: str = "logs/generalist_mixed",
    total_timesteps: int = 500000,
    num_envs: int = 4,
    batch_size: int = 256,
    learning_rate: float = 2.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    update_epochs: int = 4,
    save_interval: int = 50000
):
    """
    Train generalist agent using PPO on mixed data.
    """
    print("="*80)
    print("TRAINING GENERALIST AGENT ON MIXED DATA")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Environments: {num_envs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Data distribution: 50% wide, 50% long_cp")
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize agent
    print("\n[1/3] Initializing agent...")
    variant = AblationVariant(
        name="hetero", graph_type="hetero", gin_num_layers=2,
        use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
    )
    agent = AblationGinAgent(device=device, variant=variant, hidden_dim=128, embedding_dim=64)
    
    # Initialize with dummy forward pass
    dummy_env = create_mixed_environment(host_specs_file, seed=42)
    dummy_obs, _ = dummy_env.reset(seed=42)
    dummy_obs_tensor = torch.from_numpy(np.asarray(dummy_obs, dtype=np.float32).reshape(1, -1)).to(device)
    with torch.no_grad():
        _ = agent.get_action_and_value(dummy_obs_tensor)
    dummy_env.close()
    
    num_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"  ✓ Agent initialized ({num_params:,} parameters)")
    
    # Optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    # Create environments
    print(f"\n[2/3] Creating {num_envs} mixed environments...")
    envs = [create_mixed_environment(host_specs_file) for _ in range(num_envs)]
    print(f"  ✓ Environments created")
    
    # Training loop
    print(f"\n[3/3] Training for {total_timesteps:,} timesteps...")
    
    global_step = 0
    num_updates = total_timesteps // batch_size
    
    # Storage for rollouts
    obs_buffer = []
    actions_buffer = []
    logprobs_buffer = []
    rewards_buffer = []
    dones_buffer = []
    values_buffer = []
    
    # Reset environments
    obs_list = [env.reset(seed=42 + i)[0] for i, env in enumerate(envs)]
    
    # Training metrics
    episode_returns = []
    episode_lengths = []
    
    pbar = tqdm(range(num_updates), desc="Training", ncols=100)
    
    for update in pbar:
        # Collect rollout
        for step in range(batch_size // num_envs):
            global_step += num_envs
            
            # Get actions for all environments
            obs_tensors = [
                torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(1, -1)).to(device)
                for obs in obs_list
            ]
            
            with torch.no_grad():
                actions = []
                logprobs = []
                values = []
                
                for obs_tensor in obs_tensors:
                    action, logprob, _, value = agent.get_action_and_value(obs_tensor)
                    actions.append(action)
                    logprobs.append(logprob)
                    values.append(value)
            
            # Step environments
            next_obs_list = []
            for i, (env, obs, action) in enumerate(zip(envs, obs_list, actions)):
                obs_buffer.append(obs)
                actions_buffer.append(action.cpu())
                logprobs_buffer.append(logprobs[i].cpu())
                values_buffer.append(values[i].cpu())
                
                next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
                done = terminated or truncated
                
                rewards_buffer.append(reward)
                dones_buffer.append(done)
                
                if done:
                    # Track episode metrics
                    if 'makespan' in info:
                        episode_returns.append(-info['makespan'])
                        episode_lengths.append(step)
                    
                    next_obs, _ = env.reset()
                
                next_obs_list.append(next_obs)
            
            obs_list = next_obs_list
        
        # Compute advantages using GAE
        advantages = []
        returns = []
        
        # Bootstrap value for last observation
        with torch.no_grad():
            next_values = [
                agent.get_action_and_value(
                    torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(1, -1)).to(device)
                )[3]
                for obs in obs_list
            ]
        
        # Compute GAE
        gae = 0
        for t in reversed(range(len(rewards_buffer))):
            if t == len(rewards_buffer) - 1:
                next_value = next_values[t % num_envs]
                next_done = 0
            else:
                next_value = values_buffer[t + 1]
                next_done = dones_buffer[t]
            
            delta = rewards_buffer[t] + gamma * next_value * (1 - next_done) - values_buffer[t]
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values_buffer[t])
        
        # Convert to tensors
        obs_batch = torch.from_numpy(
            np.array([np.asarray(obs, dtype=np.float32) for obs in obs_buffer])
        ).to(device)
        actions_batch = torch.stack(actions_buffer).to(device)
        logprobs_batch = torch.stack(logprobs_buffer).to(device)
        advantages_batch = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns_batch = torch.tensor(returns, dtype=torch.float32).to(device)
        values_batch = torch.stack(values_buffer).to(device)
        
        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
        
        # PPO update
        for epoch in range(update_epochs):
            # Forward pass
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                obs_batch, actions_batch.squeeze()
            )
            
            # Policy loss
            logratio = newlogprob - logprobs_batch
            ratio = logratio.exp()
            
            pg_loss1 = -advantages_batch * ratio
            pg_loss2 = -advantages_batch * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            # Value loss
            v_loss = 0.5 * ((newvalue - returns_batch) ** 2).mean()
            
            # Entropy loss
            entropy_loss = entropy.mean()
            
            # Total loss
            loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()
        
        # Clear buffers
        obs_buffer.clear()
        actions_buffer.clear()
        logprobs_buffer.clear()
        rewards_buffer.clear()
        dones_buffer.clear()
        values_buffer.clear()
        
        # Update progress bar
        if episode_returns:
            avg_return = np.mean(episode_returns[-100:])
            pbar.set_postfix({
                'step': f'{global_step:,}',
                'return': f'{avg_return:.2f}',
                'episodes': len(episode_returns)
            })
        
        # Save checkpoint
        if (update + 1) % (save_interval // batch_size) == 0:
            checkpoint_path = output_path / f"checkpoint_{global_step}.pt"
            torch.save(agent.state_dict(), checkpoint_path)
            print(f"\n  ✓ Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = output_path / "generalist_final.pt"
    torch.save(agent.state_dict(), final_path)
    print(f"\n  ✓ Saved final model: {final_path}")
    
    # Close environments
    for env in envs:
        env.close()
    
    # Save training stats
    stats = {
        'total_timesteps': global_step,
        'num_episodes': len(episode_returns),
        'mean_return': float(np.mean(episode_returns)) if episode_returns else 0,
        'std_return': float(np.std(episode_returns)) if episode_returns else 0,
        'mean_length': float(np.mean(episode_lengths)) if episode_lengths else 0,
    }
    
    import json
    with open(output_path / "training_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nStatistics:")
    print(f"  Total timesteps: {global_step:,}")
    print(f"  Episodes: {len(episode_returns)}")
    if episode_returns:
        print(f"  Mean return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
        print(f"  Mean episode length: {np.mean(episode_lengths):.1f}")
    
    print(f"\nModel saved to: {final_path}")
    
    return agent, stats


if __name__ == "__main__":
    import sys
    
    host_specs = sys.argv[1] if len(sys.argv) > 1 else "data/host_specs.json"
    total_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500000
    
    agent, stats = train_generalist_agent(
        host_specs,
        total_timesteps=total_steps
    )
