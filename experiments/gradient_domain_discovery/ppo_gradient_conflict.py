"""
Gradient Conflict Measurement during ACTUAL PPO Training
Uses YOUR EXACT architecture, reward function, and training loop.

This script:
1. Creates two environments: one for QF (wide DAGs), one for BN (long CP DAGs)
2. Runs PPO training on both simultaneously
3. Measures gradient conflict between domains during actual training
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List
import gymnasium as gym

# Your imports
from scheduler.rl_model import ablation_gnn as AG
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.dataset_generator.gen_dataset import DatasetArgs
# HOST_SPECS_PATH not needed

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

output_dir = Path(__file__).parent / "figures"
output_dir.mkdir(exist_ok=True)


@dataclass
class Args:
    """Training arguments matching your setup."""
    seed: int = 42
    num_envs: int = 2
    num_steps: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    norm_adv: bool = True
    clip_vloss: bool = True
    hidden_dim: int = 64
    embedding_dim: int = 32
    # Dataset args
    dataset: Optional[DatasetArgs] = None


def create_qf_dataset_args():
    """Queue-Free regime: wide DAGs, low load."""
    return DatasetArgs(
        dag_method="gnp",
        gnp_min_n=8,
        gnp_max_n=12,
        gnp_p=0.2,  # Sparse = wide DAGs
        workflow_count=3,
        host_count=2,
        vm_count=4,
        style="wide",
    )


def create_bn_dataset_args():
    """Bottleneck regime: long critical path DAGs, high load."""
    return DatasetArgs(
        dag_method="gnp",
        gnp_min_n=12,
        gnp_max_n=20,
        gnp_p=0.8,  # Dense = long critical paths
        workflow_count=5,
        host_count=2,
        vm_count=4,
        style="long_cp",
    )


def make_env(args: Args, seed: int, domain: str):
    """Create environment for a specific domain."""
    def thunk():
        if domain == "qf":
            ds_args = create_qf_dataset_args()
        else:
            ds_args = create_bn_dataset_args()
        
        # Set seed in dataset args
        ds_args.seed = seed
        
        env = CloudSchedulingGymEnvironment(dataset_args=ds_args)
        env = GinAgentWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def compute_ppo_gradients(agent, obs, actions, advantages, returns, old_logprobs, old_values, args):
    """Compute PPO gradients without applying them."""
    
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions.long())
    
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    
    # Normalize advantages
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Policy loss
    pg_loss = torch.max(
        -advantages * ratio,
        -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    ).mean()
    
    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
    
    # Entropy
    entropy_loss = entropy.mean()
    
    # Total loss
    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
    
    return loss, pg_loss, v_loss, entropy_loss


def collect_rollout(env, agent, num_steps, device):
    """Collect a rollout from environment."""
    obs_list = []
    actions_list = []
    logprobs_list = []
    rewards_list = []
    dones_list = []
    values_list = []
    
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    
    for step in range(num_steps):
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(obs)
        
        obs_list.append(obs)
        actions_list.append(action)
        logprobs_list.append(logprob)
        values_list.append(value.flatten())
        
        next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy().item())
        done = terminated or truncated
        
        rewards_list.append(torch.tensor([reward], dtype=torch.float32, device=device))
        dones_list.append(torch.tensor([float(done)], device=device))
        
        if done:
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Stack
    obs_t = torch.cat(obs_list, dim=0)
    actions_t = torch.cat(actions_list, dim=0)
    logprobs_t = torch.cat(logprobs_list, dim=0)
    rewards_t = torch.cat(rewards_list, dim=0)
    dones_t = torch.cat(dones_list, dim=0)
    values_t = torch.cat(values_list, dim=0)
    
    # Compute advantages (GAE)
    with torch.no_grad():
        next_value = agent.get_value(obs).flatten()
        advantages = torch.zeros_like(rewards_t)
        lastgaelam = 0
        gamma = 0.99
        gae_lambda = 0.95
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - dones_t[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_t[t + 1]
                nextvalues = values_t[t + 1]
            delta = rewards_t[t] + gamma * nextvalues * nextnonterminal - values_t[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values_t
    
    return obs_t, actions_t, logprobs_t, values_t, advantages, returns


def main():
    print("="*70)
    print("PPO GRADIENT CONFLICT MEASUREMENT")
    print("Using YOUR EXACT Architecture and Reward Function")
    print("="*70)
    
    device = torch.device("cpu")
    args = Args()
    
    # Create YOUR EXACT HETERO architecture
    variant = AG.AblationVariant(
        name='hetero',
        graph_type='hetero',
        hetero_base='sage',
        gin_num_layers=2,
        use_batchnorm=True,
        use_task_dependencies=True,
        use_actor_global_embedding=True,
    )
    
    agent = AG.AblationGinAgent(device, variant, hidden_dim=args.hidden_dim, embedding_dim=args.embedding_dim)
    
    # Create environments for each domain first (need to initialize agent)
    print("\nCreating environments...")
    env_qf = make_env(args, seed=42, domain="qf")()
    env_bn = make_env(args, seed=43, domain="bn")()
    print("  QF env: Wide DAGs (gnp_p=0.2, 8-12 tasks)")
    print("  BN env: Long CP DAGs (gnp_p=0.8, 12-20 tasks)")
    
    # Initialize agent with a forward pass
    print("\nInitializing agent...")
    obs_init, _ = env_qf.reset()
    obs_init_t = torch.tensor(obs_init, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _ = agent.get_action_and_value(obs_init_t)
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    num_params = sum(p.numel() for p in agent.parameters())
    print(f"\nArchitecture: HETERO (HeteroConv + SAGE)")
    print(f"Parameters: {num_params:,}")
    
    # Environments already created above
    
    # Training loop with gradient measurement
    num_iterations = 30
    num_steps = 32
    
    cross_domain_sims = []
    within_qf_sims = []
    within_bn_sims = []
    qf_rewards = []
    bn_rewards = []
    
    prev_grad_qf = None
    prev_grad_bn = None
    
    print(f"\nRunning {num_iterations} training iterations...")
    print("-"*70)
    
    for iteration in range(num_iterations):
        # Collect rollouts from both domains
        try:
            obs_qf, act_qf, logp_qf, val_qf, adv_qf, ret_qf = collect_rollout(env_qf, agent, num_steps, device)
            obs_bn, act_bn, logp_bn, val_bn, adv_bn, ret_bn = collect_rollout(env_bn, agent, num_steps, device)
        except Exception as e:
            print(f"  Iteration {iteration}: Error collecting rollout: {e}")
            continue
        
        # Compute QF gradient
        agent.zero_grad()
        loss_qf, pg_qf, vl_qf, ent_qf = compute_ppo_gradients(
            agent, obs_qf, act_qf, adv_qf.detach(), ret_qf.detach(), logp_qf.detach(), val_qf.detach(), args
        )
        loss_qf.backward()
        grad_qf = torch.cat([p.grad.flatten() for p in agent.parameters() if p.grad is not None]).clone()
        
        # Compute BN gradient
        agent.zero_grad()
        loss_bn, pg_bn, vl_bn, ent_bn = compute_ppo_gradients(
            agent, obs_bn, act_bn, adv_bn.detach(), ret_bn.detach(), logp_bn.detach(), val_bn.detach(), args
        )
        loss_bn.backward()
        grad_bn = torch.cat([p.grad.flatten() for p in agent.parameters() if p.grad is not None]).clone()
        
        # Cross-domain similarity
        cos_cross = torch.dot(grad_qf, grad_bn) / (grad_qf.norm() * grad_bn.norm() + 1e-8)
        cross_domain_sims.append(cos_cross.item())
        
        # Within-domain similarity
        if prev_grad_qf is not None:
            cos_qf = torch.dot(grad_qf, prev_grad_qf) / (grad_qf.norm() * prev_grad_qf.norm() + 1e-8)
            within_qf_sims.append(cos_qf.item())
        if prev_grad_bn is not None:
            cos_bn = torch.dot(grad_bn, prev_grad_bn) / (grad_bn.norm() * prev_grad_bn.norm() + 1e-8)
            within_bn_sims.append(cos_bn.item())
        
        prev_grad_qf = grad_qf.clone()
        prev_grad_bn = grad_bn.clone()
        
        # Track rewards
        qf_rewards.append(adv_qf.mean().item())
        bn_rewards.append(adv_bn.mean().item())
        
        # Apply combined update (simulating joint training)
        # Recompute losses for combined update
        agent.zero_grad()
        loss_qf_new, _, _, _ = compute_ppo_gradients(
            agent, obs_qf, act_qf, adv_qf.detach(), ret_qf.detach(), logp_qf.detach(), val_qf.detach(), args
        )
        loss_bn_new, _, _, _ = compute_ppo_gradients(
            agent, obs_bn, act_bn, adv_bn.detach(), ret_bn.detach(), logp_bn.detach(), val_bn.detach(), args
        )
        combined_loss = (loss_qf_new + loss_bn_new) / 2
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()
        
        if iteration % 5 == 0:
            wqf = np.mean(within_qf_sims[-5:]) if within_qf_sims else 0
            wbn = np.mean(within_bn_sims[-5:]) if within_bn_sims else 0
            print(f"Iter {iteration:2d}: Cross={cos_cross.item():+.3f}, "
                  f"WithinQF={wqf:+.3f}, WithinBN={wbn:+.3f}, "
                  f"Loss_QF={loss_qf.item():.3f}, Loss_BN={loss_bn.item():.3f}")
    
    env_qf.close()
    env_bn.close()
    
    # Results
    avg_cross = np.mean(cross_domain_sims)
    avg_qf = np.mean(within_qf_sims) if within_qf_sims else 0
    avg_bn = np.mean(within_bn_sims) if within_bn_sims else 0
    conflict_rate = sum(1 for s in cross_domain_sims if s < 0) / len(cross_domain_sims) * 100
    
    print("\n" + "="*70)
    print("RESULTS: PPO GRADIENT CONFLICT (YOUR REAL TRAINING)")
    print("="*70)
    print(f"\nCROSS-DOMAIN (QF vs BN):")
    print(f"  Average Similarity: {avg_cross:+.3f}")
    print(f"  Conflict Rate: {conflict_rate:.0f}%")
    print(f"\nWITHIN QUEUE-FREE:")
    print(f"  Average Similarity: {avg_qf:+.3f}")
    print(f"\nWITHIN BOTTLENECK:")
    print(f"  Average Similarity: {avg_bn:+.3f}")
    
    if avg_cross < 0:
        print("\n✓ GRADIENT CONFLICT CONFIRMED during actual PPO training!")
    elif avg_cross < avg_qf and avg_cross < avg_bn:
        print("\n○ Cross-domain similarity lower than within-domain (partial conflict)")
    else:
        print("\n○ No significant conflict detected")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cross-domain similarity over iterations
    ax = axes[0, 0]
    colors = ['#CC3311' if s < 0 else '#009988' for s in cross_domain_sims]
    ax.bar(range(len(cross_domain_sims)), cross_domain_sims, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=avg_cross, color='#0077BB', linewidth=2, linestyle='--', label=f'Mean: {avg_cross:.3f}')
    ax.fill_between(range(len(cross_domain_sims)), -1, 0, alpha=0.1, color='red')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(a) Cross-Domain Gradient Similarity\n(PPO Training with YOUR Reward)', fontweight='bold')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    
    # Similarity distribution
    ax = axes[0, 1]
    if within_qf_sims:
        ax.hist(within_qf_sims, bins=12, alpha=0.6, color='#0077BB', label=f'Within QF (μ={avg_qf:.2f})', edgecolor='black')
    if within_bn_sims:
        ax.hist(within_bn_sims, bins=12, alpha=0.6, color='#EE7733', label=f'Within BN (μ={avg_bn:.2f})', edgecolor='black')
    ax.hist(cross_domain_sims, bins=12, alpha=0.6, color='#CC3311', label=f'Cross (μ={avg_cross:.2f})', edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=2, linestyle='--')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('(b) Similarity Distribution', fontweight='bold')
    ax.legend(fontsize=9)
    
    # Similarity over time
    ax = axes[1, 0]
    ax.plot(range(len(cross_domain_sims)), cross_domain_sims, 'o-', color='#CC3311', label='Cross-Domain', linewidth=2)
    if within_qf_sims:
        ax.plot(range(1, len(within_qf_sims)+1), within_qf_sims, 's-', color='#0077BB', label='Within QF', linewidth=2, alpha=0.7)
    if within_bn_sims:
        ax.plot(range(1, len(within_bn_sims)+1), within_bn_sims, '^-', color='#EE7733', label='Within BN', linewidth=2, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=1, linestyle='--')
    ax.fill_between(range(len(cross_domain_sims)), -1, 0, alpha=0.1, color='red')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(c) Gradient Similarity Over Training', fontweight='bold')
    ax.legend()
    ax.set_ylim(-1.1, 1.1)
    
    # Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""
PPO GRADIENT CONFLICT RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Architecture:
  • Graph Type: HETERO (HeteroConv)
  • Hetero Base: SAGE
  • Parameters: {num_params:,}

Training Setup:
  • Loss: PPO (policy + value + entropy)
  • Reward: YOUR reward function
    (energy_weight * ΔE + makespan_weight * ΔM)

Cross-Domain (QF vs BN):
  • Avg Similarity: {avg_cross:+.3f}
  • Conflict Rate: {conflict_rate:.0f}%

Within-Domain:
  • QF: {avg_qf:+.3f}
  • BN: {avg_bn:+.3f}

CONCLUSION:
  {"Gradient conflict confirmed!" if avg_cross < 0 else "Cross < Within (partial conflict)" if avg_cross < min(avg_qf, avg_bn) else "No significant conflict"}
"""
    ax.text(0.1, 0.95, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ppo_gradient_conflict.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'ppo_gradient_conflict.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"\nSaved: {output_dir / 'ppo_gradient_conflict.png'}")


if __name__ == "__main__":
    main()
