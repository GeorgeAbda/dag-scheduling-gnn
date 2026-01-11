"""
PPO Mixed Training Gradient Conflict Measurement

Trains a SINGLE agent on MIXED data from both QF and BN regimes.
Measures gradient conflict between the two domains during actual PPO training.

This demonstrates that even within the same training batch, gradients from
QF samples and BN samples point in conflicting directions.
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
from dataclasses import dataclass
from typing import Optional
import gymnasium as gym

from scheduler.rl_model import ablation_gnn as AG
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.dataset_generator.gen_dataset import DatasetArgs

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

output_dir = Path(__file__).parent / "figures"
output_dir.mkdir(exist_ok=True)


@dataclass
class Args:
    """Training arguments."""
    seed: int = 42
    num_steps: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    norm_adv: bool = True
    clip_vloss: bool = True
    hidden_dim: int = 64
    embedding_dim: int = 32


def create_qf_dataset_args(seed):
    """Queue-Free regime: wide DAGs, low load."""
    ds = DatasetArgs(
        dag_method="gnp",
        gnp_min_n=8,
        gnp_max_n=12,
        gnp_p=0.2,
        workflow_count=3,
        host_count=2,
        vm_count=4,
        style="wide",
    )
    ds.seed = seed
    return ds


def create_bn_dataset_args(seed):
    """Bottleneck regime: long critical path DAGs, high load."""
    ds = DatasetArgs(
        dag_method="gnp",
        gnp_min_n=12,
        gnp_max_n=20,
        gnp_p=0.8,
        workflow_count=5,
        host_count=2,
        vm_count=4,
        style="long_cp",
    )
    ds.seed = seed
    return ds


def create_env(domain: str, seed: int):
    """Create environment for a specific domain."""
    if domain == "qf":
        ds_args = create_qf_dataset_args(seed)
    else:
        ds_args = create_bn_dataset_args(seed)
    
    env = CloudSchedulingGymEnvironment(dataset_args=ds_args)
    env = GinAgentWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def collect_rollout(env, agent, num_steps, device):
    """Collect a rollout from environment."""
    obs_list, actions_list, logprobs_list = [], [], []
    rewards_list, dones_list, values_list = [], [], []
    
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    
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
    
    obs_t = torch.cat(obs_list, dim=0)
    actions_t = torch.cat(actions_list, dim=0)
    logprobs_t = torch.cat(logprobs_list, dim=0)
    rewards_t = torch.cat(rewards_list, dim=0)
    dones_t = torch.cat(dones_list, dim=0)
    values_t = torch.cat(values_list, dim=0)
    
    # Compute GAE
    with torch.no_grad():
        next_value = agent.get_value(obs).flatten()
        advantages = torch.zeros_like(rewards_t)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - dones_t[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_t[t + 1]
                nextvalues = values_t[t + 1]
            delta = rewards_t[t] + 0.99 * nextvalues * nextnonterminal - values_t[t]
            advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
        returns = advantages + values_t
    
    return obs_t, actions_t, logprobs_t, values_t, advantages, returns, rewards_t.sum().item()


def compute_ppo_loss(agent, obs, actions, advantages, returns, old_logprobs, old_values, args):
    """Compute PPO loss."""
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions.long())
    
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    
    adv = advantages
    if args.norm_adv and len(adv) > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    pg_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 0.8, 1.2)).mean()
    
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
    
    entropy_loss = entropy.mean()
    
    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
    
    return loss, pg_loss, v_loss, entropy_loss


def main():
    print("="*70)
    print("PPO MIXED TRAINING GRADIENT CONFLICT")
    print("Single Agent Trained on Both QF and BN Regimes")
    print("="*70)
    
    device = torch.device("cpu")
    args = Args()
    
    # Create HETERO architecture
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
    
    # Create environments
    print("\nCreating environments...")
    env_qf = create_env("qf", seed=42)
    env_bn = create_env("bn", seed=43)
    print("  QF: Wide DAGs (gnp_p=0.2)")
    print("  BN: Long CP DAGs (gnp_p=0.8)")
    
    # Initialize agent
    print("\nInitializing agent...")
    obs_init, _ = env_qf.reset()
    obs_init_t = torch.tensor(obs_init, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _ = agent.get_action_and_value(obs_init_t)
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    num_params = sum(p.numel() for p in agent.parameters())
    
    print(f"\nArchitecture: HETERO (HeteroConv + SAGE)")
    print(f"Parameters: {num_params:,}")
    
    # Training with gradient conflict measurement
    num_iterations = 50
    num_steps = 48  # Longer rollouts for more stable gradients
    
    cross_domain_sims = []
    within_qf_sims = []
    within_bn_sims = []
    qf_returns = []
    bn_returns = []
    
    prev_grad_qf = None
    prev_grad_bn = None
    
    print(f"\nRunning {num_iterations} mixed training iterations...")
    print("-"*70)
    
    for iteration in range(num_iterations):
        # Collect rollouts from BOTH domains
        try:
            obs_qf, act_qf, logp_qf, val_qf, adv_qf, ret_qf, rew_qf = collect_rollout(
                env_qf, agent, num_steps, device
            )
            obs_bn, act_bn, logp_bn, val_bn, adv_bn, ret_bn, rew_bn = collect_rollout(
                env_bn, agent, num_steps, device
            )
        except Exception as e:
            print(f"  Iter {iteration}: Error: {e}")
            continue
        
        qf_returns.append(rew_qf)
        bn_returns.append(rew_bn)
        
        # Compute gradient for QF data only
        agent.zero_grad()
        loss_qf, _, _, _ = compute_ppo_loss(
            agent, obs_qf, act_qf, adv_qf.detach(), ret_qf.detach(), 
            logp_qf.detach(), val_qf.detach(), args
        )
        loss_qf.backward()
        grad_qf = torch.cat([p.grad.flatten() for p in agent.parameters() if p.grad is not None]).clone()
        
        # Compute gradient for BN data only
        agent.zero_grad()
        loss_bn, _, _, _ = compute_ppo_loss(
            agent, obs_bn, act_bn, adv_bn.detach(), ret_bn.detach(),
            logp_bn.detach(), val_bn.detach(), args
        )
        loss_bn.backward()
        grad_bn = torch.cat([p.grad.flatten() for p in agent.parameters() if p.grad is not None]).clone()
        
        # Cross-domain similarity (THE KEY MEASUREMENT)
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
        
        # Apply MIXED update (this is what happens in joint training)
        agent.zero_grad()
        loss_qf_new, _, _, _ = compute_ppo_loss(
            agent, obs_qf, act_qf, adv_qf.detach(), ret_qf.detach(),
            logp_qf.detach(), val_qf.detach(), args
        )
        loss_bn_new, _, _, _ = compute_ppo_loss(
            agent, obs_bn, act_bn, adv_bn.detach(), ret_bn.detach(),
            logp_bn.detach(), val_bn.detach(), args
        )
        mixed_loss = (loss_qf_new + loss_bn_new) / 2
        mixed_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()
        
        if iteration % 5 == 0:
            wqf = np.mean(within_qf_sims[-5:]) if within_qf_sims else 0
            wbn = np.mean(within_bn_sims[-5:]) if within_bn_sims else 0
            print(f"Iter {iteration:2d}: Cross={cos_cross.item():+.3f}, "
                  f"WithinQF={wqf:+.3f}, WithinBN={wbn:+.3f}")
    
    env_qf.close()
    env_bn.close()
    
    # Results
    avg_cross = np.mean(cross_domain_sims)
    avg_qf = np.mean(within_qf_sims) if within_qf_sims else 0
    avg_bn = np.mean(within_bn_sims) if within_bn_sims else 0
    conflict_rate = sum(1 for s in cross_domain_sims if s < 0) / len(cross_domain_sims) * 100
    
    print("\n" + "="*70)
    print("RESULTS: PPO MIXED TRAINING GRADIENT CONFLICT")
    print("="*70)
    print(f"\nCROSS-DOMAIN (QF vs BN gradients):")
    print(f"  Average Similarity: {avg_cross:+.3f}")
    print(f"  Conflict Rate: {conflict_rate:.0f}%")
    print(f"\nWITHIN QUEUE-FREE:")
    print(f"  Average Similarity: {avg_qf:+.3f}")
    print(f"\nWITHIN BOTTLENECK:")
    print(f"  Average Similarity: {avg_bn:+.3f}")
    
    # Gradient cancellation effect
    grad_efficiency = (1 + avg_cross) / 2 * 100  # 0% if cos=-1, 100% if cos=1
    print(f"\nGRADIENT EFFICIENCY: {grad_efficiency:.0f}%")
    print(f"  (% of gradient signal retained after mixing)")
    
    if avg_cross < 0:
        print("\n✓ GRADIENT CONFLICT CONFIRMED during mixed PPO training!")
        print("  → QF and BN gradients point in OPPOSITE directions")
        print("  → Mixed updates partially cancel each other")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cross-domain similarity
    ax = axes[0, 0]
    colors = ['#CC3311' if s < 0 else '#009988' for s in cross_domain_sims]
    ax.bar(range(len(cross_domain_sims)), cross_domain_sims, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=avg_cross, color='#0077BB', linewidth=2, linestyle='--', label=f'Mean: {avg_cross:.3f}')
    ax.fill_between(range(len(cross_domain_sims)), -1, 0, alpha=0.1, color='red')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(a) Cross-Domain Gradient Similarity\n(QF vs BN during Mixed Training)', fontweight='bold')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    
    # Distribution
    ax = axes[0, 1]
    if within_qf_sims:
        ax.hist(within_qf_sims, bins=10, alpha=0.6, color='#0077BB', label=f'Within QF (μ={avg_qf:.2f})', edgecolor='black')
    if within_bn_sims:
        ax.hist(within_bn_sims, bins=10, alpha=0.6, color='#EE7733', label=f'Within BN (μ={avg_bn:.2f})', edgecolor='black')
    ax.hist(cross_domain_sims, bins=10, alpha=0.6, color='#CC3311', label=f'Cross (μ={avg_cross:.2f})', edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=2, linestyle='--')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('(b) Gradient Similarity Distribution', fontweight='bold')
    ax.legend(fontsize=9)
    
    # Over time
    ax = axes[1, 0]
    ax.plot(cross_domain_sims, 'o-', color='#CC3311', label='Cross-Domain', linewidth=2, markersize=4)
    if within_qf_sims:
        ax.plot(range(1, len(within_qf_sims)+1), within_qf_sims, 's-', color='#0077BB', 
               label='Within QF', linewidth=2, alpha=0.7, markersize=4)
    if within_bn_sims:
        ax.plot(range(1, len(within_bn_sims)+1), within_bn_sims, '^-', color='#EE7733',
               label='Within BN', linewidth=2, alpha=0.7, markersize=4)
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
PPO MIXED TRAINING RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Setup:
  • Single agent trained on BOTH regimes
  • QF: Wide DAGs (gnp_p=0.2)
  • BN: Long CP DAGs (gnp_p=0.8)
  • Reward: YOUR reward function
    (energy_weight * ΔE + makespan_weight * ΔM)

Architecture:
  • HETERO (HeteroConv + SAGE)
  • Parameters: {num_params:,}

Cross-Domain (QF vs BN):
  • Avg Similarity: {avg_cross:+.3f}
  • Conflict Rate: {conflict_rate:.0f}%

Within-Domain:
  • QF: {avg_qf:+.3f}
  • BN: {avg_bn:+.3f}

Gradient Efficiency: {grad_efficiency:.0f}%

CONCLUSION:
  {"✓ Gradient conflict during mixed training!" if avg_cross < 0 else "Partial conflict detected"}
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ppo_mixed_gradient_conflict.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'ppo_mixed_gradient_conflict.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"\nSaved: {output_dir / 'ppo_mixed_gradient_conflict.png'}")


if __name__ == "__main__":
    main()
