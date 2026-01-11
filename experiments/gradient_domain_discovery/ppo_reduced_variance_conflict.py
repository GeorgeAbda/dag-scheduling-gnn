"""
PPO Gradient Conflict with Reduced Variance

Uses multiple techniques to reduce gradient variance:
1. Average gradients over multiple rollouts per domain
2. Use deterministic policy for evaluation
3. Larger batch sizes
4. Compare policy gradients (actor) separately from value gradients (critic)
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


def create_env(domain: str, seed: int):
    """Create environment for a specific domain."""
    if domain == "qf":
        ds = DatasetArgs(
            dag_method="gnp", gnp_min_n=8, gnp_max_n=12, gnp_p=0.2,
            workflow_count=3, host_count=2, vm_count=4, style="wide",
        )
    else:
        ds = DatasetArgs(
            dag_method="gnp", gnp_min_n=12, gnp_max_n=20, gnp_p=0.8,
            workflow_count=5, host_count=2, vm_count=4, style="long_cp",
        )
    ds.seed = seed
    env = CloudSchedulingGymEnvironment(dataset_args=ds)
    env = GinAgentWrapper(env)
    return env


def collect_rollout(env, agent, num_steps, device, deterministic=False):
    """Collect rollout with optional deterministic policy."""
    obs_list, actions_list, logprobs_list = [], [], []
    rewards_list, dones_list, values_list = [], [], []
    
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    
    for step in range(num_steps):
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(obs, deterministic=deterministic)
        
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
    
    # GAE
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
    
    return obs_t, actions_t, logprobs_t, values_t, advantages, returns


def compute_actor_gradient(agent, obs, actions, advantages, old_logprobs):
    """Compute ONLY actor (policy) gradient."""
    _, newlogprob, entropy, _ = agent.get_action_and_value(obs, actions.long())
    
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    
    adv = advantages
    if len(adv) > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    # Policy loss only
    pg_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 0.8, 1.2)).mean()
    entropy_loss = entropy.mean()
    
    actor_loss = pg_loss - 0.01 * entropy_loss
    
    # Get only actor gradients
    agent.zero_grad()
    actor_loss.backward()
    
    actor_grads = []
    for name, p in agent.named_parameters():
        if 'actor' in name and p.grad is not None:
            actor_grads.append(p.grad.flatten())
    
    if actor_grads:
        return torch.cat(actor_grads).clone()
    else:
        return torch.cat([p.grad.flatten() for p in agent.parameters() if p.grad is not None]).clone()


def compute_critic_gradient(agent, obs, returns, old_values):
    """Compute ONLY critic (value) gradient."""
    _, _, _, newvalue = agent.get_action_and_value(obs)
    newvalue = newvalue.view(-1)
    
    v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
    
    agent.zero_grad()
    v_loss.backward()
    
    critic_grads = []
    for name, p in agent.named_parameters():
        if 'critic' in name and p.grad is not None:
            critic_grads.append(p.grad.flatten())
    
    if critic_grads:
        return torch.cat(critic_grads).clone()
    else:
        return torch.cat([p.grad.flatten() for p in agent.parameters() if p.grad is not None]).clone()


def main():
    print("="*70)
    print("PPO GRADIENT CONFLICT - REDUCED VARIANCE")
    print("Averaging over multiple rollouts per measurement")
    print("="*70)
    
    device = torch.device("cpu")
    
    # Create architecture
    variant = AG.AblationVariant(
        name='hetero', graph_type='hetero', hetero_base='sage',
        gin_num_layers=2, use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True,
    )
    
    agent = AG.AblationGinAgent(device, variant, hidden_dim=64, embedding_dim=32)
    
    # Create multiple environments per domain for averaging
    print("\nCreating environments...")
    num_envs_per_domain = 3
    envs_qf = [create_env("qf", seed=42+i) for i in range(num_envs_per_domain)]
    envs_bn = [create_env("bn", seed=100+i) for i in range(num_envs_per_domain)]
    print(f"  QF: {num_envs_per_domain} environments (Wide DAGs)")
    print(f"  BN: {num_envs_per_domain} environments (Long CP DAGs)")
    
    # Initialize agent
    print("\nInitializing agent...")
    obs_init, _ = envs_qf[0].reset()
    obs_init_t = torch.tensor(obs_init, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _ = agent.get_action_and_value(obs_init_t)
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    num_params = sum(p.numel() for p in agent.parameters())
    
    print(f"\nArchitecture: HETERO (HeteroConv + SAGE)")
    print(f"Parameters: {num_params:,}")
    
    # Measurement
    num_iterations = 30
    num_steps = 64
    
    actor_cross_sims = []
    actor_within_qf = []
    actor_within_bn = []
    critic_cross_sims = []
    
    prev_actor_qf = None
    prev_actor_bn = None
    
    print(f"\nRunning {num_iterations} iterations (averaging over {num_envs_per_domain} rollouts each)...")
    print("-"*70)
    
    for iteration in range(num_iterations):
        # Collect and average gradients from multiple QF environments
        qf_actor_grads = []
        qf_critic_grads = []
        
        for env in envs_qf:
            try:
                obs, act, logp, val, adv, ret = collect_rollout(env, agent, num_steps, device, deterministic=True)
                
                grad_actor = compute_actor_gradient(agent, obs, act, adv.detach(), logp.detach())
                qf_actor_grads.append(grad_actor)
                
                grad_critic = compute_critic_gradient(agent, obs, ret.detach(), val.detach())
                qf_critic_grads.append(grad_critic)
            except Exception as e:
                print(f"  QF env error: {e}")
                continue
        
        # Collect and average gradients from multiple BN environments
        bn_actor_grads = []
        bn_critic_grads = []
        
        for env in envs_bn:
            try:
                obs, act, logp, val, adv, ret = collect_rollout(env, agent, num_steps, device, deterministic=True)
                
                grad_actor = compute_actor_gradient(agent, obs, act, adv.detach(), logp.detach())
                bn_actor_grads.append(grad_actor)
                
                grad_critic = compute_critic_gradient(agent, obs, ret.detach(), val.detach())
                bn_critic_grads.append(grad_critic)
            except Exception as e:
                print(f"  BN env error: {e}")
                continue
        
        if not qf_actor_grads or not bn_actor_grads:
            continue
        
        # Average gradients
        avg_qf_actor = torch.stack(qf_actor_grads).mean(dim=0)
        avg_bn_actor = torch.stack(bn_actor_grads).mean(dim=0)
        avg_qf_critic = torch.stack(qf_critic_grads).mean(dim=0)
        avg_bn_critic = torch.stack(bn_critic_grads).mean(dim=0)
        
        # Cross-domain similarity (ACTOR)
        cos_actor = torch.dot(avg_qf_actor, avg_bn_actor) / (avg_qf_actor.norm() * avg_bn_actor.norm() + 1e-8)
        actor_cross_sims.append(cos_actor.item())
        
        # Cross-domain similarity (CRITIC)
        cos_critic = torch.dot(avg_qf_critic, avg_bn_critic) / (avg_qf_critic.norm() * avg_bn_critic.norm() + 1e-8)
        critic_cross_sims.append(cos_critic.item())
        
        # Within-domain (ACTOR)
        if prev_actor_qf is not None:
            cos_qf = torch.dot(avg_qf_actor, prev_actor_qf) / (avg_qf_actor.norm() * prev_actor_qf.norm() + 1e-8)
            actor_within_qf.append(cos_qf.item())
        if prev_actor_bn is not None:
            cos_bn = torch.dot(avg_bn_actor, prev_actor_bn) / (avg_bn_actor.norm() * prev_actor_bn.norm() + 1e-8)
            actor_within_bn.append(cos_bn.item())
        
        prev_actor_qf = avg_qf_actor.clone()
        prev_actor_bn = avg_bn_actor.clone()
        
        # Apply mixed update
        agent.zero_grad()
        combined_grad = (avg_qf_actor + avg_bn_actor) / 2
        # Manual gradient application
        offset = 0
        for name, p in agent.named_parameters():
            if 'actor' in name and p.requires_grad:
                numel = p.numel()
                if offset + numel <= combined_grad.numel():
                    p.grad = combined_grad[offset:offset+numel].view_as(p)
                    offset += numel
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()
        
        if iteration % 5 == 0:
            wqf = np.mean(actor_within_qf[-5:]) if actor_within_qf else 0
            wbn = np.mean(actor_within_bn[-5:]) if actor_within_bn else 0
            print(f"Iter {iteration:2d}: Actor_Cross={cos_actor.item():+.3f}, "
                  f"Critic_Cross={cos_critic.item():+.3f}, "
                  f"WithinQF={wqf:+.3f}, WithinBN={wbn:+.3f}")
    
    # Close environments
    for env in envs_qf + envs_bn:
        env.close()
    
    # Results
    avg_actor_cross = np.mean(actor_cross_sims)
    avg_critic_cross = np.mean(critic_cross_sims)
    avg_within_qf = np.mean(actor_within_qf) if actor_within_qf else 0
    avg_within_bn = np.mean(actor_within_bn) if actor_within_bn else 0
    actor_conflict_rate = sum(1 for s in actor_cross_sims if s < 0) / len(actor_cross_sims) * 100
    critic_conflict_rate = sum(1 for s in critic_cross_sims if s < 0) / len(critic_cross_sims) * 100
    
    print("\n" + "="*70)
    print("RESULTS: REDUCED VARIANCE GRADIENT CONFLICT")
    print("="*70)
    print(f"\nACTOR (Policy) Gradients:")
    print(f"  Cross-Domain Similarity: {avg_actor_cross:+.3f}")
    print(f"  Conflict Rate: {actor_conflict_rate:.0f}%")
    print(f"  Within-QF: {avg_within_qf:+.3f}")
    print(f"  Within-BN: {avg_within_bn:+.3f}")
    print(f"\nCRITIC (Value) Gradients:")
    print(f"  Cross-Domain Similarity: {avg_critic_cross:+.3f}")
    print(f"  Conflict Rate: {critic_conflict_rate:.0f}%")
    
    if avg_actor_cross < 0:
        print("\n✓ ACTOR GRADIENT CONFLICT CONFIRMED!")
    elif avg_actor_cross < avg_within_qf and avg_actor_cross < avg_within_bn:
        print("\n○ Cross-domain lower than within-domain (partial conflict)")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Actor cross-domain
    ax = axes[0, 0]
    colors = ['#CC3311' if s < 0 else '#009988' for s in actor_cross_sims]
    ax.bar(range(len(actor_cross_sims)), actor_cross_sims, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=avg_actor_cross, color='#0077BB', linewidth=2, linestyle='--', label=f'Mean: {avg_actor_cross:.3f}')
    ax.fill_between(range(len(actor_cross_sims)), -1, 0, alpha=0.1, color='red')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(a) ACTOR Cross-Domain Similarity\n(Averaged over multiple rollouts)', fontweight='bold')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    
    # Critic cross-domain
    ax = axes[0, 1]
    colors = ['#CC3311' if s < 0 else '#009988' for s in critic_cross_sims]
    ax.bar(range(len(critic_cross_sims)), critic_cross_sims, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=avg_critic_cross, color='#0077BB', linewidth=2, linestyle='--', label=f'Mean: {avg_critic_cross:.3f}')
    ax.fill_between(range(len(critic_cross_sims)), -1, 0, alpha=0.1, color='red')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(b) CRITIC Cross-Domain Similarity', fontweight='bold')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    
    # Distribution comparison
    ax = axes[1, 0]
    if actor_within_qf:
        ax.hist(actor_within_qf, bins=10, alpha=0.5, color='#0077BB', label=f'Within QF (μ={avg_within_qf:.2f})', edgecolor='black')
    if actor_within_bn:
        ax.hist(actor_within_bn, bins=10, alpha=0.5, color='#EE7733', label=f'Within BN (μ={avg_within_bn:.2f})', edgecolor='black')
    ax.hist(actor_cross_sims, bins=10, alpha=0.5, color='#CC3311', label=f'Cross (μ={avg_actor_cross:.2f})', edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=2, linestyle='--')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('(c) Actor Gradient Distribution', fontweight='bold')
    ax.legend(fontsize=9)
    
    # Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""
REDUCED VARIANCE RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Variance Reduction:
  • {num_envs_per_domain} rollouts averaged per domain
  • Deterministic policy evaluation
  • {num_steps} steps per rollout

Architecture:
  • HETERO (HeteroConv + SAGE)
  • Parameters: {num_params:,}

ACTOR (Policy) Gradients:
  • Cross-Domain: {avg_actor_cross:+.3f}
  • Conflict Rate: {actor_conflict_rate:.0f}%
  • Within-QF: {avg_within_qf:+.3f}
  • Within-BN: {avg_within_bn:+.3f}

CRITIC (Value) Gradients:
  • Cross-Domain: {avg_critic_cross:+.3f}
  • Conflict Rate: {critic_conflict_rate:.0f}%

CONCLUSION:
  {"✓ Actor gradient conflict!" if avg_actor_cross < 0 else "Cross < Within" if avg_actor_cross < min(avg_within_qf, avg_within_bn) else "High variance remains"}
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ppo_reduced_variance_conflict.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'ppo_reduced_variance_conflict.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"\nSaved: {output_dir / 'ppo_reduced_variance_conflict.png'}")


if __name__ == "__main__":
    main()
