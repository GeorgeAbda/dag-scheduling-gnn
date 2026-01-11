"""
Parameter Scaling vs Gradient Conflict

Experiment: Increase model size and measure how gradient conflict changes.
Tests multiple hidden_dim / embedding_dim configurations.
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


def collect_rollout(env, agent, num_steps, device):
    """Collect rollout."""
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


def compute_ppo_gradient(agent, obs, actions, advantages, old_logprobs, returns, old_values):
    """Compute full PPO gradient."""
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions.long())
    
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    
    adv = advantages
    if len(adv) > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    pg_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 0.8, 1.2)).mean()
    
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
    
    entropy_loss = entropy.mean()
    
    loss = pg_loss - 0.01 * entropy_loss + 0.5 * v_loss
    
    agent.zero_grad()
    loss.backward()
    
    return torch.cat([p.grad.flatten() for p in agent.parameters() if p.grad is not None]).clone()


def run_experiment_for_config(hidden_dim, embedding_dim, num_iterations=20, num_steps=48):
    """Run gradient conflict experiment for a specific model size."""
    
    device = torch.device("cpu")
    
    # Create architecture
    variant = AG.AblationVariant(
        name='hetero', graph_type='hetero', hetero_base='sage',
        gin_num_layers=2, use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True,
    )
    
    agent = AG.AblationGinAgent(device, variant, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    
    # Create environments
    env_qf = create_env("qf", seed=42)
    env_bn = create_env("bn", seed=43)
    
    # Initialize agent
    obs_init, _ = env_qf.reset()
    obs_init_t = torch.tensor(obs_init, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _ = agent.get_action_and_value(obs_init_t)
    
    num_params = sum(p.numel() for p in agent.parameters())
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    
    # Collect metrics
    cross_sims = []
    within_qf = []
    within_bn = []
    
    prev_grad_qf = None
    prev_grad_bn = None
    
    for iteration in range(num_iterations):
        try:
            # Collect rollouts
            obs_qf, act_qf, logp_qf, val_qf, adv_qf, ret_qf = collect_rollout(env_qf, agent, num_steps, device)
            obs_bn, act_bn, logp_bn, val_bn, adv_bn, ret_bn = collect_rollout(env_bn, agent, num_steps, device)
            
            # Compute gradients
            grad_qf = compute_ppo_gradient(agent, obs_qf, act_qf, adv_qf.detach(), logp_qf.detach(), ret_qf.detach(), val_qf.detach())
            grad_bn = compute_ppo_gradient(agent, obs_bn, act_bn, adv_bn.detach(), logp_bn.detach(), ret_bn.detach(), val_bn.detach())
            
            # Cross-domain
            cos_cross = torch.dot(grad_qf, grad_bn) / (grad_qf.norm() * grad_bn.norm() + 1e-8)
            cross_sims.append(cos_cross.item())
            
            # Within-domain
            if prev_grad_qf is not None:
                cos_qf = torch.dot(grad_qf, prev_grad_qf) / (grad_qf.norm() * prev_grad_qf.norm() + 1e-8)
                within_qf.append(cos_qf.item())
            if prev_grad_bn is not None:
                cos_bn = torch.dot(grad_bn, prev_grad_bn) / (grad_bn.norm() * prev_grad_bn.norm() + 1e-8)
                within_bn.append(cos_bn.item())
            
            prev_grad_qf = grad_qf.clone()
            prev_grad_bn = grad_bn.clone()
            
            # Update
            agent.zero_grad()
            combined = (grad_qf + grad_bn) / 2
            offset = 0
            for p in agent.parameters():
                if p.requires_grad:
                    numel = p.numel()
                    if offset + numel <= combined.numel():
                        p.grad = combined[offset:offset+numel].view_as(p)
                        offset += numel
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            
        except Exception as e:
            continue
    
    env_qf.close()
    env_bn.close()
    
    # Compute metrics
    avg_cross = np.mean(cross_sims) if cross_sims else 0
    conflict_rate = sum(1 for s in cross_sims if s < 0) / len(cross_sims) * 100 if cross_sims else 0
    avg_within_qf = np.mean(within_qf) if within_qf else 0
    avg_within_bn = np.mean(within_bn) if within_bn else 0
    
    return {
        'num_params': num_params,
        'hidden_dim': hidden_dim,
        'embedding_dim': embedding_dim,
        'cross_domain_sim': avg_cross,
        'conflict_rate': conflict_rate,
        'within_qf': avg_within_qf,
        'within_bn': avg_within_bn,
    }


def main():
    print("="*70)
    print("PARAMETER SCALING vs GRADIENT CONFLICT")
    print("="*70)
    
    # Different model sizes to test
    configs = [
        (16, 8),    # Tiny
        (32, 16),   # Small
        (48, 24),   # Medium-Small
        (64, 32),   # Medium (default)
        (96, 48),   # Medium-Large
        (128, 64),  # Large
        (192, 96),  # XL
    ]
    
    results = []
    
    for hidden_dim, embedding_dim in configs:
        print(f"\n{'='*50}")
        print(f"Testing: hidden_dim={hidden_dim}, embedding_dim={embedding_dim}")
        print(f"{'='*50}")
        
        result = run_experiment_for_config(hidden_dim, embedding_dim, num_iterations=20, num_steps=48)
        results.append(result)
        
        print(f"  Parameters: {result['num_params']:,}")
        print(f"  Cross-Domain Sim: {result['cross_domain_sim']:+.3f}")
        print(f"  Conflict Rate: {result['conflict_rate']:.0f}%")
        print(f"  Within-QF: {result['within_qf']:+.3f}")
        print(f"  Within-BN: {result['within_bn']:+.3f}")
    
    # Extract data for plotting
    params = [r['num_params'] for r in results]
    cross_sims = [r['cross_domain_sim'] for r in results]
    conflict_rates = [r['conflict_rate'] for r in results]
    within_qf = [r['within_qf'] for r in results]
    within_bn = [r['within_bn'] for r in results]
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cross-domain similarity vs params
    ax = axes[0, 0]
    ax.plot(params, cross_sims, 'o-', color='#CC3311', linewidth=2, markersize=8, label='Cross-Domain')
    ax.plot(params, within_qf, 's--', color='#0077BB', linewidth=2, markersize=6, alpha=0.7, label='Within-QF')
    ax.plot(params, within_bn, '^--', color='#EE7733', linewidth=2, markersize=6, alpha=0.7, label='Within-BN')
    ax.axhline(y=0, color='black', linewidth=1, linestyle='--')
    ax.fill_between(params, -1, 0, alpha=0.1, color='red')
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(a) Gradient Similarity vs Model Size', fontweight='bold')
    ax.legend()
    ax.set_xscale('log')
    ax.set_ylim(-0.5, 1.0)
    
    # Conflict rate vs params
    ax = axes[0, 1]
    ax.bar(range(len(params)), conflict_rates, color='#CC3311', edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels([f'{p//1000}K' for p in params], rotation=45)
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Conflict Rate (%)')
    ax.set_title('(b) Conflict Rate vs Model Size', fontweight='bold')
    ax.axhline(y=50, color='black', linewidth=1, linestyle='--', label='50% baseline')
    ax.legend()
    
    # Gap between cross and within
    ax = axes[1, 0]
    avg_within = [(qf + bn) / 2 for qf, bn in zip(within_qf, within_bn)]
    gap = [w - c for w, c in zip(avg_within, cross_sims)]
    colors = ['#009988' if g > 0 else '#CC3311' for g in gap]
    ax.bar(range(len(params)), gap, color=colors, edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels([f'{p//1000}K' for p in params], rotation=45)
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Within - Cross (Gap)')
    ax.set_title('(c) Within-Domain vs Cross-Domain Gap\n(Positive = Conflict)', fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=2)
    
    # Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create table data
    table_data = []
    for r in results:
        table_data.append([
            f"{r['num_params']//1000}K",
            f"{r['cross_domain_sim']:+.2f}",
            f"{r['conflict_rate']:.0f}%",
            f"{r['within_qf']:+.2f}",
            f"{r['within_bn']:+.2f}",
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Params', 'Cross', 'Conflict%', 'Within-QF', 'Within-BN'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color cells based on conflict
    for i, r in enumerate(results):
        if r['cross_domain_sim'] < 0:
            table[(i+1, 1)].set_facecolor('#FFCCCC')
        if r['conflict_rate'] > 50:
            table[(i+1, 2)].set_facecolor('#FFCCCC')
    
    ax.set_title('(d) Summary Table', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'param_scaling_conflict.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'param_scaling_conflict.png', dpi=300, facecolor='white')
    plt.close()
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nSaved: {output_dir / 'param_scaling_conflict.png'}")
    
    # Print summary
    print("\nSUMMARY:")
    print("-"*50)
    for r in results:
        print(f"  {r['num_params']:>7,} params: Cross={r['cross_domain_sim']:+.2f}, Conflict={r['conflict_rate']:.0f}%")


if __name__ == "__main__":
    main()
