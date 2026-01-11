"""
Actor-Only Gradient Conflict Measurement

Focus on ACTOR (policy) gradients only.
Aggregate over multiple rollouts to reduce variance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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


def collect_states_actions(env, agent, num_steps, device):
    """Collect states and actions from environment."""
    states = []
    actions = []
    rewards = []
    
    obs, _ = env.reset()
    
    for _ in range(num_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_t)
        
        states.append(obs_t)
        actions.append(action)
        
        next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy().item())
        rewards.append(reward)
        
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    
    return torch.cat(states), torch.cat(actions), torch.tensor(rewards, dtype=torch.float32, device=device)


def compute_actor_gradient(agent, states, actions, rewards):
    """
    Compute ONLY actor gradient using policy gradient.
    """
    agent.zero_grad()
    
    # Get log probabilities
    _, logprob, entropy, _ = agent.get_action_and_value(states, actions.long())
    
    # Normalize rewards as advantages
    if len(rewards) > 1:
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    else:
        adv = rewards
    
    # Policy gradient loss
    pg_loss = -(logprob * adv).mean()
    entropy_loss = entropy.mean()
    
    actor_loss = pg_loss - 0.01 * entropy_loss
    actor_loss.backward()
    
    # Extract ONLY actor gradients
    actor_grads = []
    for name, p in agent.named_parameters():
        if 'actor' in name and p.grad is not None:
            actor_grads.append(p.grad.flatten())
    
    if actor_grads:
        return torch.cat(actor_grads).clone()
    else:
        # Fallback to all gradients
        return torch.cat([p.grad.flatten() for p in agent.parameters() if p.grad is not None]).clone()


def main():
    print("="*70)
    print("ACTOR-ONLY GRADIENT CONFLICT")
    print("="*70)
    
    device = torch.device("cpu")
    
    # Create architecture
    variant = AG.AblationVariant(
        name='hetero', graph_type='hetero', hetero_base='sage',
        gin_num_layers=2, use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True,
    )
    
    agent = AG.AblationGinAgent(device, variant, hidden_dim=64, embedding_dim=32)
    
    # Create environments
    num_envs = 3
    print(f"\nCreating {num_envs} environments per domain...")
    envs_qf = [create_env("qf", seed=42+i) for i in range(num_envs)]
    envs_bn = [create_env("bn", seed=100+i) for i in range(num_envs)]
    
    # Initialize
    obs_init, _ = envs_qf[0].reset()
    obs_init_t = torch.tensor(obs_init, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _ = agent.get_action_and_value(obs_init_t)
    
    # Count actor parameters
    actor_params = sum(p.numel() for name, p in agent.named_parameters() if 'actor' in name)
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Actor parameters: {actor_params:,} / {total_params:,} total")
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    
    # Experiment
    num_iterations = 15
    num_rollouts_per_env = 2
    num_steps = 40
    
    cross_sims = []
    within_qf = []
    within_bn = []
    
    prev_qf = None
    prev_bn = None
    
    print(f"\nRunning {num_iterations} iterations...")
    print(f"  {num_envs} envs x {num_rollouts_per_env} rollouts x {num_steps} steps = {num_envs * num_rollouts_per_env * num_steps} transitions aggregated")
    print("-"*70)
    
    for iteration in range(num_iterations):
        # Aggregate QF actor gradients
        qf_grads = []
        for env in envs_qf:
            for _ in range(num_rollouts_per_env):
                try:
                    states, actions, rewards = collect_states_actions(env, agent, num_steps, device)
                    grad = compute_actor_gradient(agent, states, actions, rewards)
                    qf_grads.append(grad)
                except Exception as e:
                    continue
        
        # Aggregate BN actor gradients
        bn_grads = []
        for env in envs_bn:
            for _ in range(num_rollouts_per_env):
                try:
                    states, actions, rewards = collect_states_actions(env, agent, num_steps, device)
                    grad = compute_actor_gradient(agent, states, actions, rewards)
                    bn_grads.append(grad)
                except Exception as e:
                    continue
        
        if not qf_grads or not bn_grads:
            continue
        
        # Average
        avg_qf = torch.stack(qf_grads).mean(dim=0)
        avg_bn = torch.stack(bn_grads).mean(dim=0)
        
        # Cross-domain
        cos_cross = torch.dot(avg_qf, avg_bn) / (avg_qf.norm() * avg_bn.norm() + 1e-8)
        cross_sims.append(cos_cross.item())
        
        # Within-domain
        if prev_qf is not None:
            cos_qf = torch.dot(avg_qf, prev_qf) / (avg_qf.norm() * prev_qf.norm() + 1e-8)
            within_qf.append(cos_qf.item())
        if prev_bn is not None:
            cos_bn = torch.dot(avg_bn, prev_bn) / (avg_bn.norm() * prev_bn.norm() + 1e-8)
            within_bn.append(cos_bn.item())
        
        prev_qf = avg_qf.clone()
        prev_bn = avg_bn.clone()
        
        # Update
        agent.zero_grad()
        combined = (avg_qf + avg_bn) / 2
        offset = 0
        for name, p in agent.named_parameters():
            if 'actor' in name and p.requires_grad:
                numel = p.numel()
                if offset + numel <= combined.numel():
                    p.grad = combined[offset:offset+numel].view_as(p)
                    offset += numel
        optimizer.step()
        
        wqf = np.mean(within_qf[-3:]) if within_qf else 0
        wbn = np.mean(within_bn[-3:]) if within_bn else 0
        print(f"Iter {iteration:2d}: Cross={cos_cross.item():+.3f}, WithinQF={wqf:+.3f}, WithinBN={wbn:+.3f}")
    
    # Close
    for env in envs_qf + envs_bn:
        env.close()
    
    # Results
    avg_cross = np.mean(cross_sims)
    std_cross = np.std(cross_sims)
    avg_wqf = np.mean(within_qf) if within_qf else 0
    avg_wbn = np.mean(within_bn) if within_bn else 0
    conflict_rate = sum(1 for s in cross_sims if s < 0) / len(cross_sims) * 100
    
    print("\n" + "="*70)
    print("ACTOR GRADIENT CONFLICT RESULTS")
    print("="*70)
    print(f"\nCross-Domain: {avg_cross:+.3f} ± {std_cross:.3f}")
    print(f"Conflict Rate: {conflict_rate:.0f}%")
    print(f"Within-QF: {avg_wqf:+.3f}")
    print(f"Within-BN: {avg_wbn:+.3f}")
    
    gap = (avg_wqf + avg_wbn) / 2 - avg_cross
    print(f"\nGap (Within - Cross): {gap:+.3f}")
    
    if avg_cross < 0:
        print("\n✓ ACTOR GRADIENT CONFLICT CONFIRMED!")
    elif gap > 0.1:
        print("\n○ Cross < Within: Partial conflict detected")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Cross-domain over iterations
    ax = axes[0]
    colors = ['#CC3311' if s < 0 else '#009988' for s in cross_sims]
    ax.bar(range(len(cross_sims)), cross_sims, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=avg_cross, color='#0077BB', linewidth=2, linestyle='--', label=f'Mean: {avg_cross:.3f}')
    ax.fill_between(range(len(cross_sims)), -1, 0, alpha=0.1, color='red')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(a) Actor Cross-Domain Similarity', fontweight='bold')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    
    # Distribution
    ax = axes[1]
    if within_qf:
        ax.hist(within_qf, bins=8, alpha=0.6, color='#0077BB', label=f'Within-QF (μ={avg_wqf:.2f})', edgecolor='black')
    if within_bn:
        ax.hist(within_bn, bins=8, alpha=0.6, color='#EE7733', label=f'Within-BN (μ={avg_wbn:.2f})', edgecolor='black')
    ax.hist(cross_sims, bins=8, alpha=0.6, color='#CC3311', label=f'Cross (μ={avg_cross:.2f})', edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=2, linestyle='--')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('(b) Actor Gradient Distribution', fontweight='bold')
    ax.legend(fontsize=9)
    
    # Summary
    ax = axes[2]
    ax.axis('off')
    summary = f"""
ACTOR GRADIENT CONFLICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Setup:
  • Actor parameters: {actor_params:,}
  • Aggregated: {num_envs * num_rollouts_per_env * num_steps} transitions

Results:
  • Cross-Domain: {avg_cross:+.3f} ± {std_cross:.3f}
  • Conflict Rate: {conflict_rate:.0f}%
  • Within-QF: {avg_wqf:+.3f}
  • Within-BN: {avg_wbn:+.3f}

Gap (Within - Cross): {gap:+.3f}

CONCLUSION:
  {"✓ Actor conflict confirmed!" if avg_cross < 0 else "○ Cross < Within" if gap > 0.05 else "High variance"}
"""
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'actor_gradient_conflict.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'actor_gradient_conflict.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"\nSaved: {output_dir / 'actor_gradient_conflict.png'}")


if __name__ == "__main__":
    main()
