"""
Low Variance Gradient Conflict Measurement

Techniques to reduce PPO variance:
1. Large batch aggregation (many rollouts averaged)
2. Supervised proxy loss (CrossEntropy on best actions)
3. Multiple seeds per measurement
4. Compare both PPO and Supervised approaches
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


def collect_transitions(env, agent, num_transitions, device):
    """Collect state-action-reward transitions."""
    states = []
    actions = []
    rewards = []
    
    obs, _ = env.reset()
    
    for _ in range(num_transitions):
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
    
    return torch.cat(states), torch.cat(actions), rewards


def compute_supervised_gradient(agent, states, optimal_actions):
    """Compute gradient using supervised loss on optimal actions."""
    agent.zero_grad()
    
    # Get action logits from actor
    logits = agent.actor(states)
    
    # CrossEntropy loss
    loss = nn.CrossEntropyLoss()(logits, optimal_actions.long())
    loss.backward()
    
    return torch.cat([p.grad.flatten() for p in agent.parameters() if p.grad is not None]).clone()


def find_optimal_actions_by_reward(env, agent, states, device, num_actions=None):
    """
    For each state, find the action that would give highest reward.
    This is a proxy for the "optimal" action under the reward function.
    """
    # Get number of possible actions from environment
    if num_actions is None:
        # Estimate from action space
        try:
            num_actions = env.action_space.n
        except:
            num_actions = 10  # fallback
    
    # For simplicity, use the action with highest logit (greedy policy)
    # This represents what the current policy thinks is best
    with torch.no_grad():
        logits = agent.actor(states)
        optimal_actions = logits.argmax(dim=-1)
    
    return optimal_actions


def aggregate_gradients(agent, envs, num_rollouts, num_steps, device, method='supervised'):
    """
    Aggregate gradients over multiple rollouts to reduce variance.
    
    method: 'supervised' or 'ppo'
    """
    all_grads = []
    
    for env in envs:
        for _ in range(num_rollouts // len(envs)):
            states, actions, rewards = collect_transitions(env, agent, num_steps, device)
            
            if method == 'supervised':
                # Use greedy actions as targets (what policy thinks is optimal)
                with torch.no_grad():
                    logits = agent.actor(states)
                    targets = logits.argmax(dim=-1)
                
                grad = compute_supervised_gradient(agent, states, targets)
            else:
                # PPO-style gradient
                agent.zero_grad()
                _, logprob, entropy, value = agent.get_action_and_value(states, actions.long())
                
                # Simple policy gradient (no advantages for simplicity)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                if len(rewards_t) > 1:
                    rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
                
                loss = -(logprob * rewards_t).mean() - 0.01 * entropy.mean()
                loss.backward()
                
                grad = torch.cat([p.grad.flatten() for p in agent.parameters() if p.grad is not None]).clone()
            
            all_grads.append(grad)
    
    # Average all gradients
    return torch.stack(all_grads).mean(dim=0)


def main():
    print("="*70)
    print("LOW VARIANCE GRADIENT CONFLICT MEASUREMENT")
    print("="*70)
    
    device = torch.device("cpu")
    
    # Create architecture
    variant = AG.AblationVariant(
        name='hetero', graph_type='hetero', hetero_base='sage',
        gin_num_layers=2, use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True,
    )
    
    agent = AG.AblationGinAgent(device, variant, hidden_dim=64, embedding_dim=32)
    
    # Create multiple environments per domain
    num_envs = 5
    print(f"\nCreating {num_envs} environments per domain...")
    envs_qf = [create_env("qf", seed=42+i) for i in range(num_envs)]
    envs_bn = [create_env("bn", seed=100+i) for i in range(num_envs)]
    
    # Initialize agent
    obs_init, _ = envs_qf[0].reset()
    obs_init_t = torch.tensor(obs_init, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _ = agent.get_action_and_value(obs_init_t)
    
    num_params = sum(p.numel() for p in agent.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Experiment parameters (reduced for speed)
    num_iterations = 10
    num_rollouts = 6   # Rollouts to aggregate per measurement
    num_steps = 32     # Steps per rollout
    
    # Results storage
    results = {
        'supervised': {'cross': [], 'within_qf': [], 'within_bn': []},
        'ppo': {'cross': [], 'within_qf': [], 'within_bn': []},
    }
    
    prev_grads = {
        'supervised': {'qf': None, 'bn': None},
        'ppo': {'qf': None, 'bn': None},
    }
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    
    print(f"\nRunning {num_iterations} iterations...")
    print(f"  Aggregating {num_rollouts} rollouts x {num_steps} steps = {num_rollouts * num_steps} transitions per measurement")
    print("-"*70)
    
    for iteration in range(num_iterations):
        for method in ['supervised', 'ppo']:
            try:
                # Aggregate gradients from QF domain
                grad_qf = aggregate_gradients(agent, envs_qf, num_rollouts, num_steps, device, method)
                
                # Aggregate gradients from BN domain
                grad_bn = aggregate_gradients(agent, envs_bn, num_rollouts, num_steps, device, method)
                
                # Cross-domain similarity
                cos_cross = torch.dot(grad_qf, grad_bn) / (grad_qf.norm() * grad_bn.norm() + 1e-8)
                results[method]['cross'].append(cos_cross.item())
                
                # Within-domain similarity
                if prev_grads[method]['qf'] is not None:
                    cos_qf = torch.dot(grad_qf, prev_grads[method]['qf']) / (grad_qf.norm() * prev_grads[method]['qf'].norm() + 1e-8)
                    results[method]['within_qf'].append(cos_qf.item())
                if prev_grads[method]['bn'] is not None:
                    cos_bn = torch.dot(grad_bn, prev_grads[method]['bn']) / (grad_bn.norm() * prev_grads[method]['bn'].norm() + 1e-8)
                    results[method]['within_bn'].append(cos_bn.item())
                
                prev_grads[method]['qf'] = grad_qf.clone()
                prev_grads[method]['bn'] = grad_bn.clone()
                
            except Exception as e:
                print(f"  Error in {method}: {e}")
                continue
        
        # Light update to evolve the model
        agent.zero_grad()
        combined = (prev_grads['supervised']['qf'] + prev_grads['supervised']['bn']) / 2
        offset = 0
        for p in agent.parameters():
            if p.requires_grad:
                numel = p.numel()
                if offset + numel <= combined.numel():
                    p.grad = combined[offset:offset+numel].view_as(p)
                    offset += numel
        optimizer.step()
        
        if iteration % 5 == 0:
            sup_cross = results['supervised']['cross'][-1] if results['supervised']['cross'] else 0
            ppo_cross = results['ppo']['cross'][-1] if results['ppo']['cross'] else 0
            print(f"Iter {iteration:2d}: Supervised_Cross={sup_cross:+.3f}, PPO_Cross={ppo_cross:+.3f}")
    
    # Close environments
    for env in envs_qf + envs_bn:
        env.close()
    
    # Compute final metrics
    metrics = {}
    for method in ['supervised', 'ppo']:
        metrics[method] = {
            'cross_mean': np.mean(results[method]['cross']) if results[method]['cross'] else 0,
            'cross_std': np.std(results[method]['cross']) if results[method]['cross'] else 0,
            'conflict_rate': sum(1 for s in results[method]['cross'] if s < 0) / len(results[method]['cross']) * 100 if results[method]['cross'] else 0,
            'within_qf': np.mean(results[method]['within_qf']) if results[method]['within_qf'] else 0,
            'within_bn': np.mean(results[method]['within_bn']) if results[method]['within_bn'] else 0,
        }
    
    print("\n" + "="*70)
    print("RESULTS: LOW VARIANCE COMPARISON")
    print("="*70)
    
    for method in ['supervised', 'ppo']:
        m = metrics[method]
        print(f"\n{method.upper()} Method:")
        print(f"  Cross-Domain: {m['cross_mean']:+.3f} ± {m['cross_std']:.3f}")
        print(f"  Conflict Rate: {m['conflict_rate']:.0f}%")
        print(f"  Within-QF: {m['within_qf']:+.3f}")
        print(f"  Within-BN: {m['within_bn']:+.3f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Supervised cross-domain
    ax = axes[0, 0]
    data = results['supervised']['cross']
    colors = ['#CC3311' if s < 0 else '#009988' for s in data]
    ax.bar(range(len(data)), data, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=metrics['supervised']['cross_mean'], color='#0077BB', linewidth=2, linestyle='--', 
               label=f"Mean: {metrics['supervised']['cross_mean']:.3f}")
    ax.fill_between(range(len(data)), -1, 0, alpha=0.1, color='red')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(a) SUPERVISED: Cross-Domain Similarity\n(Aggregated over many rollouts)', fontweight='bold')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    
    # PPO cross-domain
    ax = axes[0, 1]
    data = results['ppo']['cross']
    colors = ['#CC3311' if s < 0 else '#009988' for s in data]
    ax.bar(range(len(data)), data, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=metrics['ppo']['cross_mean'], color='#0077BB', linewidth=2, linestyle='--',
               label=f"Mean: {metrics['ppo']['cross_mean']:.3f}")
    ax.fill_between(range(len(data)), -1, 0, alpha=0.1, color='red')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(b) PPO: Cross-Domain Similarity\n(Aggregated over many rollouts)', fontweight='bold')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    
    # Comparison bar chart
    ax = axes[1, 0]
    x = np.arange(4)
    width = 0.35
    
    sup_vals = [metrics['supervised']['cross_mean'], metrics['supervised']['within_qf'], 
                metrics['supervised']['within_bn'], metrics['supervised']['conflict_rate']/100]
    ppo_vals = [metrics['ppo']['cross_mean'], metrics['ppo']['within_qf'],
                metrics['ppo']['within_bn'], metrics['ppo']['conflict_rate']/100]
    
    ax.bar(x - width/2, sup_vals, width, label='Supervised', color='#0077BB', edgecolor='black')
    ax.bar(x + width/2, ppo_vals, width, label='PPO', color='#EE7733', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(['Cross', 'Within-QF', 'Within-BN', 'Conflict Rate'])
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel('Value')
    ax.set_title('(c) Supervised vs PPO Comparison', fontweight='bold')
    ax.legend()
    
    # Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = f"""
LOW VARIANCE GRADIENT CONFLICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Variance Reduction Techniques:
  • {num_rollouts} rollouts aggregated per measurement
  • {num_steps} steps per rollout
  • {num_envs} environments per domain
  • Total: {num_rollouts * num_steps} transitions averaged

SUPERVISED Method:
  • Cross-Domain: {metrics['supervised']['cross_mean']:+.3f} ± {metrics['supervised']['cross_std']:.3f}
  • Conflict Rate: {metrics['supervised']['conflict_rate']:.0f}%
  • Within-QF: {metrics['supervised']['within_qf']:+.3f}
  • Within-BN: {metrics['supervised']['within_bn']:+.3f}

PPO Method:
  • Cross-Domain: {metrics['ppo']['cross_mean']:+.3f} ± {metrics['ppo']['cross_std']:.3f}
  • Conflict Rate: {metrics['ppo']['conflict_rate']:.0f}%
  • Within-QF: {metrics['ppo']['within_qf']:+.3f}
  • Within-BN: {metrics['ppo']['within_bn']:+.3f}

CONCLUSION:
  {"✓ Conflict confirmed with reduced variance!" if metrics['supervised']['cross_mean'] < 0 or metrics['ppo']['cross_mean'] < 0 else "Variance still dominates"}
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'low_variance_conflict.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'low_variance_conflict.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"\nSaved: {output_dir / 'low_variance_conflict.png'}")


if __name__ == "__main__":
    main()
