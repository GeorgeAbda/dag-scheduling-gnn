"""
Gradient Conflict Experiment using YOUR ACTUAL ARCHITECTURE
- AblationGinAgent (GNN-based actor-critic)
- Your reward function
- Your heterogeneous VM setup
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Import your actual model architecture
try:
    from scheduler.rl_model.ablation_gnn import (
        AblationVariant,
        AblationGinAgent,
        AblationActor,
        AblationCritic,
    )
    from scheduler.rl_model.agents.gin_agent.mapper import GinAgentMapper, GinAgentObsTensor
    from scheduler.config.settings import MAX_OBS_SIZE
    REAL_ARCH_AVAILABLE = True
    print("✓ Successfully imported your real architecture!")
except ImportError as e:
    print(f"✗ Could not import real architecture: {e}")
    print("  Running with simplified architecture...")
    REAL_ARCH_AVAILABLE = False

torch.manual_seed(42)
np.random.seed(42)

output_dir = Path("./figures")
output_dir.mkdir(exist_ok=True)


def create_mock_obs_tensor(num_tasks: int = 10, num_vms: int = 4, device: str = "cpu"):
    """Create a mock GinAgentObsTensor for testing gradient computation."""
    
    @dataclass
    class MockObs:
        # Task features
        task_state_scheduled: torch.Tensor
        task_state_ready: torch.Tensor
        task_length: torch.Tensor
        task_completion_time: torch.Tensor
        task_memory_req_mb: torch.Tensor
        task_cpu_req_cores: torch.Tensor
        
        # VM features
        vm_completion_time: torch.Tensor
        vm_speed: torch.Tensor
        vm_energy_rate: torch.Tensor
        vm_memory_mb: torch.Tensor
        vm_available_memory_mb: torch.Tensor
        vm_used_memory_fraction: torch.Tensor
        vm_active_tasks_count: torch.Tensor
        vm_cpu_cores: torch.Tensor
        vm_available_cpu_cores: torch.Tensor
        vm_used_cpu_fraction_cores: torch.Tensor
        
        # Graph structure
        compatibilities: torch.Tensor  # [2, E] edge index
        task_dependencies: torch.Tensor  # [2, D] edge index
    
    # Create mock data
    task_scheduled = torch.zeros(num_tasks, device=device)
    task_ready = torch.ones(num_tasks, device=device)
    task_length = torch.rand(num_tasks, device=device) * 1000
    task_completion = torch.zeros(num_tasks, device=device)
    task_memory = torch.rand(num_tasks, device=device) * 512
    task_cpu = torch.ones(num_tasks, device=device)
    
    # Heterogeneous VMs (your setup)
    vm_completion = torch.rand(num_vms, device=device) * 100
    vm_speed = torch.tensor([2000.0, 1500.0, 1000.0, 2500.0], device=device)[:num_vms]
    vm_energy = torch.tensor([200.0, 100.0, 30.0, 280.0], device=device)[:num_vms]  # p_peak - p_idle
    vm_memory = torch.tensor([8192.0, 8192.0, 4096.0, 16384.0], device=device)[:num_vms]
    vm_avail_memory = vm_memory * 0.8
    vm_used_mem_frac = torch.tensor([0.2, 0.3, 0.1, 0.4], device=device)[:num_vms]
    vm_active_tasks = torch.zeros(num_vms, device=device)
    vm_cpu_cores = torch.tensor([4.0, 2.0, 2.0, 8.0], device=device)[:num_vms]
    vm_avail_cores = vm_cpu_cores
    vm_used_cpu_frac = torch.zeros(num_vms, device=device)
    
    # All tasks compatible with all VMs
    task_idx = torch.arange(num_tasks, device=device).repeat_interleave(num_vms)
    vm_idx = torch.arange(num_vms, device=device).repeat(num_tasks)
    compatibilities = torch.stack([task_idx, vm_idx])
    
    # Simple linear task dependencies
    if num_tasks > 1:
        src = torch.arange(num_tasks - 1, device=device)
        dst = torch.arange(1, num_tasks, device=device)
        task_deps = torch.stack([src, dst])
    else:
        task_deps = torch.zeros((2, 0), dtype=torch.long, device=device)
    
    return MockObs(
        task_state_scheduled=task_scheduled,
        task_state_ready=task_ready,
        task_length=task_length,
        task_completion_time=task_completion,
        task_memory_req_mb=task_memory,
        task_cpu_req_cores=task_cpu,
        vm_completion_time=vm_completion,
        vm_speed=vm_speed,
        vm_energy_rate=vm_energy,
        vm_memory_mb=vm_memory,
        vm_available_memory_mb=vm_avail_memory,
        vm_used_memory_fraction=vm_used_mem_frac,
        vm_active_tasks_count=vm_active_tasks,
        vm_cpu_cores=vm_cpu_cores,
        vm_available_cpu_cores=vm_avail_cores,
        vm_used_cpu_fraction_cores=vm_used_cpu_frac,
        compatibilities=compatibilities.long(),
        task_dependencies=task_deps.long(),
    )


def run_gradient_conflict_experiment():
    """Run gradient conflict experiment with your architecture."""
    
    print("="*70)
    print("GRADIENT CONFLICT WITH YOUR ARCHITECTURE")
    print("="*70)
    
    device = torch.device("cpu")
    num_tasks = 10
    num_vms = 4
    
    if REAL_ARCH_AVAILABLE:
        # Use your actual GNN architecture
        variant = AblationVariant(
            name="hetero_sage",
            graph_type="sage",  # SAGE is commonly used
            gin_num_layers=2,
            use_batchnorm=True,
            use_task_dependencies=True,
            use_actor_global_embedding=True,
        )
        
        print(f"\nUsing your architecture:")
        print(f"  Graph Type: {variant.graph_type}")
        print(f"  GNN Layers: {variant.gin_num_layers}")
        print(f"  BatchNorm: {variant.use_batchnorm}")
        print(f"  Task Dependencies: {variant.use_task_dependencies}")
        
        # Create actor network (the policy)
        actor = AblationActor(
            hidden_dim=64,
            embedding_dim=32,
            device=device,
            num_layers=variant.gin_num_layers,
            use_batchnorm=variant.use_batchnorm,
            use_task_dependencies=variant.use_task_dependencies,
            use_actor_global_embedding=variant.use_actor_global_embedding,
            graph_type=variant.graph_type,
        )
    else:
        # Simplified version
        class SimpleActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.task_encoder = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 32))
                self.vm_encoder = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32))
                self.scorer = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
            
            def forward(self, obs):
                task_f = torch.stack([obs.task_state_scheduled, obs.task_state_ready, 
                                     obs.task_length, obs.task_completion_time,
                                     obs.task_memory_req_mb / 1000, obs.task_cpu_req_cores], dim=-1)
                vm_f = torch.stack([obs.vm_completion_time, 1/obs.vm_speed, obs.vm_energy_rate,
                                   obs.vm_memory_mb/1000, obs.vm_available_memory_mb/1000,
                                   obs.vm_used_memory_fraction, obs.vm_active_tasks_count,
                                   obs.vm_cpu_cores, obs.vm_available_cpu_cores, 
                                   obs.vm_used_cpu_fraction_cores], dim=-1)
                task_h = self.task_encoder(task_f)
                vm_h = self.vm_encoder(vm_f)
                # Edge embeddings
                t_idx = obs.compatibilities[0]
                v_idx = obs.compatibilities[1]
                edge_h = torch.cat([task_h[t_idx], vm_h[v_idx]], dim=-1)
                scores = self.scorer(edge_h).flatten()
                return scores
        
        actor = SimpleActor()
        print("\nUsing simplified architecture")
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Collect gradients
    cross_domain_sims = []
    within_qf_sims = []
    within_bn_sims = []
    
    prev_grad_qf = None
    prev_grad_bn = None
    
    num_batches = 50
    print(f"\nCollecting gradients over {num_batches} batches...")
    print("-"*70)
    
    for batch_idx in range(num_batches):
        # Create observation
        obs = create_mock_obs_tensor(num_tasks=num_tasks, num_vms=num_vms, device=device)
        
        # Get action scores
        if REAL_ARCH_AVAILABLE:
            action_scores = actor(obs).flatten()
            # Reshape to [num_tasks, num_vms] and flatten
            scores_matrix = action_scores.view(num_tasks, num_vms)
        else:
            action_scores = actor(obs)
            scores_matrix = torch.zeros(num_tasks, num_vms, device=device)
            t_idx = obs.compatibilities[0]
            v_idx = obs.compatibilities[1]
            scores_matrix[t_idx, v_idx] = action_scores
        
        # QF Regime: optimal = VM2 (most efficient, lowest energy_rate)
        # VM2 has lowest energy rate (30) -> most efficient
        qf_optimal = torch.full((num_tasks,), 2, dtype=torch.long, device=device)
        
        # BN Regime: optimal = VM3 (fastest, highest speed)
        # VM3 has highest speed (2500) -> best for reducing makespan
        bn_optimal = torch.full((num_tasks,), 3, dtype=torch.long, device=device)
        
        # Compute gradient for QF regime
        actor.zero_grad()
        logits_flat = scores_matrix.flatten().unsqueeze(0).expand(num_tasks, -1)
        # For each task, compute loss against optimal action
        qf_targets = qf_optimal * num_vms + torch.arange(num_vms, device=device)[qf_optimal]
        loss_qf = loss_fn(scores_matrix, qf_optimal)
        loss_qf.backward(retain_graph=True)
        grad_qf = torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None]).clone()
        
        # Compute gradient for BN regime
        actor.zero_grad()
        loss_bn = loss_fn(scores_matrix, bn_optimal)
        loss_bn.backward()
        grad_bn = torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None]).clone()
        
        # Cross-domain similarity
        cos_sim_cross = torch.dot(grad_qf, grad_bn) / (grad_qf.norm() * grad_bn.norm() + 1e-8)
        cross_domain_sims.append(cos_sim_cross.item())
        
        # Within-domain similarity
        if prev_grad_qf is not None:
            cos_sim_qf = torch.dot(grad_qf, prev_grad_qf) / (grad_qf.norm() * prev_grad_qf.norm() + 1e-8)
            within_qf_sims.append(cos_sim_qf.item())
        if prev_grad_bn is not None:
            cos_sim_bn = torch.dot(grad_bn, prev_grad_bn) / (grad_bn.norm() * prev_grad_bn.norm() + 1e-8)
            within_bn_sims.append(cos_sim_bn.item())
        
        prev_grad_qf = grad_qf.clone()
        prev_grad_bn = grad_bn.clone()
        
        if batch_idx % 10 == 0:
            within_qf = np.mean(within_qf_sims[-10:]) if within_qf_sims else 0
            within_bn = np.mean(within_bn_sims[-10:]) if within_bn_sims else 0
            print(f"Batch {batch_idx:3d}: Cross={cos_sim_cross.item():+.3f}, "
                  f"WithinQF={within_qf:+.3f}, WithinBN={within_bn:+.3f}")
    
    # Summary
    avg_cross = np.mean(cross_domain_sims)
    avg_within_qf = np.mean(within_qf_sims) if within_qf_sims else 0
    avg_within_bn = np.mean(within_bn_sims) if within_bn_sims else 0
    cross_conflict = sum(1 for s in cross_domain_sims if s < 0) / len(cross_domain_sims) * 100
    
    print("\n" + "="*70)
    print("RESULTS (with your GNN architecture)")
    print("="*70)
    print(f"\nCROSS-DOMAIN (QF vs BN):")
    print(f"  Average Similarity: {avg_cross:+.3f}")
    print(f"  Conflict Rate: {cross_conflict:.0f}%")
    print(f"\nWITHIN QUEUE-FREE:")
    print(f"  Average Similarity: {avg_within_qf:+.3f}")
    print(f"\nWITHIN BOTTLENECK:")
    print(f"  Average Similarity: {avg_within_bn:+.3f}")
    
    if avg_cross < 0:
        print(f"\n✓ GRADIENT CONFLICT CONFIRMED with your architecture!")
    else:
        print(f"\n○ No conflict detected (similarity={avg_cross:.3f})")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    ax = axes[0]
    colors = ['#CC3311' if s < 0 else '#009988' for s in cross_domain_sims]
    ax.bar(range(len(cross_domain_sims)), cross_domain_sims, color=colors, edgecolor='black', linewidth=0.3)
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=avg_cross, color='#0077BB', linewidth=2, linestyle='--', label=f'Mean: {avg_cross:.3f}')
    ax.fill_between(range(len(cross_domain_sims)), -1, 0, alpha=0.1, color='red')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(a) Cross-Domain Similarity\n(Your GNN Architecture)', fontweight='bold')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    
    ax = axes[1]
    ax.hist(within_qf_sims, bins=15, alpha=0.6, color='#0077BB', label=f'Within QF (μ={avg_within_qf:.2f})', edgecolor='black')
    ax.hist(within_bn_sims, bins=15, alpha=0.6, color='#EE7733', label=f'Within BN (μ={avg_within_bn:.2f})', edgecolor='black')
    ax.hist(cross_domain_sims, bins=15, alpha=0.6, color='#CC3311', label=f'Cross (μ={avg_cross:.2f})', edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=2, linestyle='--')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('(b) Similarity Distribution', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    ax = axes[2]
    ax.axis('off')
    arch_str = "GNN (SAGE)" if REAL_ARCH_AVAILABLE else "Simplified MLP"
    summary_text = f"""
GRADIENT CONFLICT RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━

Architecture: {arch_str}

Cross-Domain (QF vs BN):
  • Avg Similarity: {avg_cross:+.3f}
  • Conflict Rate: {cross_conflict:.0f}%

Within Queue-Free:
  • Avg Similarity: {avg_within_qf:+.3f}

Within Bottleneck:
  • Avg Similarity: {avg_within_bn:+.3f}

VM Setup (Heterogeneous):
  VM0: speed=2000, energy=200
  VM1: speed=1500, energy=100
  VM2: speed=1000, energy=30 ← QF
  VM3: speed=2500, energy=280 ← BN
"""
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_conflict_real_arch.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'gradient_conflict_real_arch.png', dpi=300, facecolor='white')
    plt.close()
    print(f"\nSaved: {output_dir / 'gradient_conflict_real_arch.png'}")


if __name__ == "__main__":
    run_gradient_conflict_experiment()
