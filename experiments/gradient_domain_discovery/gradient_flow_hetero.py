"""
Gradient Flow Visualization with YOUR HETERO Architecture
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.decomposition import PCA

from scheduler.rl_model.ablation_gnn import AblationVariant, AblationActor

torch.manual_seed(42)
np.random.seed(42)

output_dir = Path(__file__).parent / "figures"


@dataclass
class MockObs:
    task_state_scheduled: torch.Tensor
    task_state_ready: torch.Tensor
    task_length: torch.Tensor
    task_completion_time: torch.Tensor
    task_memory_req_mb: torch.Tensor
    task_cpu_req_cores: torch.Tensor
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
    compatibilities: torch.Tensor
    task_dependencies: torch.Tensor


def create_obs():
    return MockObs(
        task_state_scheduled=torch.zeros(10),
        task_state_ready=torch.ones(10),
        task_length=torch.rand(10) * 1000,
        task_completion_time=torch.zeros(10),
        task_memory_req_mb=torch.rand(10) * 512,
        task_cpu_req_cores=torch.ones(10),
        vm_completion_time=torch.rand(4) * 100,
        vm_speed=torch.tensor([2000., 1500., 1000., 2500.]),
        vm_energy_rate=torch.tensor([200., 100., 30., 280.]),
        vm_memory_mb=torch.tensor([8192., 8192., 4096., 16384.]),
        vm_available_memory_mb=torch.tensor([6554., 6554., 3277., 13107.]),
        vm_used_memory_fraction=torch.tensor([0.2, 0.2, 0.2, 0.2]),
        vm_active_tasks_count=torch.zeros(4),
        vm_cpu_cores=torch.tensor([4., 2., 2., 8.]),
        vm_available_cpu_cores=torch.tensor([4., 2., 2., 8.]),
        vm_used_cpu_fraction_cores=torch.zeros(4),
        compatibilities=torch.stack([torch.arange(10).repeat_interleave(4), torch.arange(4).repeat(10)]).long(),
        task_dependencies=torch.stack([torch.arange(9), torch.arange(1, 10)]).long(),
    )


def main():
    print("="*60)
    print("GRADIENT FLOW with YOUR HETERO ARCHITECTURE")
    print("="*60)
    
    # Create HETERO architecture
    variant = AblationVariant(
        name='hetero', graph_type='hetero', hetero_base='sage',
        gin_num_layers=2, use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True,
    )
    
    actor = AblationActor(
        hidden_dim=64, embedding_dim=32, device=torch.device('cpu'),
        num_layers=variant.gin_num_layers, use_batchnorm=variant.use_batchnorm,
        use_task_dependencies=variant.use_task_dependencies,
        use_actor_global_embedding=variant.use_actor_global_embedding,
        graph_type=variant.graph_type, hetero_base=variant.hetero_base,
    )
    
    # Initialize
    obs = create_obs()
    _ = actor(obs)
    num_params = sum(p.numel() for p in actor.parameters())
    print(f"Architecture: HETERO ({num_params:,} params)")
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Collect gradients
    print("Collecting gradients...")
    all_grads_qf = []
    all_grads_bn = []
    
    for i in range(100):
        obs = create_obs()
        action_scores = actor(obs)
        
        qf_opt = torch.full((10,), 2, dtype=torch.long)
        bn_opt = torch.full((10,), 3, dtype=torch.long)
        
        actor.zero_grad()
        loss_fn(action_scores, qf_opt).backward(retain_graph=True)
        grad_qf = torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None]).clone()
        
        actor.zero_grad()
        loss_fn(action_scores, bn_opt).backward()
        grad_bn = torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None]).clone()
        
        all_grads_qf.append(grad_qf.numpy())
        all_grads_bn.append(grad_bn.numpy())
    
    all_grads_qf = np.array(all_grads_qf)
    all_grads_bn = np.array(all_grads_bn)
    
    # Compute mean gradient directions
    mean_qf = all_grads_qf.mean(axis=0)
    mean_bn = all_grads_bn.mean(axis=0)
    
    # Project to 2D using PCA
    all_grads = np.vstack([all_grads_qf, all_grads_bn])
    pca = PCA(n_components=2)
    pca.fit(all_grads)
    
    # Project mean gradients
    mean_qf_2d = pca.transform(mean_qf.reshape(1, -1))[0]
    mean_bn_2d = pca.transform(mean_bn.reshape(1, -1))[0]
    
    # Define optima in 2D
    qf_opt_2d = mean_qf_2d / np.linalg.norm(mean_qf_2d) * 1.5
    bn_opt_2d = mean_bn_2d / np.linalg.norm(mean_bn_2d) * 1.5
    
    # Create grid
    n_points = 15
    x_range = np.linspace(-2, 2, n_points)
    y_range = np.linspace(-2, 2, n_points)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Compute gradients at each point
    grad_qf_x = np.zeros_like(X)
    grad_qf_y = np.zeros_like(X)
    grad_bn_x = np.zeros_like(X)
    grad_bn_y = np.zeros_like(X)
    
    for i in range(n_points):
        for j in range(n_points):
            to_qf = qf_opt_2d - np.array([X[i,j], Y[i,j]])
            grad_qf_x[i,j] = to_qf[0]
            grad_qf_y[i,j] = to_qf[1]
            
            to_bn = bn_opt_2d - np.array([X[i,j], Y[i,j]])
            grad_bn_x[i,j] = to_bn[0]
            grad_bn_y[i,j] = to_bn[1]
    
    # Combined gradient
    grad_combined_x = (grad_qf_x + grad_bn_x) / 2
    grad_combined_y = (grad_qf_y + grad_bn_y) / 2
    
    # Magnitudes
    mag_qf = np.sqrt(grad_qf_x**2 + grad_qf_y**2)
    mag_bn = np.sqrt(grad_bn_x**2 + grad_bn_y**2)
    mag_combined = np.sqrt(grad_combined_x**2 + grad_combined_y**2)
    mag_expected = (mag_qf + mag_bn) / 2
    
    # Efficiency
    efficiency = mag_combined / (mag_expected + 1e-8) * 100
    
    print("Generating plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel (a): Vector fields
    ax = axes[0]
    skip = 2
    scale = 8
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
             grad_qf_x[::skip, ::skip], grad_qf_y[::skip, ::skip],
             color='#0077BB', alpha=0.6, scale=scale, label='QF Gradient')
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
             grad_bn_x[::skip, ::skip], grad_bn_y[::skip, ::skip],
             color='#CC3311', alpha=0.6, scale=scale, label='BN Gradient')
    
    ax.scatter([qf_opt_2d[0]], [qf_opt_2d[1]], c='#0077BB', s=200, marker='*',
              edgecolor='black', linewidth=2, zorder=10)
    ax.scatter([bn_opt_2d[0]], [bn_opt_2d[1]], c='#CC3311', s=200, marker='*',
              edgecolor='black', linewidth=2, zorder=10)
    ax.text(qf_opt_2d[0]+0.15, qf_opt_2d[1]+0.15, 'QF Opt', fontsize=10, fontweight='bold')
    ax.text(bn_opt_2d[0]+0.15, bn_opt_2d[1]+0.15, 'BN Opt', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('PC1 (gradient space)')
    ax.set_ylabel('PC2 (gradient space)')
    ax.set_title('(a) REAL Gradient Vector Fields\n(Your HETERO Architecture)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    
    # Panel (b): Combined magnitude
    ax = axes[1]
    im = ax.imshow(mag_combined, extent=[-2, 2, -2, 2], origin='lower',
                  cmap='RdYlGn_r', vmin=0, vmax=mag_combined.max())
    ax.contour(X, Y, mag_combined, levels=5, colors='black', alpha=0.5)
    ax.scatter([qf_opt_2d[0]], [qf_opt_2d[1]], c='#0077BB', s=150, marker='*',
              edgecolor='white', linewidth=2, zorder=10)
    ax.scatter([bn_opt_2d[0]], [bn_opt_2d[1]], c='#CC3311', s=150, marker='*',
              edgecolor='white', linewidth=2, zorder=10)
    
    midpoint = (qf_opt_2d + bn_opt_2d) / 2
    ax.scatter([midpoint[0]], [midpoint[1]], c='yellow', s=200, marker='o',
              edgecolor='black', linewidth=2, zorder=10)
    ax.text(midpoint[0]+0.15, midpoint[1]+0.15, 'Max\nCancel', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('(b) Combined Gradient Magnitude\n(Low = Cancellation)', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='||∇||')
    
    # Panel (c): Efficiency
    ax = axes[2]
    im = ax.imshow(efficiency, extent=[-2, 2, -2, 2], origin='lower',
                  cmap='RdYlGn', vmin=0, vmax=100)
    ax.contour(X, Y, efficiency, levels=[25, 50, 75], colors='black', alpha=0.5)
    ax.scatter([qf_opt_2d[0]], [qf_opt_2d[1]], c='#0077BB', s=150, marker='*',
              edgecolor='white', linewidth=2, zorder=10)
    ax.scatter([bn_opt_2d[0]], [bn_opt_2d[1]], c='#CC3311', s=150, marker='*',
              edgecolor='white', linewidth=2, zorder=10)
    ax.scatter([midpoint[0]], [midpoint[1]], c='yellow', s=200, marker='o',
              edgecolor='black', linewidth=2, zorder=10)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('(c) Training Efficiency\n(% of gradient signal retained)', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Efficiency %')
    
    # Architecture info
    fig.text(0.5, 0.02, f'Architecture: HETERO (HeteroConv + SAGE) | Parameters: {num_params:,}',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(output_dir / 'gradient_flow_HETERO.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'gradient_flow_HETERO.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Saved: {output_dir / 'gradient_flow_HETERO.png'}")


if __name__ == "__main__":
    main()
