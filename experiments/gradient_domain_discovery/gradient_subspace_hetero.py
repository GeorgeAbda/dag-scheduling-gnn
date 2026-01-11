"""
Gradient Subspace Analysis with YOUR EXACT HETERO Architecture
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dataclasses import dataclass

from scheduler.rl_model.ablation_gnn import AblationVariant, AblationActor

torch.manual_seed(42)
np.random.seed(42)

output_dir = Path(__file__).parent / "figures"
output_dir.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.facecolor': 'white',
})


@dataclass
class MockObs:
    """Mock observation matching GinAgentObsTensor."""
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


def create_obs(num_tasks=10, num_vms=4):
    """Create observation with heterogeneous VMs."""
    return MockObs(
        task_state_scheduled=torch.zeros(num_tasks),
        task_state_ready=torch.ones(num_tasks),
        task_length=torch.rand(num_tasks) * 1000,
        task_completion_time=torch.zeros(num_tasks),
        task_memory_req_mb=torch.rand(num_tasks) * 512,
        task_cpu_req_cores=torch.ones(num_tasks),
        vm_completion_time=torch.rand(num_vms) * 100,
        vm_speed=torch.tensor([2000., 1500., 1000., 2500.]),
        vm_energy_rate=torch.tensor([200., 100., 30., 280.]),
        vm_memory_mb=torch.tensor([8192., 8192., 4096., 16384.]),
        vm_available_memory_mb=torch.tensor([6554., 6554., 3277., 13107.]),
        vm_used_memory_fraction=torch.tensor([0.2, 0.2, 0.2, 0.2]),
        vm_active_tasks_count=torch.zeros(num_vms),
        vm_cpu_cores=torch.tensor([4., 2., 2., 8.]),
        vm_available_cpu_cores=torch.tensor([4., 2., 2., 8.]),
        vm_used_cpu_fraction_cores=torch.zeros(num_vms),
        compatibilities=torch.stack([
            torch.arange(num_tasks).repeat_interleave(num_vms),
            torch.arange(num_vms).repeat(num_tasks)
        ]).long(),
        task_dependencies=torch.stack([
            torch.arange(num_tasks-1),
            torch.arange(1, num_tasks)
        ]).long() if num_tasks > 1 else torch.zeros((2, 0), dtype=torch.long),
    )


def create_hetero_actor():
    """Create YOUR EXACT HETERO architecture."""
    variant = AblationVariant(
        name='hetero',
        graph_type='hetero',
        hetero_base='sage',
        gin_num_layers=2,
        use_batchnorm=True,
        use_task_dependencies=True,
        use_actor_global_embedding=True,
    )
    
    actor = AblationActor(
        hidden_dim=64,
        embedding_dim=32,
        device=torch.device('cpu'),
        num_layers=variant.gin_num_layers,
        use_batchnorm=variant.use_batchnorm,
        use_task_dependencies=variant.use_task_dependencies,
        use_actor_global_embedding=variant.use_actor_global_embedding,
        graph_type=variant.graph_type,
        hetero_base=variant.hetero_base,
    )
    
    # Initialize lazy modules
    obs = create_obs()
    _ = actor(obs)
    
    return actor, variant


def collect_gradients(actor, num_batches=50):
    """Collect gradients from QF and BN regimes using HETERO architecture."""
    loss_fn = nn.CrossEntropyLoss()
    
    gradients_qf = []
    gradients_bn = []
    
    for batch in range(num_batches):
        obs = create_obs(num_tasks=10, num_vms=4)
        action_scores = actor(obs)
        
        # QF optimal: VM2 (efficient)
        qf_optimal = torch.full((10,), 2, dtype=torch.long)
        # BN optimal: VM3 (fast)
        bn_optimal = torch.full((10,), 3, dtype=torch.long)
        
        # QF gradient
        actor.zero_grad()
        loss_fn(action_scores, qf_optimal).backward(retain_graph=True)
        grad_qf = torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None]).clone()
        gradients_qf.append(grad_qf.numpy())
        
        # BN gradient
        actor.zero_grad()
        loss_fn(action_scores, bn_optimal).backward()
        grad_bn = torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None]).clone()
        gradients_bn.append(grad_bn.numpy())
    
    return np.array(gradients_qf), np.array(gradients_bn)


def plot_pca_gradient_space(gradients_qf, gradients_bn, variant):
    """Plot gradients in PCA space."""
    all_grads = np.vstack([gradients_qf, gradients_bn])
    pca = PCA(n_components=3)
    grads_pca = pca.fit_transform(all_grads)
    
    n_qf = len(gradients_qf)
    qf_pca = grads_pca[:n_qf]
    bn_pca = grads_pca[n_qf:]
    
    fig = plt.figure(figsize=(16, 5))
    
    # 2D PCA
    ax1 = fig.add_subplot(131)
    ax1.scatter(qf_pca[:, 0], qf_pca[:, 1], c='#0077BB', s=60, alpha=0.7,
               label='Queue-Free', edgecolor='black', linewidth=0.5)
    ax1.scatter(bn_pca[:, 0], bn_pca[:, 1], c='#CC3311', s=60, alpha=0.7,
               label='Bottleneck', edgecolor='black', linewidth=0.5)
    
    qf_centroid = qf_pca.mean(axis=0)
    bn_centroid = bn_pca.mean(axis=0)
    ax1.scatter([qf_centroid[0]], [qf_centroid[1]], c='#0077BB', s=200, marker='*',
               edgecolor='black', linewidth=2, zorder=10)
    ax1.scatter([bn_centroid[0]], [bn_centroid[1]], c='#CC3311', s=200, marker='*',
               edgecolor='black', linewidth=2, zorder=10)
    
    ax1.annotate('', xy=(bn_centroid[0], bn_centroid[1]),
                xytext=(qf_centroid[0], qf_centroid[1]),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    ax1.set_title(f'(a) Gradient Space (PCA)\nYour {variant.graph_type.upper()} Architecture', fontweight='bold')
    ax1.legend(loc='upper right')
    
    # Gradient directions
    ax2 = fig.add_subplot(132)
    for i in range(min(20, len(qf_pca))):
        norm = np.sqrt(qf_pca[i, 0]**2 + qf_pca[i, 1]**2)
        if norm > 0:
            ax2.arrow(0, 0, qf_pca[i, 0]/norm*0.8, qf_pca[i, 1]/norm*0.8,
                     head_width=0.05, head_length=0.03, fc='#0077BB', ec='#0077BB', alpha=0.5)
    for i in range(min(20, len(bn_pca))):
        norm = np.sqrt(bn_pca[i, 0]**2 + bn_pca[i, 1]**2)
        if norm > 0:
            ax2.arrow(0, 0, bn_pca[i, 0]/norm*0.8, bn_pca[i, 1]/norm*0.8,
                     head_width=0.05, head_length=0.03, fc='#CC3311', ec='#CC3311', alpha=0.5)
    
    qf_mean_dir = qf_pca.mean(axis=0)[:2]
    qf_mean_dir = qf_mean_dir / np.linalg.norm(qf_mean_dir)
    bn_mean_dir = bn_pca.mean(axis=0)[:2]
    bn_mean_dir = bn_mean_dir / np.linalg.norm(bn_mean_dir)
    
    ax2.arrow(0, 0, qf_mean_dir[0], qf_mean_dir[1],
             head_width=0.1, head_length=0.05, fc='#0077BB', ec='black', linewidth=2)
    ax2.arrow(0, 0, bn_mean_dir[0], bn_mean_dir[1],
             head_width=0.1, head_length=0.05, fc='#CC3311', ec='black', linewidth=2)
    
    angle = np.arccos(np.clip(np.dot(qf_mean_dir, bn_mean_dir), -1, 1)) * 180 / np.pi
    ax2.text(0.05, 0.95, f'Angle: {angle:.0f}°', transform=ax2.transAxes,
            fontsize=12, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('PC1 (normalized)')
    ax2.set_ylabel('PC2 (normalized)')
    ax2.set_title('(b) Gradient Directions\n(Your HETERO Architecture)', fontweight='bold')
    
    # Variance spectrum
    ax3 = fig.add_subplot(133)
    pca_qf = PCA(n_components=min(10, gradients_qf.shape[0])).fit(gradients_qf)
    pca_bn = PCA(n_components=min(10, gradients_bn.shape[0])).fit(gradients_bn)
    n_components = min(10, len(pca_qf.explained_variance_ratio_))
    
    x = np.arange(n_components)
    width = 0.35
    ax3.bar(x - width/2, pca_qf.explained_variance_ratio_[:n_components] * 100,
           width, label='Queue-Free', color='#0077BB', edgecolor='black')
    ax3.bar(x + width/2, pca_bn.explained_variance_ratio_[:n_components] * 100,
           width, label='Bottleneck', color='#CC3311', edgecolor='black')
    ax3.set_xlabel('Principal Component')
    ax3.set_ylabel('Variance Explained (%)')
    ax3.set_title('(c) Gradient Variance Spectrum', fontweight='bold')
    ax3.legend()
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'PC{i+1}' for i in x])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_subspace_pca_HETERO.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'gradient_subspace_pca_HETERO.png', dpi=300, facecolor='white')
    plt.close()
    print("Saved: gradient_subspace_pca_HETERO.png")


def plot_gradient_trajectory(variant, num_epochs=50):
    """Show gradient evolution during training with HETERO architecture."""
    
    actor, _ = create_hetero_actor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    trajectory_qf = []
    trajectory_bn = []
    similarities = []
    losses_qf = []
    losses_bn = []
    
    for epoch in range(num_epochs):
        obs = create_obs(num_tasks=10, num_vms=4)
        action_scores = actor(obs)
        
        qf_optimal = torch.full((10,), 2, dtype=torch.long)
        bn_optimal = torch.full((10,), 3, dtype=torch.long)
        
        # QF gradient
        actor.zero_grad()
        loss_qf = loss_fn(action_scores, qf_optimal)
        loss_qf.backward(retain_graph=True)
        grad_qf = torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None]).clone()
        
        # BN gradient
        actor.zero_grad()
        loss_bn = loss_fn(action_scores, bn_optimal)
        loss_bn.backward()
        grad_bn = torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None]).clone()
        
        trajectory_qf.append(grad_qf.numpy())
        trajectory_bn.append(grad_bn.numpy())
        losses_qf.append(loss_qf.item())
        losses_bn.append(loss_bn.item())
        
        cos_sim = torch.dot(grad_qf, grad_bn) / (grad_qf.norm() * grad_bn.norm() + 1e-8)
        similarities.append(cos_sim.item())
        
        # Update with mixed gradient
        combined = (grad_qf + grad_bn) / 2
        offset = 0
        for p in actor.parameters():
            if p.grad is not None:
                numel = p.numel()
                p.grad = combined[offset:offset+numel].view_as(p)
                offset += numel
        optimizer.step()
    
    # Project trajectories
    all_grads = np.vstack([np.array(trajectory_qf), np.array(trajectory_bn)])
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_grads)
    
    traj_qf_pca = all_pca[:num_epochs]
    traj_bn_pca = all_pca[num_epochs:]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Trajectory
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, num_epochs))
    for i in range(num_epochs - 1):
        ax.plot([traj_qf_pca[i, 0], traj_qf_pca[i+1, 0]],
               [traj_qf_pca[i, 1], traj_qf_pca[i+1, 1]],
               color=colors[i], linewidth=2, alpha=0.7)
        ax.plot([traj_bn_pca[i, 0], traj_bn_pca[i+1, 0]],
               [traj_bn_pca[i, 1], traj_bn_pca[i+1, 1]],
               color=colors[i], linewidth=2, alpha=0.7, linestyle='--')
    
    ax.scatter([traj_qf_pca[0, 0]], [traj_qf_pca[0, 1]], c='#0077BB', s=150,
              marker='o', edgecolor='black', linewidth=2, zorder=10, label='QF Start')
    ax.scatter([traj_qf_pca[-1, 0]], [traj_qf_pca[-1, 1]], c='#0077BB', s=150,
              marker='s', edgecolor='black', linewidth=2, zorder=10, label='QF End')
    ax.scatter([traj_bn_pca[0, 0]], [traj_bn_pca[0, 1]], c='#CC3311', s=150,
              marker='o', edgecolor='black', linewidth=2, zorder=10, label='BN Start')
    ax.scatter([traj_bn_pca[-1, 0]], [traj_bn_pca[-1, 1]], c='#CC3311', s=150,
              marker='s', edgecolor='black', linewidth=2, zorder=10, label='BN End')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'(a) Gradient Trajectory (HETERO)\n(Solid=QF, Dashed=BN)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=num_epochs))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Epoch')
    
    # Similarity
    ax = axes[1]
    ax.fill_between(range(num_epochs), -1, 0, alpha=0.1, color='red')
    ax.plot(range(num_epochs), similarities, color='#CC3311', linewidth=2)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axhline(y=np.mean(similarities), color='#0077BB', linestyle='--',
              label=f'Mean: {np.mean(similarities):.3f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(b) Gradient Similarity Over Training\n(HETERO Architecture)', fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.text(num_epochs*0.7, -0.5, 'CONFLICT\nZONE', fontsize=11, color='#CC3311', fontweight='bold')
    
    # Loss curves
    ax = axes[2]
    ax.plot(range(num_epochs), losses_qf, color='#0077BB', linewidth=2, label='QF Loss')
    ax.plot(range(num_epochs), losses_bn, color='#CC3311', linewidth=2, label='BN Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(c) Loss Curves (HETERO)', fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_trajectory_HETERO.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'gradient_trajectory_HETERO.png', dpi=300, facecolor='white')
    plt.close()
    print("Saved: gradient_trajectory_HETERO.png")


def plot_tsne_clustering(gradients_qf, gradients_bn, variant):
    """t-SNE visualization with HETERO gradients."""
    
    all_grads = np.vstack([gradients_qf, gradients_bn])
    n_qf = len(gradients_qf)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, len(all_grads)-1))
    grads_tsne = tsne.fit_transform(all_grads)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.scatter(grads_tsne[:n_qf, 0], grads_tsne[:n_qf, 1], c='#0077BB', s=60,
              alpha=0.7, label='Queue-Free', edgecolor='black', linewidth=0.5)
    ax.scatter(grads_tsne[n_qf:, 0], grads_tsne[n_qf:, 1], c='#CC3311', s=60,
              alpha=0.7, label='Bottleneck', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f'(a) t-SNE of Gradient Vectors\n(Your {variant.graph_type.upper()} Architecture)', fontweight='bold')
    ax.legend()
    
    # Cluster separation
    ax = axes[1]
    from scipy.spatial.distance import cdist
    
    qf_tsne = grads_tsne[:n_qf]
    bn_tsne = grads_tsne[n_qf:]
    
    intra_qf = cdist(qf_tsne, qf_tsne).mean()
    intra_bn = cdist(bn_tsne, bn_tsne).mean()
    inter = cdist(qf_tsne, bn_tsne).mean()
    
    bars = ax.bar(['Within\nQueue-Free', 'Within\nBottleneck', 'Between\nDomains'],
                 [intra_qf, intra_bn, inter],
                 color=['#0077BB', '#CC3311', '#EE7733'], edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars, [intra_qf, intra_bn, inter]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Average Distance')
    ax.set_title('(b) Cluster Separation (HETERO)', fontweight='bold')
    
    ratio = inter / ((intra_qf + intra_bn) / 2)
    ax.text(0.5, 0.9, f'Separation Ratio: {ratio:.2f}x', transform=ax.transAxes,
           fontsize=12, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='#FFFFCC', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_tsne_HETERO.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'gradient_tsne_HETERO.png', dpi=300, facecolor='white')
    plt.close()
    print("Saved: gradient_tsne_HETERO.png")


def plot_gradient_decomposition(gradients_qf, gradients_bn, variant):
    """Gradient decomposition analysis with HETERO architecture."""
    
    mean_qf = gradients_qf.mean(axis=0)
    mean_bn = gradients_bn.mean(axis=0)
    
    mean_qf_norm = mean_qf / np.linalg.norm(mean_qf)
    mean_bn_norm = mean_bn / np.linalg.norm(mean_bn)
    
    qf_components = {'aligned': [], 'cross': []}
    bn_components = {'aligned': [], 'cross': []}
    
    for g in gradients_qf:
        qf_components['aligned'].append(abs(np.dot(g, mean_qf_norm)))
        qf_components['cross'].append(abs(np.dot(g, mean_bn_norm)))
    
    for g in gradients_bn:
        bn_components['aligned'].append(abs(np.dot(g, mean_bn_norm)))
        bn_components['cross'].append(abs(np.dot(g, mean_qf_norm)))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Stacked bar
    ax = axes[0]
    x = np.arange(2)
    width = 0.6
    
    ax.bar(x, [np.mean(qf_components['aligned']), np.mean(bn_components['aligned'])],
          width, label='Self-Aligned', color='#009988', edgecolor='black')
    ax.bar(x, [np.mean(qf_components['cross']), np.mean(bn_components['cross'])],
          width, bottom=[np.mean(qf_components['aligned']), np.mean(bn_components['aligned'])],
          label='Cross-Domain', color='#CC3311', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Queue-Free\nGradients', 'Bottleneck\nGradients'])
    ax.set_ylabel('Component Magnitude')
    ax.set_title(f'(a) Gradient Decomposition\n({variant.graph_type.upper()} Architecture)', fontweight='bold')
    ax.legend()
    
    # Component ratio
    ax = axes[1]
    qf_ratio = np.array(qf_components['cross']) / (np.array(qf_components['aligned']) + 1e-8)
    bn_ratio = np.array(bn_components['cross']) / (np.array(bn_components['aligned']) + 1e-8)
    
    ax.hist(qf_ratio, bins=15, alpha=0.6, color='#0077BB', label='QF gradients', edgecolor='black')
    ax.hist(bn_ratio, bins=15, alpha=0.6, color='#CC3311', label='BN gradients', edgecolor='black')
    ax.axvline(x=1, color='black', linestyle='--', linewidth=2, label='Equal components')
    ax.set_xlabel('Cross-Domain / Self-Aligned Ratio')
    ax.set_ylabel('Frequency')
    ax.set_title('(b) Component Ratio Distribution', fontweight='bold')
    ax.legend()
    
    # Subspace overlap
    ax = axes[2]
    n_comp = min(5, gradients_qf.shape[0]-1)
    pca_qf = PCA(n_components=n_comp).fit(gradients_qf)
    pca_bn = PCA(n_components=n_comp).fit(gradients_bn)
    
    overlap = np.abs(pca_qf.components_ @ pca_bn.components_.T)
    
    im = ax.imshow(overlap, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax.set_xticks(range(n_comp))
    ax.set_yticks(range(n_comp))
    ax.set_xticklabels([f'BN-PC{i+1}' for i in range(n_comp)])
    ax.set_yticklabels([f'QF-PC{i+1}' for i in range(n_comp)])
    ax.set_title('(c) Subspace Overlap Matrix\n(HETERO)', fontweight='bold')
    
    for i in range(n_comp):
        for j in range(n_comp):
            ax.text(j, i, f'{overlap[i,j]:.2f}', ha='center', va='center',
                   color='white' if overlap[i,j] > 0.5 else 'black', fontsize=10)
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_decomposition_HETERO.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'gradient_decomposition_HETERO.png', dpi=300, facecolor='white')
    plt.close()
    print("Saved: gradient_decomposition_HETERO.png")


def main():
    print("="*60)
    print("GRADIENT SUBSPACE ANALYSIS")
    print("With YOUR EXACT HETERO Architecture")
    print("="*60)
    
    print("\nCreating HETERO architecture...")
    actor, variant = create_hetero_actor()
    num_params = sum(p.numel() for p in actor.parameters())
    print(f"  Graph Type: {variant.graph_type}")
    print(f"  Hetero Base: {variant.hetero_base}")
    print(f"  Parameters: {num_params:,}")
    
    print("\nCollecting gradients...")
    gradients_qf, gradients_bn = collect_gradients(actor, num_batches=50)
    print(f"  QF gradients shape: {gradients_qf.shape}")
    print(f"  BN gradients shape: {gradients_bn.shape}")
    
    print("\nGenerating plots...")
    plot_pca_gradient_space(gradients_qf, gradients_bn, variant)
    plot_gradient_trajectory(variant, num_epochs=50)
    plot_tsne_clustering(gradients_qf, gradients_bn, variant)
    plot_gradient_decomposition(gradients_qf, gradients_bn, variant)
    
    print("\n" + "="*60)
    print("ALL PLOTS GENERATED WITH YOUR HETERO ARCHITECTURE")
    print("="*60)
    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
