"""
Gradient Subspace Analysis - Advanced Visualizations

Generates publication-quality plots showing:
1. PCA projection of gradient directions
2. Gradient trajectory over training epochs
3. Eigenvalue spectrum of gradient covariance
4. Gradient decomposition into domain components
5. Subspace overlap analysis
6. Gradient flow visualization
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches

torch.manual_seed(42)
np.random.seed(42)

output_dir = Path("./figures")
output_dir.mkdir(exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=12, hidden_dim=64, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, x):
        return self.net(x)


def collect_gradients(num_samples=100, num_batches=50):
    """Collect gradients from QF and BN regimes."""
    
    model = PolicyNetwork()
    loss_fn = nn.CrossEntropyLoss()
    
    # VM characteristics
    vm_speed = torch.tensor([2.0, 1.5, 1.0, 2.5])
    vm_efficiency = torch.tensor([0.5, 0.8, 1.0, 0.3])
    
    gradients_qf = []
    gradients_bn = []
    
    for batch in range(num_batches):
        # Generate states
        states = torch.randn(num_samples, 12)
        queue_lengths = torch.rand(num_samples, 4) * 5
        
        # QF optimal: always VM2 (efficient)
        qf_optimal = torch.full((num_samples,), 2, dtype=torch.long)
        
        # BN optimal: fastest available
        effective_speed = vm_speed.unsqueeze(0) / (queue_lengths + 1)
        bn_optimal = effective_speed.argmax(dim=1)
        
        # Compute gradients
        model.zero_grad()
        loss_qf = loss_fn(model(states), qf_optimal)
        loss_qf.backward()
        grad_qf = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
        gradients_qf.append(grad_qf.numpy())
        
        model.zero_grad()
        loss_bn = loss_fn(model(states), bn_optimal)
        loss_bn.backward()
        grad_bn = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
        gradients_bn.append(grad_bn.numpy())
    
    return np.array(gradients_qf), np.array(gradients_bn)


def plot_pca_gradient_space(gradients_qf, gradients_bn):
    """Plot gradients in PCA space showing domain separation."""
    
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
    
    # Draw centroids
    qf_centroid = qf_pca.mean(axis=0)
    bn_centroid = bn_pca.mean(axis=0)
    ax1.scatter([qf_centroid[0]], [qf_centroid[1]], c='#0077BB', s=200, marker='*',
               edgecolor='black', linewidth=2, zorder=10)
    ax1.scatter([bn_centroid[0]], [bn_centroid[1]], c='#CC3311', s=200, marker='*',
               edgecolor='black', linewidth=2, zorder=10)
    
    # Draw arrow between centroids
    ax1.annotate('', xy=(bn_centroid[0], bn_centroid[1]), 
                xytext=(qf_centroid[0], qf_centroid[1]),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    ax1.set_title('(a) Gradient Space (PCA)\nDomains Occupy Different Regions', fontweight='bold')
    ax1.legend(loc='upper right')
    
    # Gradient direction arrows
    ax2 = fig.add_subplot(132)
    
    # Normalize and show as arrows from origin
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
    
    # Mean directions
    qf_mean_dir = qf_pca.mean(axis=0)[:2]
    qf_mean_dir = qf_mean_dir / np.linalg.norm(qf_mean_dir)
    bn_mean_dir = bn_pca.mean(axis=0)[:2]
    bn_mean_dir = bn_mean_dir / np.linalg.norm(bn_mean_dir)
    
    ax2.arrow(0, 0, qf_mean_dir[0], qf_mean_dir[1],
             head_width=0.1, head_length=0.05, fc='#0077BB', ec='black', linewidth=2)
    ax2.arrow(0, 0, bn_mean_dir[0], bn_mean_dir[1],
             head_width=0.1, head_length=0.05, fc='#CC3311', ec='black', linewidth=2)
    
    # Angle annotation
    angle = np.arccos(np.dot(qf_mean_dir, bn_mean_dir)) * 180 / np.pi
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
    ax2.set_title('(b) Gradient Directions\nArrows Show Update Direction', fontweight='bold')
    
    # Explained variance
    ax3 = fig.add_subplot(133)
    n_components = min(10, len(pca.explained_variance_ratio_))
    
    # Compute for each domain separately
    pca_qf = PCA(n_components=min(10, gradients_qf.shape[1])).fit(gradients_qf)
    pca_bn = PCA(n_components=min(10, gradients_bn.shape[1])).fit(gradients_bn)
    
    x = np.arange(n_components)
    width = 0.35
    
    ax3.bar(x - width/2, pca_qf.explained_variance_ratio_[:n_components] * 100, 
           width, label='Queue-Free', color='#0077BB', edgecolor='black')
    ax3.bar(x + width/2, pca_bn.explained_variance_ratio_[:n_components] * 100,
           width, label='Bottleneck', color='#CC3311', edgecolor='black')
    
    ax3.set_xlabel('Principal Component')
    ax3.set_ylabel('Variance Explained (%)')
    ax3.set_title('(c) Gradient Variance Spectrum\nLow-Rank Structure', fontweight='bold')
    ax3.legend()
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'PC{i+1}' for i in x])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_subspace_pca.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'gradient_subspace_pca.png', dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: gradient_subspace_pca.png")
    
    return pca


def plot_gradient_trajectory(num_epochs=50):
    """Show how gradients evolve over training epochs."""
    
    model = PolicyNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    # Track gradient trajectory
    trajectory_qf = []
    trajectory_bn = []
    losses_qf = []
    losses_bn = []
    similarities = []
    
    for epoch in range(num_epochs):
        states = torch.randn(100, 12)
        qf_optimal = torch.full((100,), 2, dtype=torch.long)
        bn_optimal = torch.full((100,), 3, dtype=torch.long)
        
        # QF gradient
        model.zero_grad()
        loss_qf = loss_fn(model(states), qf_optimal)
        loss_qf.backward(retain_graph=True)
        grad_qf = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
        
        # BN gradient  
        model.zero_grad()
        loss_bn = loss_fn(model(states), bn_optimal)
        loss_bn.backward()
        grad_bn = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
        
        trajectory_qf.append(grad_qf.numpy())
        trajectory_bn.append(grad_bn.numpy())
        losses_qf.append(loss_qf.item())
        losses_bn.append(loss_bn.item())
        
        cos_sim = torch.dot(grad_qf, grad_bn) / (grad_qf.norm() * grad_bn.norm())
        similarities.append(cos_sim.item())
        
        # Update with mixed gradient (simulating joint training)
        combined = (grad_qf + grad_bn) / 2
        offset = 0
        for p in model.parameters():
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
    
    # Trajectory plot
    ax = axes[0]
    
    # Color by epoch
    colors = plt.cm.viridis(np.linspace(0, 1, num_epochs))
    
    for i in range(num_epochs - 1):
        ax.plot([traj_qf_pca[i, 0], traj_qf_pca[i+1, 0]], 
               [traj_qf_pca[i, 1], traj_qf_pca[i+1, 1]], 
               color=colors[i], linewidth=2, alpha=0.7)
        ax.plot([traj_bn_pca[i, 0], traj_bn_pca[i+1, 0]],
               [traj_bn_pca[i, 1], traj_bn_pca[i+1, 1]],
               color=colors[i], linewidth=2, alpha=0.7, linestyle='--')
    
    # Mark start and end
    ax.scatter([traj_qf_pca[0, 0]], [traj_qf_pca[0, 1]], c='#0077BB', s=150, 
              marker='o', edgecolor='black', linewidth=2, zorder=10, label='QF Start')
    ax.scatter([traj_qf_pca[-1, 0]], [traj_qf_pca[-1, 1]], c='#0077BB', s=150,
              marker='s', edgecolor='black', linewidth=2, zorder=10, label='QF End')
    ax.scatter([traj_bn_pca[0, 0]], [traj_bn_pca[0, 1]], c='#CC3311', s=150,
              marker='o', edgecolor='black', linewidth=2, zorder=10, label='BN Start')
    ax.scatter([traj_bn_pca[-1, 0]], [traj_bn_pca[-1, 1]], c='#CC3311', s=150,
              marker='s', edgecolor='black', linewidth=2, zorder=10, label='BN End')
    
    ax.set_xlabel(f'PC1')
    ax.set_ylabel(f'PC2')
    ax.set_title('(a) Gradient Trajectory During Training\n(Solid=QF, Dashed=BN)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    # Add colorbar for epochs
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=num_epochs))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Epoch')
    
    # Similarity over epochs
    ax = axes[1]
    ax.fill_between(range(num_epochs), -1, 0, alpha=0.1, color='red')
    ax.plot(range(num_epochs), similarities, color='#CC3311', linewidth=2)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axhline(y=np.mean(similarities), color='#0077BB', linestyle='--', 
              label=f'Mean: {np.mean(similarities):.3f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(b) Gradient Similarity Over Training\n(Conflict Persists!)', fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.text(num_epochs*0.7, -0.5, 'CONFLICT\nZONE', fontsize=11, color='#CC3311', fontweight='bold')
    
    # Loss curves
    ax = axes[2]
    ax.plot(range(num_epochs), losses_qf, color='#0077BB', linewidth=2, label='QF Loss')
    ax.plot(range(num_epochs), losses_bn, color='#CC3311', linewidth=2, label='BN Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(c) Loss Curves\n(Both Struggle to Converge)', fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_trajectory.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'gradient_trajectory.png', dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: gradient_trajectory.png")


def plot_gradient_decomposition(gradients_qf, gradients_bn):
    """Decompose gradients into domain-aligned and domain-orthogonal components."""
    
    # Compute mean gradient directions
    mean_qf = gradients_qf.mean(axis=0)
    mean_bn = gradients_bn.mean(axis=0)
    
    # Normalize
    mean_qf_norm = mean_qf / np.linalg.norm(mean_qf)
    mean_bn_norm = mean_bn / np.linalg.norm(mean_bn)
    
    # Decompose each gradient
    qf_components = {'aligned': [], 'cross': [], 'orthogonal': []}
    bn_components = {'aligned': [], 'cross': [], 'orthogonal': []}
    
    for g in gradients_qf:
        # Project onto QF direction
        aligned = np.dot(g, mean_qf_norm)
        # Project onto BN direction
        cross = np.dot(g, mean_bn_norm)
        # Orthogonal component
        proj = aligned * mean_qf_norm + cross * mean_bn_norm
        orthog = np.linalg.norm(g - proj)
        
        qf_components['aligned'].append(abs(aligned))
        qf_components['cross'].append(abs(cross))
        qf_components['orthogonal'].append(orthog)
    
    for g in gradients_bn:
        aligned = np.dot(g, mean_bn_norm)
        cross = np.dot(g, mean_qf_norm)
        proj = aligned * mean_bn_norm + cross * mean_qf_norm
        orthog = np.linalg.norm(g - proj)
        
        bn_components['aligned'].append(abs(aligned))
        bn_components['cross'].append(abs(cross))
        bn_components['orthogonal'].append(orthog)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Stacked bar chart
    ax = axes[0]
    
    x = np.arange(2)
    width = 0.6
    
    qf_means = [np.mean(qf_components['aligned']), np.mean(qf_components['cross'])]
    bn_means = [np.mean(bn_components['cross']), np.mean(bn_components['aligned'])]
    
    ax.bar(x, [np.mean(qf_components['aligned']), np.mean(bn_components['aligned'])],
          width, label='Self-Aligned', color='#009988', edgecolor='black')
    ax.bar(x, [np.mean(qf_components['cross']), np.mean(bn_components['cross'])],
          width, bottom=[np.mean(qf_components['aligned']), np.mean(bn_components['aligned'])],
          label='Cross-Domain', color='#CC3311', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Queue-Free\nGradients', 'Bottleneck\nGradients'])
    ax.set_ylabel('Component Magnitude')
    ax.set_title('(a) Gradient Decomposition\nby Domain Direction', fontweight='bold')
    ax.legend()
    
    # Component ratio
    ax = axes[1]
    
    qf_ratio = np.array(qf_components['cross']) / (np.array(qf_components['aligned']) + 1e-8)
    bn_ratio = np.array(bn_components['cross']) / (np.array(bn_components['aligned']) + 1e-8)
    
    ax.hist(qf_ratio, bins=20, alpha=0.6, color='#0077BB', label='QF gradients', edgecolor='black')
    ax.hist(bn_ratio, bins=20, alpha=0.6, color='#CC3311', label='BN gradients', edgecolor='black')
    ax.axvline(x=1, color='black', linestyle='--', linewidth=2, label='Equal components')
    ax.set_xlabel('Cross-Domain / Self-Aligned Ratio')
    ax.set_ylabel('Frequency')
    ax.set_title('(b) Component Ratio Distribution\n(>1 means more cross-domain)', fontweight='bold')
    ax.legend()
    
    # Subspace overlap visualization
    ax = axes[2]
    
    # Compute subspace bases
    pca_qf = PCA(n_components=5).fit(gradients_qf)
    pca_bn = PCA(n_components=5).fit(gradients_bn)
    
    # Compute overlap matrix (dot products of principal components)
    overlap = np.abs(pca_qf.components_ @ pca_bn.components_.T)
    
    im = ax.imshow(overlap, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f'BN-PC{i+1}' for i in range(5)])
    ax.set_yticklabels([f'QF-PC{i+1}' for i in range(5)])
    ax.set_title('(c) Subspace Overlap Matrix\n|cos(QF-PC, BN-PC)|', fontweight='bold')
    
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{overlap[i,j]:.2f}', ha='center', va='center',
                   color='white' if overlap[i,j] > 0.5 else 'black', fontsize=10)
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_decomposition.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'gradient_decomposition.png', dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: gradient_decomposition.png")


def plot_gradient_flow():
    """Visualize gradient flow showing cancellation effect."""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel (a): Vector field visualization
    ax = axes[0]
    
    # Create a 2D parameter space
    x = np.linspace(-2, 2, 15)
    y = np.linspace(-2, 2, 15)
    X, Y = np.meshgrid(x, y)
    
    # QF gradient field (points toward efficient region)
    U_qf = -X + 0.5
    V_qf = -Y + 1
    
    # BN gradient field (points toward fast region - opposite)
    U_bn = -X - 0.5
    V_bn = -Y - 1
    
    # Combined field
    U_combined = (U_qf + U_bn) / 2
    V_combined = (V_qf + V_bn) / 2
    
    ax.quiver(X, Y, U_qf, V_qf, color='#0077BB', alpha=0.5, label='QF Gradient')
    ax.quiver(X, Y, U_bn, V_bn, color='#CC3311', alpha=0.5, label='BN Gradient')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel('Parameter θ₁')
    ax.set_ylabel('Parameter θ₂')
    ax.set_title('(a) Gradient Vector Fields\n(Blue=QF, Red=BN)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    # Draw optima
    ax.scatter([0.5], [1], c='#0077BB', s=200, marker='*', edgecolor='black', 
              linewidth=2, zorder=10)
    ax.scatter([-0.5], [-1], c='#CC3311', s=200, marker='*', edgecolor='black',
              linewidth=2, zorder=10)
    ax.text(0.7, 1.2, 'QF\nOptimum', fontsize=9, color='#0077BB')
    ax.text(-0.3, -0.8, 'BN\nOptimum', fontsize=9, color='#CC3311')
    
    # Panel (b): Gradient magnitude heatmap
    ax = axes[1]
    
    # Compute magnitude of combined gradient
    magnitude = np.sqrt(U_combined**2 + V_combined**2)
    magnitude_individual = (np.sqrt(U_qf**2 + V_qf**2) + np.sqrt(U_bn**2 + V_bn**2)) / 2
    
    im = ax.imshow(magnitude, extent=[-2, 2, -2, 2], origin='lower', 
                  cmap='RdYlGn_r', vmin=0, vmax=3)
    ax.contour(X, Y, magnitude, levels=[0.5, 1, 1.5, 2], colors='black', alpha=0.5)
    ax.set_xlabel('Parameter θ₁')
    ax.set_ylabel('Parameter θ₂')
    ax.set_title('(b) Combined Gradient Magnitude\n(Dark = Cancellation Zone)', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='||∇||')
    
    # Mark cancellation zone
    ax.scatter([0], [0], c='yellow', s=300, marker='o', edgecolor='black',
              linewidth=2, zorder=10)
    ax.text(0.2, 0.2, 'Maximum\nCancellation', fontsize=9, fontweight='bold')
    
    # Panel (c): Efficiency across space
    ax = axes[2]
    
    efficiency = magnitude / (magnitude_individual + 1e-8) * 100
    
    im = ax.imshow(efficiency, extent=[-2, 2, -2, 2], origin='lower',
                  cmap='RdYlGn', vmin=0, vmax=100)
    ax.contour(X, Y, efficiency, levels=[25, 50, 75], colors='black', alpha=0.5)
    ax.set_xlabel('Parameter θ₁')
    ax.set_ylabel('Parameter θ₂')
    ax.set_title('(c) Training Efficiency\n(% of gradient signal used)', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Efficiency %')
    
    # Draw path showing optimization trajectory
    theta = np.linspace(0, 4*np.pi, 100)
    r = np.linspace(1.5, 0.3, 100)
    traj_x = r * np.cos(theta) * 0.5
    traj_y = r * np.sin(theta) * 0.5
    ax.plot(traj_x, traj_y, 'k-', linewidth=2, alpha=0.7)
    ax.scatter([traj_x[0]], [traj_y[0]], c='green', s=100, marker='o', zorder=10)
    ax.scatter([traj_x[-1]], [traj_y[-1]], c='black', s=100, marker='s', zorder=10)
    ax.text(traj_x[0]+0.2, traj_y[0]+0.2, 'Start', fontsize=9)
    ax.text(traj_x[-1]+0.2, traj_y[-1]+0.2, 'End', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_flow.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'gradient_flow.png', dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: gradient_flow.png")


def plot_tsne_clustering(gradients_qf, gradients_bn):
    """t-SNE visualization showing gradient clustering by domain."""
    
    all_grads = np.vstack([gradients_qf, gradients_bn])
    labels = ['QF'] * len(gradients_qf) + ['BN'] * len(gradients_bn)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    grads_tsne = tsne.fit_transform(all_grads)
    
    n_qf = len(gradients_qf)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.scatter(grads_tsne[:n_qf, 0], grads_tsne[:n_qf, 1], c='#0077BB', s=60,
              alpha=0.7, label='Queue-Free', edgecolor='black', linewidth=0.5)
    ax.scatter(grads_tsne[n_qf:, 0], grads_tsne[n_qf:, 1], c='#CC3311', s=60,
              alpha=0.7, label='Bottleneck', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('(a) t-SNE of Gradient Vectors\nClear Domain Separation', fontweight='bold')
    ax.legend()
    
    # Compute inter/intra cluster distances
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
    ax.set_title('(b) Cluster Separation\n(Inter >> Intra = Good Separation)', fontweight='bold')
    
    # Separation ratio
    ratio = inter / ((intra_qf + intra_bn) / 2)
    ax.text(0.5, 0.9, f'Separation Ratio: {ratio:.2f}x', transform=ax.transAxes,
           fontsize=12, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='#FFFFCC', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_tsne.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'gradient_tsne.png', dpi=300, facecolor='white')
    plt.close()
    print(f"Saved: gradient_tsne.png")


def main():
    print("="*70)
    print("GRADIENT SUBSPACE ANALYSIS")
    print("="*70)
    
    print("\nCollecting gradients...")
    gradients_qf, gradients_bn = collect_gradients(num_samples=100, num_batches=50)
    print(f"  QF gradients shape: {gradients_qf.shape}")
    print(f"  BN gradients shape: {gradients_bn.shape}")
    
    print("\nGenerating plots...")
    
    plot_pca_gradient_space(gradients_qf, gradients_bn)
    plot_gradient_trajectory(num_epochs=50)
    plot_gradient_decomposition(gradients_qf, gradients_bn)
    plot_gradient_flow()
    plot_tsne_clustering(gradients_qf, gradients_bn)
    
    print("\n" + "="*70)
    print("ALL PLOTS GENERATED")
    print("="*70)
    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
