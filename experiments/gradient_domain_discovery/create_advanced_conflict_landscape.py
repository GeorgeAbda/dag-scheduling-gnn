"""
Create advanced conflict landscape visualizations
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# Setup
output_dir = Path('experiments/gradient_domain_discovery/publication_figures')
output_dir.mkdir(exist_ok=True, parents=True)

print("Loading WGDD results...")
with open('logs/wgdd/wgdd_results.json', 'r') as f:
    results = json.load(f)

distance_matrix = np.load('logs/wgdd/distance_matrix.npy')
conflict_tensor = np.load('logs/wgdd/conflict_tensor.npy')

# Extract data
clusters = np.array(results['clusters'])
objectives = np.array(results['objectives'])
pareto_scores = np.array(results['pareto_scores'])

edge_probs = np.array([cfg['edge_prob'] for cfg in results['mdp_configs']])
task_counts = np.array([(cfg['min_tasks'] + cfg['max_tasks']) / 2 for cfg in results['mdp_configs']])
task_lengths = np.array([(cfg['min_length'] + cfg['max_length']) / 2 for cfg in results['mdp_configs']])

# Compute various conflict metrics
n_obj = len(objectives)
n_mdp = len(clusters)

# 1. Mean conflict across all objectives
mean_conflict = np.array([np.mean(np.abs(conflict_tensor[i, :, :])) for i in range(n_mdp)])

# 2. Max conflict (worst-case)
max_conflict = np.array([np.max(np.abs(conflict_tensor[i, :, :])) for i in range(n_mdp)])

# 3. Conflict variance (stability)
conflict_var = np.array([np.var(conflict_tensor[i, :, :]) for i in range(n_mdp)])

# 4. Directional conflict (makespan vs energy extremes)
directional_conflict = np.array([conflict_tensor[i, 0, -1] for i in range(n_mdp)])  # pure makespan vs pure energy

print("Creating advanced conflict landscapes...")

# ============================================================================
# MEGA PLOT: 6-Panel Conflict Landscape
# ============================================================================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Custom colormap
colors_custom = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
n_bins = 100
cmap_custom = LinearSegmentedColormap.from_list('custom', colors_custom, N=n_bins)

# ============================================================================
# Panel 1: Mean Conflict Landscape (smooth contours)
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

x_grid = np.linspace(edge_probs.min(), edge_probs.max(), 100)
y_grid = np.linspace(pareto_scores.min(), pareto_scores.max(), 100)
X, Y = np.meshgrid(x_grid, y_grid)

Z = griddata((edge_probs, pareto_scores), mean_conflict, (X, Y), method='cubic')
Z_smooth = gaussian_filter(Z, sigma=1.5)

contour = ax1.contourf(X, Y, Z_smooth, levels=20, cmap=cmap_custom, alpha=0.8)
cbar = plt.colorbar(contour, ax=ax1, fraction=0.046)
cbar.set_label('Mean Conflict', fontsize=9)

# Overlay points
for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    marker = 'o' if label == 0 else 's'
    label_name = 'Wide' if label == 0 else 'LongCP'
    ax1.scatter(edge_probs[mask], pareto_scores[mask], s=80, c=color,
               marker=marker, alpha=0.9, edgecolors='white', linewidth=2, label=label_name)

ax1.set_xlabel('Edge Probability', fontsize=10, weight='bold')
ax1.set_ylabel('Pareto Alignment', fontsize=10, weight='bold')
ax1.set_title('(a) Mean Conflict Landscape', fontsize=11, weight='bold')
ax1.legend(fontsize=9, loc='best', framealpha=0.95)
ax1.grid(True, alpha=0.2, color='white', linewidth=0.5)

# ============================================================================
# Panel 2: Max Conflict (Worst-Case)
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

Z2 = griddata((edge_probs, pareto_scores), max_conflict, (X, Y), method='cubic')
Z2_smooth = gaussian_filter(Z2, sigma=1.5)

contour2 = ax2.contourf(X, Y, Z2_smooth, levels=20, cmap='Reds', alpha=0.8)
cbar2 = plt.colorbar(contour2, ax=ax2, fraction=0.046)
cbar2.set_label('Max Conflict', fontsize=9)

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    marker = 'o' if label == 0 else 's'
    ax2.scatter(edge_probs[mask], pareto_scores[mask], s=80, c=color,
               marker=marker, alpha=0.9, edgecolors='white', linewidth=2)

ax2.set_xlabel('Edge Probability', fontsize=10, weight='bold')
ax2.set_ylabel('Pareto Alignment', fontsize=10, weight='bold')
ax2.set_title('(b) Worst-Case Conflict', fontsize=11, weight='bold')
ax2.grid(True, alpha=0.2, color='white', linewidth=0.5)

# ============================================================================
# Panel 3: Conflict Variance (Stability)
# ============================================================================
ax3 = fig.add_subplot(gs[0, 2])

Z3 = griddata((edge_probs, pareto_scores), conflict_var, (X, Y), method='cubic')
Z3_smooth = gaussian_filter(Z3, sigma=1.5)

contour3 = ax3.contourf(X, Y, Z3_smooth, levels=20, cmap='viridis', alpha=0.8)
cbar3 = plt.colorbar(contour3, ax=ax3, fraction=0.046)
cbar3.set_label('Conflict Variance', fontsize=9)

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    marker = 'o' if label == 0 else 's'
    ax3.scatter(edge_probs[mask], pareto_scores[mask], s=80, c=color,
               marker=marker, alpha=0.9, edgecolors='white', linewidth=2)

ax3.set_xlabel('Edge Probability', fontsize=10, weight='bold')
ax3.set_ylabel('Pareto Alignment', fontsize=10, weight='bold')
ax3.set_title('(c) Conflict Stability', fontsize=11, weight='bold')
ax3.grid(True, alpha=0.2, color='white', linewidth=0.5)

# ============================================================================
# Panel 4: 3D Surface (Edge Prob vs Task Count vs Conflict)
# ============================================================================
ax4 = fig.add_subplot(gs[1, :], projection='3d')

# Create meshgrid for surface
x_surf = np.linspace(edge_probs.min(), edge_probs.max(), 50)
y_surf = np.linspace(task_counts.min(), task_counts.max(), 50)
X_surf, Y_surf = np.meshgrid(x_surf, y_surf)

Z_surf = griddata((edge_probs, task_counts), mean_conflict, (X_surf, Y_surf), method='cubic')

# Plot surface
surf = ax4.plot_surface(X_surf, Y_surf, Z_surf, cmap=cmap_custom, alpha=0.7,
                        linewidth=0, antialiased=True, edgecolor='none')

# Overlay scatter
for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    ax4.scatter(edge_probs[mask], task_counts[mask], mean_conflict[mask],
               s=100, c=color, alpha=0.9, edgecolors='black', linewidth=1.5,
               depthshade=True)

ax4.set_xlabel('Edge Probability', fontsize=10, weight='bold', labelpad=10)
ax4.set_ylabel('Task Count', fontsize=10, weight='bold', labelpad=10)
ax4.set_zlabel('Mean Conflict', fontsize=10, weight='bold', labelpad=10)
ax4.set_title('(d) 3D Conflict Surface: Structure × Parallelism', fontsize=11, weight='bold', pad=20)
ax4.view_init(elev=25, azim=45)

cbar4 = fig.colorbar(surf, ax=ax4, shrink=0.5, aspect=10)
cbar4.set_label('Conflict', fontsize=9)

# ============================================================================
# Panel 5: Directional Conflict Heatmap
# ============================================================================
ax5 = fig.add_subplot(gs[2, 0])

Z5 = griddata((edge_probs, pareto_scores), directional_conflict, (X, Y), method='cubic')
Z5_smooth = gaussian_filter(Z5, sigma=1.5)

contour5 = ax5.contourf(X, Y, Z5_smooth, levels=20, cmap='RdBu_r', alpha=0.8, vmin=-1, vmax=1)
cbar5 = plt.colorbar(contour5, ax=ax5, fraction=0.046)
cbar5.set_label('Makespan ↔ Energy', fontsize=9)

# Add contour lines
ax5.contour(X, Y, Z5_smooth, levels=10, colors='black', alpha=0.3, linewidths=0.5)

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    marker = 'o' if label == 0 else 's'
    ax5.scatter(edge_probs[mask], pareto_scores[mask], s=80, c=color,
               marker=marker, alpha=0.9, edgecolors='white', linewidth=2)

ax5.set_xlabel('Edge Probability', fontsize=10, weight='bold')
ax5.set_ylabel('Pareto Alignment', fontsize=10, weight='bold')
ax5.set_title('(e) Directional Conflict (Makespan vs Energy)', fontsize=11, weight='bold')
ax5.grid(True, alpha=0.2, color='white', linewidth=0.5)

# ============================================================================
# Panel 6: Conflict Gradient Field
# ============================================================================
ax6 = fig.add_subplot(gs[2, 1])

# Compute gradient of conflict landscape
dy, dx = np.gradient(Z_smooth)

# Subsample for quiver plot
step = 5
X_sub = X[::step, ::step]
Y_sub = Y[::step, ::step]
dx_sub = dx[::step, ::step]
dy_sub = dy[::step, ::step]

# Background contour
ax6.contourf(X, Y, Z_smooth, levels=15, cmap=cmap_custom, alpha=0.4)

# Gradient field
quiver = ax6.quiver(X_sub, Y_sub, dx_sub, dy_sub, 
                    np.sqrt(dx_sub**2 + dy_sub**2),
                    cmap='plasma', alpha=0.8, scale=20, width=0.003)

cbar6 = plt.colorbar(quiver, ax=ax6, fraction=0.046)
cbar6.set_label('Gradient Magnitude', fontsize=9)

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    marker = 'o' if label == 0 else 's'
    ax6.scatter(edge_probs[mask], pareto_scores[mask], s=80, c=color,
               marker=marker, alpha=0.9, edgecolors='white', linewidth=2)

ax6.set_xlabel('Edge Probability', fontsize=10, weight='bold')
ax6.set_ylabel('Pareto Alignment', fontsize=10, weight='bold')
ax6.set_title('(f) Conflict Gradient Field', fontsize=11, weight='bold')
ax6.grid(True, alpha=0.2, color='white', linewidth=0.5)

# ============================================================================
# Panel 7: Domain Separation Boundary
# ============================================================================
ax7 = fig.add_subplot(gs[2, 2])

# Plot conflict with decision boundary
Z7 = griddata((edge_probs, pareto_scores), mean_conflict, (X, Y), method='cubic')
Z7_smooth = gaussian_filter(Z7, sigma=1.5)

contour7 = ax7.contourf(X, Y, Z7_smooth, levels=20, cmap=cmap_custom, alpha=0.6)

# Draw decision boundary (approximate)
boundary_x = 0.5
ax7.axvline(boundary_x, color='white', linestyle='--', linewidth=3, 
           alpha=0.9, label='Decision Boundary')

# Shade regions
ax7.axvspan(edge_probs.min(), boundary_x, alpha=0.15, color='green', label='Wide Region')
ax7.axvspan(boundary_x, edge_probs.max(), alpha=0.15, color='red', label='LongCP Region')

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    marker = 'o' if label == 0 else 's'
    label_name = 'Wide' if label == 0 else 'LongCP'
    ax7.scatter(edge_probs[mask], pareto_scores[mask], s=100, c=color,
               marker=marker, alpha=0.9, edgecolors='white', linewidth=2, label=label_name)

ax7.set_xlabel('Edge Probability', fontsize=10, weight='bold')
ax7.set_ylabel('Pareto Alignment', fontsize=10, weight='bold')
ax7.set_title('(g) Domain Separation in Conflict Space', fontsize=11, weight='bold')
ax7.legend(fontsize=8, loc='best', framealpha=0.95, ncol=2)
ax7.grid(True, alpha=0.2, color='white', linewidth=0.5)

# Main title
fig.suptitle('Advanced Conflict Landscape Analysis', fontsize=16, weight='bold', y=0.995)

plt.savefig(output_dir / '19_advanced_conflict_landscape.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: 19_advanced_conflict_landscape.png")
plt.close()

# ============================================================================
# BONUS: Interactive-Style Multi-Metric Comparison
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

metrics = [
    (mean_conflict, 'Mean Conflict', cmap_custom),
    (max_conflict, 'Max Conflict', 'Reds'),
    (conflict_var, 'Conflict Variance', 'viridis'),
    (directional_conflict, 'Directional Conflict', 'RdBu_r'),
    (pareto_scores, 'Pareto Alignment', 'coolwarm'),
    (distance_matrix.mean(axis=1), 'Mean Distance', 'plasma')
]

for idx, (ax, (metric, title, cmap)) in enumerate(zip(axes.flat, metrics)):
    # Create smooth landscape
    Z_metric = griddata((edge_probs, pareto_scores), metric, (X, Y), method='cubic')
    Z_metric_smooth = gaussian_filter(Z_metric, sigma=1.5)
    
    vmin, vmax = None, None
    if 'Directional' in title or 'Pareto' in title:
        vmin, vmax = -1, 1
    
    contour = ax.contourf(X, Y, Z_metric_smooth, levels=20, cmap=cmap, alpha=0.8,
                          vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(contour, ax=ax, fraction=0.046)
    cbar.ax.tick_params(labelsize=8)
    
    # Overlay points
    for label in [0, 1]:
        mask = clusters == label
        color = '#32CD32' if label == 0 else '#FF6B6B'
        marker = 'o' if label == 0 else 's'
        ax.scatter(edge_probs[mask], pareto_scores[mask], s=60, c=color,
                  marker=marker, alpha=0.9, edgecolors='white', linewidth=1.5)
    
    ax.set_xlabel('Edge Probability', fontsize=9, weight='bold')
    ax.set_ylabel('Pareto Alignment', fontsize=9, weight='bold')
    ax.set_title(f'({chr(97+idx)}) {title}', fontsize=10, weight='bold')
    ax.grid(True, alpha=0.2, color='white', linewidth=0.5)

plt.suptitle('Multi-Metric Conflict Landscape Comparison', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(output_dir / '20_multimetric_landscape.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: 20_multimetric_landscape.png")
plt.close()

print("="*70)
print("✅ Advanced conflict landscapes created!")
print(f"\nOutput directory: {output_dir}")
print("\nNew figures:")
print("  19_advanced_conflict_landscape.png (7-panel mega plot)")
print("  20_multimetric_landscape.png (6-metric comparison)")
