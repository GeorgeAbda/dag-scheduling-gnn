"""
Create better conflict landscape visualizations
Using more meaningful axes than edge probability
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

# Compute conflict metrics
n_mdp = len(clusters)
mean_conflict = np.array([np.mean(np.abs(conflict_tensor[i, :, :])) for i in range(n_mdp)])
max_conflict = np.array([np.max(np.abs(conflict_tensor[i, :, :])) for i in range(n_mdp)])
conflict_var = np.array([np.var(conflict_tensor[i, :, :]) for i in range(n_mdp)])
directional_conflict = np.array([conflict_tensor[i, 0, -1] for i in range(n_mdp)])

# Compute derived metrics
parallelism = task_counts  # Width of DAG
computation = np.log10(task_lengths)  # Depth/length (log scale)
structure_ratio = task_counts / (computation + 1)  # Width/depth ratio

print("Creating improved conflict landscapes...")

# Custom colormap
colors_custom = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
cmap_custom = LinearSegmentedColormap.from_list('custom', colors_custom, N=100)

# ============================================================================
# MEGA PLOT: Better Axes
# ============================================================================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# ============================================================================
# Panel 1: Parallelism vs Pareto Alignment
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

x_grid = np.linspace(parallelism.min(), parallelism.max(), 100)
y_grid = np.linspace(pareto_scores.min(), pareto_scores.max(), 100)
X, Y = np.meshgrid(x_grid, y_grid)

Z = griddata((parallelism, pareto_scores), mean_conflict, (X, Y), method='cubic')
Z_smooth = gaussian_filter(Z, sigma=1.5)

contour = ax1.contourf(X, Y, Z_smooth, levels=20, cmap=cmap_custom, alpha=0.8)
cbar = plt.colorbar(contour, ax=ax1, fraction=0.046)
cbar.set_label('Mean Conflict', fontsize=9)

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    marker = 'o' if label == 0 else 's'
    label_name = 'Wide' if label == 0 else 'LongCP'
    ax1.scatter(parallelism[mask], pareto_scores[mask], s=80, c=color,
               marker=marker, alpha=0.9, edgecolors='white', linewidth=2, label=label_name)

ax1.set_xlabel('Parallelism (Task Count)', fontsize=10, weight='bold')
ax1.set_ylabel('Pareto Alignment', fontsize=10, weight='bold')
ax1.set_title('(a) Conflict vs Parallelism', fontsize=11, weight='bold')
ax1.legend(fontsize=9, loc='best', framealpha=0.95)
ax1.grid(True, alpha=0.2, color='white', linewidth=0.5)

# ============================================================================
# Panel 2: Computation vs Pareto Alignment
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

x_grid2 = np.linspace(computation.min(), computation.max(), 100)
X2, Y2 = np.meshgrid(x_grid2, y_grid)

Z2 = griddata((computation, pareto_scores), mean_conflict, (X2, Y2), method='cubic')
Z2_smooth = gaussian_filter(Z2, sigma=1.5)

contour2 = ax2.contourf(X2, Y2, Z2_smooth, levels=20, cmap=cmap_custom, alpha=0.8)
cbar2 = plt.colorbar(contour2, ax=ax2, fraction=0.046)
cbar2.set_label('Mean Conflict', fontsize=9)

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    marker = 'o' if label == 0 else 's'
    ax2.scatter(computation[mask], pareto_scores[mask], s=80, c=color,
               marker=marker, alpha=0.9, edgecolors='white', linewidth=2)

ax2.set_xlabel('Computation (log₁₀ Task Length)', fontsize=10, weight='bold')
ax2.set_ylabel('Pareto Alignment', fontsize=10, weight='bold')
ax2.set_title('(b) Conflict vs Computation', fontsize=11, weight='bold')
ax2.grid(True, alpha=0.2, color='white', linewidth=0.5)

# ============================================================================
# Panel 3: Structure Ratio vs Pareto Alignment
# ============================================================================
ax3 = fig.add_subplot(gs[0, 2])

x_grid3 = np.linspace(structure_ratio.min(), structure_ratio.max(), 100)
X3, Y3 = np.meshgrid(x_grid3, y_grid)

Z3 = griddata((structure_ratio, pareto_scores), mean_conflict, (X3, Y3), method='cubic')
Z3_smooth = gaussian_filter(Z3, sigma=1.5)

contour3 = ax3.contourf(X3, Y3, Z3_smooth, levels=20, cmap=cmap_custom, alpha=0.8)
cbar3 = plt.colorbar(contour3, ax=ax3, fraction=0.046)
cbar3.set_label('Mean Conflict', fontsize=9)

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    marker = 'o' if label == 0 else 's'
    ax3.scatter(structure_ratio[mask], pareto_scores[mask], s=80, c=color,
               marker=marker, alpha=0.9, edgecolors='white', linewidth=2)

ax3.set_xlabel('Structure Ratio (Width/Depth)', fontsize=10, weight='bold')
ax3.set_ylabel('Pareto Alignment', fontsize=10, weight='bold')
ax3.set_title('(c) Conflict vs DAG Shape', fontsize=11, weight='bold')
ax3.grid(True, alpha=0.2, color='white', linewidth=0.5)

# ============================================================================
# Panel 4: 3D Surface (Parallelism vs Computation vs Conflict)
# ============================================================================
ax4 = fig.add_subplot(gs[1, :], projection='3d')

x_surf = np.linspace(parallelism.min(), parallelism.max(), 50)
y_surf = np.linspace(computation.min(), computation.max(), 50)
X_surf, Y_surf = np.meshgrid(x_surf, y_surf)

Z_surf = griddata((parallelism, computation), mean_conflict, (X_surf, Y_surf), method='cubic')

surf = ax4.plot_surface(X_surf, Y_surf, Z_surf, cmap=cmap_custom, alpha=0.7,
                        linewidth=0, antialiased=True, edgecolor='none')

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    ax4.scatter(parallelism[mask], computation[mask], mean_conflict[mask],
               s=100, c=color, alpha=0.9, edgecolors='black', linewidth=1.5,
               depthshade=True)

ax4.set_xlabel('Parallelism\n(Task Count)', fontsize=10, weight='bold', labelpad=10)
ax4.set_ylabel('Computation\n(log Task Length)', fontsize=10, weight='bold', labelpad=10)
ax4.set_zlabel('Mean Conflict', fontsize=10, weight='bold', labelpad=10)
ax4.set_title('(d) 3D Conflict Surface: Parallelism × Computation', fontsize=11, weight='bold', pad=20)
ax4.view_init(elev=25, azim=45)

cbar4 = fig.colorbar(surf, ax=ax4, shrink=0.5, aspect=10)
cbar4.set_label('Conflict', fontsize=9)

# ============================================================================
# Panel 5: Directional Conflict (Parallelism axis)
# ============================================================================
ax5 = fig.add_subplot(gs[2, 0])

Z5 = griddata((parallelism, pareto_scores), directional_conflict, (X, Y), method='cubic')
Z5_smooth = gaussian_filter(Z5, sigma=1.5)

contour5 = ax5.contourf(X, Y, Z5_smooth, levels=20, cmap='RdBu_r', alpha=0.8, vmin=-1, vmax=1)
cbar5 = plt.colorbar(contour5, ax=ax5, fraction=0.046)
cbar5.set_label('Makespan ↔ Energy', fontsize=9)

ax5.contour(X, Y, Z5_smooth, levels=10, colors='black', alpha=0.3, linewidths=0.5)

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    marker = 'o' if label == 0 else 's'
    ax5.scatter(parallelism[mask], pareto_scores[mask], s=80, c=color,
               marker=marker, alpha=0.9, edgecolors='white', linewidth=2)

ax5.set_xlabel('Parallelism (Task Count)', fontsize=10, weight='bold')
ax5.set_ylabel('Pareto Alignment', fontsize=10, weight='bold')
ax5.set_title('(e) Directional Conflict Landscape', fontsize=11, weight='bold')
ax5.grid(True, alpha=0.2, color='white', linewidth=0.5)

# ============================================================================
# Panel 6: Gradient Field (Parallelism axis)
# ============================================================================
ax6 = fig.add_subplot(gs[2, 1])

dy, dx = np.gradient(Z_smooth)

step = 5
X_sub = X[::step, ::step]
Y_sub = Y[::step, ::step]
dx_sub = dx[::step, ::step]
dy_sub = dy[::step, ::step]

ax6.contourf(X, Y, Z_smooth, levels=15, cmap=cmap_custom, alpha=0.4)

quiver = ax6.quiver(X_sub, Y_sub, dx_sub, dy_sub, 
                    np.sqrt(dx_sub**2 + dy_sub**2),
                    cmap='plasma', alpha=0.8, scale=20, width=0.003)

cbar6 = plt.colorbar(quiver, ax=ax6, fraction=0.046)
cbar6.set_label('Gradient Magnitude', fontsize=9)

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    marker = 'o' if label == 0 else 's'
    ax6.scatter(parallelism[mask], pareto_scores[mask], s=80, c=color,
               marker=marker, alpha=0.9, edgecolors='white', linewidth=2)

ax6.set_xlabel('Parallelism (Task Count)', fontsize=10, weight='bold')
ax6.set_ylabel('Pareto Alignment', fontsize=10, weight='bold')
ax6.set_title('(f) Conflict Gradient Field', fontsize=11, weight='bold')
ax6.grid(True, alpha=0.2, color='white', linewidth=0.5)

# ============================================================================
# Panel 7: Parallelism vs Computation (Domain Separation)
# ============================================================================
ax7 = fig.add_subplot(gs[2, 2])

x_grid7 = np.linspace(parallelism.min(), parallelism.max(), 100)
y_grid7 = np.linspace(computation.min(), computation.max(), 100)
X7, Y7 = np.meshgrid(x_grid7, y_grid7)

Z7 = griddata((parallelism, computation), mean_conflict, (X7, Y7), method='cubic')
Z7_smooth = gaussian_filter(Z7, sigma=1.5)

contour7 = ax7.contourf(X7, Y7, Z7_smooth, levels=20, cmap=cmap_custom, alpha=0.6)

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    marker = 'o' if label == 0 else 's'
    label_name = 'Wide' if label == 0 else 'LongCP'
    ax7.scatter(parallelism[mask], computation[mask], s=100, c=color,
               marker=marker, alpha=0.9, edgecolors='white', linewidth=2, label=label_name)

ax7.set_xlabel('Parallelism (Task Count)', fontsize=10, weight='bold')
ax7.set_ylabel('Computation (log Task Length)', fontsize=10, weight='bold')
ax7.set_title('(g) Domain Separation: Width vs Depth', fontsize=11, weight='bold')
ax7.legend(fontsize=9, loc='best', framealpha=0.95)
ax7.grid(True, alpha=0.2, color='white', linewidth=0.5)

fig.suptitle('Conflict Landscape: Parallelism & Computation Perspectives', 
             fontsize=16, weight='bold', y=0.995)

plt.savefig(output_dir / '19_advanced_conflict_landscape.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: 19_advanced_conflict_landscape.png")
plt.close()

# ============================================================================
# BONUS: Multi-Axis Comparison
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axis_configs = [
    (parallelism, pareto_scores, 'Parallelism (Tasks)', 'Pareto Alignment'),
    (computation, pareto_scores, 'Computation (log Length)', 'Pareto Alignment'),
    (structure_ratio, pareto_scores, 'Width/Depth Ratio', 'Pareto Alignment'),
    (parallelism, computation, 'Parallelism (Tasks)', 'Computation (log Length)'),
    (parallelism, distance_matrix.mean(axis=1), 'Parallelism (Tasks)', 'Mean Distance'),
    (computation, distance_matrix.mean(axis=1), 'Computation (log Length)', 'Mean Distance'),
]

for idx, (ax, (x_data, y_data, xlabel, ylabel)) in enumerate(zip(axes.flat, axis_configs)):
    x_g = np.linspace(x_data.min(), x_data.max(), 100)
    y_g = np.linspace(y_data.min(), y_data.max(), 100)
    Xg, Yg = np.meshgrid(x_g, y_g)
    
    Zg = griddata((x_data, y_data), mean_conflict, (Xg, Yg), method='cubic')
    Zg_smooth = gaussian_filter(Zg, sigma=1.5)
    
    contour = ax.contourf(Xg, Yg, Zg_smooth, levels=20, cmap=cmap_custom, alpha=0.8)
    cbar = plt.colorbar(contour, ax=ax, fraction=0.046)
    cbar.ax.tick_params(labelsize=8)
    
    for label in [0, 1]:
        mask = clusters == label
        color = '#32CD32' if label == 0 else '#FF6B6B'
        marker = 'o' if label == 0 else 's'
        ax.scatter(x_data[mask], y_data[mask], s=60, c=color,
                  marker=marker, alpha=0.9, edgecolors='white', linewidth=1.5)
    
    ax.set_xlabel(xlabel, fontsize=9, weight='bold')
    ax.set_ylabel(ylabel, fontsize=9, weight='bold')
    ax.set_title(f'({chr(97+idx)}) {xlabel} vs {ylabel}', fontsize=10, weight='bold')
    ax.grid(True, alpha=0.2, color='white', linewidth=0.5)

plt.suptitle('Multi-Axis Conflict Landscape Analysis', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(output_dir / '20_multimetric_landscape.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: 20_multimetric_landscape.png")
plt.close()

print("="*70)
print("✅ Improved conflict landscapes created!")
print(f"\nOutput directory: {output_dir}")
print("\nNew axes used:")
print("  • Parallelism (task count) - width of DAG")
print("  • Computation (log task length) - depth of DAG")
print("  • Structure ratio (width/depth) - shape of DAG")
print("  • Pareto alignment - objective conflict")
print("\nFigures:")
print("  19_advanced_conflict_landscape.png (7-panel)")
print("  20_multimetric_landscape.png (6-axis comparison)")
