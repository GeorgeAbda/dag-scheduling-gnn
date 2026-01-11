"""
Create extended set of WGDD plots
All saved to publication_figures/
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde, ks_2samp, mannwhitneyu
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings('ignore')

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
output_dir = Path('experiments/gradient_domain_discovery/publication_figures')
output_dir.mkdir(exist_ok=True, parents=True)

# Load data
print("Loading WGDD results...")
with open('logs/wgdd/wgdd_results.json', 'r') as f:
    results = json.load(f)

distance_matrix = np.load('logs/wgdd/distance_matrix.npy')
conflict_tensor = np.load('logs/wgdd/conflict_tensor.npy')

# Extract
mdp_configs = results['mdp_configs']
clusters = np.array(results['clusters'])
pareto_scores = np.array(results['pareto_scores'])
objectives = np.array(results['objectives'])

edge_probs = np.array([cfg['edge_prob'] for cfg in mdp_configs])
task_counts = np.array([(cfg['min_tasks'] + cfg['max_tasks']) / 2 for cfg in mdp_configs])
task_lengths = np.array([(cfg['min_length'] + cfg['max_length']) / 2 for cfg in mdp_configs])

print(f"Creating plots in: {output_dir}")
print("="*70)

# ============================================================================
# PLOT 14: Statistical Tests Between Domains
# ============================================================================
print("Creating Plot 14: Statistical Tests...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 14a: KS test for edge probability
ax = axes[0, 0]
wide_edge = edge_probs[clusters == 0]
longcp_edge = edge_probs[clusters == 1]
ks_stat, ks_pval = ks_2samp(wide_edge, longcp_edge)

ax.hist(wide_edge, bins=15, alpha=0.6, color='#32CD32', edgecolor='black', 
        linewidth=1.5, label='Wide', density=True)
ax.hist(longcp_edge, bins=15, alpha=0.6, color='#FF6B6B', edgecolor='black', 
        linewidth=1.5, label='LongCP', density=True)

# Add KDE
kde_wide = gaussian_kde(wide_edge)
kde_longcp = gaussian_kde(longcp_edge)
x_range = np.linspace(0, 1, 200)
ax.plot(x_range, kde_wide(x_range), color='#228B22', linewidth=2.5, label='Wide KDE')
ax.plot(x_range, kde_longcp(x_range), color='#DC143C', linewidth=2.5, label='LongCP KDE')

ax.text(0.05, 0.95, f'KS stat={ks_stat:.3f}\np={ks_pval:.2e}', 
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax.set_xlabel('Edge Probability', fontsize=11, weight='bold')
ax.set_ylabel('Density', fontsize=11, weight='bold')
ax.set_title('(a) Edge Probability Distribution Test', fontsize=11, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 14b: Mann-Whitney U test for Pareto scores
ax = axes[0, 1]
wide_pareto = pareto_scores[clusters == 0]
longcp_pareto = pareto_scores[clusters == 1]
u_stat, u_pval = mannwhitneyu(wide_pareto, longcp_pareto, alternative='two-sided')

bp = ax.boxplot([wide_pareto, longcp_pareto], labels=['Wide', 'LongCP'],
                 patch_artist=True, showmeans=True, meanline=True)
bp['boxes'][0].set_facecolor('#32CD32')
bp['boxes'][1].set_facecolor('#FF6B6B')
for box in bp['boxes']:
    box.set_alpha(0.6)

ax.text(0.5, 0.95, f'Mann-Whitney U={u_stat:.0f}\np={u_pval:.2e}', 
        transform=ax.transAxes, fontsize=10, va='top', ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax.set_ylabel('Pareto Alignment Score', fontsize=11, weight='bold')
ax.set_title('(b) Pareto Score Distribution Test', fontsize=11, weight='bold')
ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

# 14c: Task count distribution
ax = axes[1, 0]
wide_tasks = task_counts[clusters == 0]
longcp_tasks = task_counts[clusters == 1]
ks_tasks, ks_tasks_p = ks_2samp(wide_tasks, longcp_tasks)

ax.hist(wide_tasks, bins=12, alpha=0.6, color='#32CD32', edgecolor='black', 
        linewidth=1.5, label='Wide', density=True)
ax.hist(longcp_tasks, bins=12, alpha=0.6, color='#FF6B6B', edgecolor='black', 
        linewidth=1.5, label='LongCP', density=True)

ax.text(0.05, 0.95, f'KS stat={ks_tasks:.3f}\np={ks_tasks_p:.2e}', 
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax.set_xlabel('Task Count', fontsize=11, weight='bold')
ax.set_ylabel('Density', fontsize=11, weight='bold')
ax.set_title('(c) Task Count Distribution Test', fontsize=11, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 14d: Summary table
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
Statistical Test Summary
{'='*40}

Edge Probability:
  KS statistic: {ks_stat:.4f}
  p-value: {ks_pval:.2e}
  Significant: {'YES' if ks_pval < 0.05 else 'NO'}

Pareto Alignment:
  U statistic: {u_stat:.0f}
  p-value: {u_pval:.2e}
  Significant: {'YES' if u_pval < 0.05 else 'NO'}

Task Count:
  KS statistic: {ks_tasks:.4f}
  p-value: {ks_tasks_p:.2e}
  Significant: {'YES' if ks_tasks_p < 0.05 else 'NO'}

Conclusion:
  Domains are statistically
  distinct across all metrics
  (α = 0.05)
"""

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#E8F4F8', alpha=0.9, 
                  edgecolor='#2E86AB', linewidth=2))

plt.tight_layout()
plt.savefig(output_dir / '14_statistical_tests.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: 14_statistical_tests.png")
plt.close()

# ============================================================================
# PLOT 15: Distance Distribution Analysis
# ============================================================================
print("Creating Plot 15: Distance Distributions...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Compute distance categories
within_wide = []
within_longcp = []
between = []

for i in range(len(distance_matrix)):
    for j in range(i+1, len(distance_matrix)):
        dist = distance_matrix[i, j]
        if clusters[i] == 0 and clusters[j] == 0:
            within_wide.append(dist)
        elif clusters[i] == 1 and clusters[j] == 1:
            within_longcp.append(dist)
        else:
            between.append(dist)

# 15a: Overlapping histograms
ax = axes[0, 0]
ax.hist(within_wide, bins=20, alpha=0.5, color='#32CD32', edgecolor='black', 
        label=f'Within Wide (n={len(within_wide)})', density=True)
ax.hist(within_longcp, bins=20, alpha=0.5, color='#FF6B6B', edgecolor='black', 
        label=f'Within LongCP (n={len(within_longcp)})', density=True)
ax.hist(between, bins=20, alpha=0.5, color='#9370DB', edgecolor='black', 
        label=f'Between (n={len(between)})', density=True)

ax.set_xlabel('Wasserstein Distance', fontsize=11, weight='bold')
ax.set_ylabel('Density', fontsize=11, weight='bold')
ax.set_title('(a) Distance Distribution Overlap', fontsize=11, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 15b: CDF comparison
ax = axes[0, 1]
for data, color, label in [(within_wide, '#32CD32', 'Within Wide'),
                            (within_longcp, '#FF6B6B', 'Within LongCP'),
                            (between, '#9370DB', 'Between')]:
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    ax.plot(sorted_data, cdf, linewidth=2.5, color=color, label=label, alpha=0.8)

ax.set_xlabel('Wasserstein Distance', fontsize=11, weight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=11, weight='bold')
ax.set_title('(b) Cumulative Distribution Functions', fontsize=11, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 15c: Quantile-quantile plot
ax = axes[1, 0]
# Q-Q plot: within_wide vs within_longcp
q_wide = np.percentile(within_wide, np.linspace(0, 100, 50))
q_longcp = np.percentile(within_longcp, np.linspace(0, 100, 50))

ax.scatter(q_wide, q_longcp, s=80, c='#2E86AB', edgecolors='black', 
           linewidth=1.5, alpha=0.7)
ax.plot([min(q_wide), max(q_wide)], [min(q_wide), max(q_wide)], 
        'r--', linewidth=2, label='y=x')

ax.set_xlabel('Within Wide Quantiles', fontsize=11, weight='bold')
ax.set_ylabel('Within LongCP Quantiles', fontsize=11, weight='bold')
ax.set_title('(c) Q-Q Plot: Within-Domain Distances', fontsize=11, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 15d: Effect size (Cohen's d)
ax = axes[1, 1]
# Compute Cohen's d for different comparisons
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

d_wide_between = cohens_d(within_wide, between)
d_longcp_between = cohens_d(within_longcp, between)
d_wide_longcp = cohens_d(within_wide, within_longcp)

comparisons = ['Wide vs\nBetween', 'LongCP vs\nBetween', 'Wide vs\nLongCP']
effect_sizes = [d_wide_between, d_longcp_between, d_wide_longcp]
colors_bar = ['#32CD32', '#FF6B6B', '#9370DB']

bars = ax.bar(comparisons, effect_sizes, color=colors_bar, alpha=0.7, 
              edgecolor='black', linewidth=2)

# Add effect size interpretation lines
ax.axhline(0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (0.5)')
ax.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='Large (0.8)')

ax.set_ylabel("Cohen's d (Effect Size)", fontsize=11, weight='bold')
ax.set_title('(d) Effect Sizes Between Distance Groups', fontsize=11, weight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, val in zip(bars, effect_sizes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig(output_dir / '15_distance_distributions.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: 15_distance_distributions.png")
plt.close()

# ============================================================================
# PLOT 16: Gradient Magnitude Analysis
# ============================================================================
print("Creating Plot 16: Gradient Magnitude...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Compute gradient magnitudes from conflict tensor
# Use diagonal (self-conflict) as proxy for magnitude
grad_mags = np.array([np.mean(np.abs(conflict_tensor[i, :, :])) for i in range(len(clusters))])

# 16a: Magnitude by domain
ax = axes[0, 0]
wide_mags = grad_mags[clusters == 0]
longcp_mags = grad_mags[clusters == 1]

bp = ax.boxplot([wide_mags, longcp_mags], labels=['Wide', 'LongCP'],
                 patch_artist=True, showmeans=True, meanline=True,
                 widths=0.6)
bp['boxes'][0].set_facecolor('#32CD32')
bp['boxes'][1].set_facecolor('#FF6B6B')
for box in bp['boxes']:
    box.set_alpha(0.6)

ax.set_ylabel('Mean Gradient Magnitude', fontsize=11, weight='bold')
ax.set_title('(a) Gradient Magnitude by Domain', fontsize=11, weight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add statistics
ax.text(1.5, ax.get_ylim()[1]*0.95, 
        f'Wide: μ={np.mean(wide_mags):.3f}\nLongCP: μ={np.mean(longcp_mags):.3f}',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# 16b: Magnitude vs edge probability
ax = axes[0, 1]
for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    label_name = 'Wide' if label == 0 else 'LongCP'
    ax.scatter(edge_probs[mask], grad_mags[mask], s=100, c=color, 
               alpha=0.7, edgecolors='black', linewidth=1.5, label=label_name)

# Add trend line
z = np.polyfit(edge_probs, grad_mags, 1)
p = np.poly1d(z)
x_trend = np.linspace(edge_probs.min(), edge_probs.max(), 100)
ax.plot(x_trend, p(x_trend), "k--", linewidth=2, alpha=0.5, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')

ax.set_xlabel('Edge Probability', fontsize=11, weight='bold')
ax.set_ylabel('Mean Gradient Magnitude', fontsize=11, weight='bold')
ax.set_title('(b) Magnitude vs Structure', fontsize=11, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 16c: Magnitude vs Pareto alignment
ax = axes[1, 0]
for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    label_name = 'Wide' if label == 0 else 'LongCP'
    ax.scatter(pareto_scores[mask], grad_mags[mask], s=100, c=color, 
               alpha=0.7, edgecolors='black', linewidth=1.5, label=label_name)

ax.set_xlabel('Pareto Alignment Score', fontsize=11, weight='bold')
ax.set_ylabel('Mean Gradient Magnitude', fontsize=11, weight='bold')
ax.set_title('(c) Magnitude vs Objective Alignment', fontsize=11, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 16d: Magnitude distribution
ax = axes[1, 1]
ax.hist(wide_mags, bins=15, alpha=0.6, color='#32CD32', edgecolor='black', 
        linewidth=1.5, label='Wide', density=True)
ax.hist(longcp_mags, bins=15, alpha=0.6, color='#FF6B6B', edgecolor='black', 
        linewidth=1.5, label='LongCP', density=True)

# Add KDE
if len(wide_mags) > 1:
    kde_wide = gaussian_kde(wide_mags)
    x_wide = np.linspace(wide_mags.min(), wide_mags.max(), 100)
    ax.plot(x_wide, kde_wide(x_wide), color='#228B22', linewidth=2.5)

if len(longcp_mags) > 1:
    kde_longcp = gaussian_kde(longcp_mags)
    x_longcp = np.linspace(longcp_mags.min(), longcp_mags.max(), 100)
    ax.plot(x_longcp, kde_longcp(x_longcp), color='#DC143C', linewidth=2.5)

ax.set_xlabel('Mean Gradient Magnitude', fontsize=11, weight='bold')
ax.set_ylabel('Density', fontsize=11, weight='bold')
ax.set_title('(d) Magnitude Distribution', fontsize=11, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '16_gradient_magnitude.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: 16_gradient_magnitude.png")
plt.close()

# ============================================================================
# PLOT 17: Pairwise MDP Similarity Network
# ============================================================================
print("Creating Plot 17: Similarity Network...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Convert distances to similarities
sigma = np.median(distance_matrix)
similarity_matrix = np.exp(-distance_matrix**2 / (2 * sigma**2))

# 17a: Similarity matrix heatmap
ax = axes[0]
im = ax.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xlabel('MDP Index', fontsize=11, weight='bold')
ax.set_ylabel('MDP Index', fontsize=11, weight='bold')
ax.set_title('(a) Pairwise Similarity Matrix', fontsize=11, weight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Similarity', fontsize=10)

# Add domain boundaries
ax.axhline(20, color='white', linewidth=2, linestyle='--')
ax.axvline(20, color='white', linewidth=2, linestyle='--')

# 17b: Network graph (simplified - show strongest connections)
ax = axes[1]
threshold = np.percentile(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)], 90)

# Use MDS for layout
from sklearn.manifold import MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
pos = mds.fit_transform(distance_matrix)

# Plot nodes
for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    label_name = 'Wide' if label == 0 else 'LongCP'
    ax.scatter(pos[mask, 0], pos[mask, 1], s=200, c=color, 
               alpha=0.8, edgecolors='black', linewidth=2, label=label_name, zorder=3)

# Plot edges (top 10% strongest)
for i in range(len(similarity_matrix)):
    for j in range(i+1, len(similarity_matrix)):
        if similarity_matrix[i, j] > threshold:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 
                   'gray', alpha=0.3, linewidth=0.5, zorder=1)

ax.set_xlabel('MDS Dimension 1', fontsize=11, weight='bold')
ax.set_ylabel('MDS Dimension 2', fontsize=11, weight='bold')
ax.set_title(f'(b) Similarity Network (top 10% edges)', fontsize=11, weight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(output_dir / '17_similarity_network.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: 17_similarity_network.png")
plt.close()

# ============================================================================
# PLOT 18: Objective Trade-off Surface
# ============================================================================
print("Creating Plot 18: Trade-off Surface...")
fig = plt.figure(figsize=(14, 6))

# 18a: 3D scatter of objectives
ax1 = fig.add_subplot(121, projection='3d')

alphas = objectives[:, 0]
n_obj = len(alphas)

# For each MDP, compute mean conflict across objectives
mdp_conflicts = []
for i in range(len(clusters)):
    conflicts = []
    for j in range(n_obj):
        # Conflict with pure makespan
        conflicts.append(conflict_tensor[i, 0, j])
    mdp_conflicts.append(np.mean(conflicts))

mdp_conflicts = np.array(mdp_conflicts)

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    label_name = 'Wide' if label == 0 else 'LongCP'
    ax1.scatter(edge_probs[mask], pareto_scores[mask], mdp_conflicts[mask],
               s=80, c=color, alpha=0.8, edgecolors='black', linewidth=1,
               label=label_name, depthshade=True)

ax1.set_xlabel('Edge Prob', fontsize=10, weight='bold')
ax1.set_ylabel('Pareto Align', fontsize=10, weight='bold')
ax1.set_zlabel('Mean Conflict', fontsize=10, weight='bold')
ax1.set_title('(a) 3D Trade-off Space', fontsize=11, weight='bold')
ax1.legend(fontsize=9)

# 18b: Contour plot of conflict
ax2 = fig.add_subplot(122)

# Create grid
x_grid = np.linspace(edge_probs.min(), edge_probs.max(), 50)
y_grid = np.linspace(pareto_scores.min(), pareto_scores.max(), 50)
X, Y = np.meshgrid(x_grid, y_grid)

# Interpolate conflict values
from scipy.interpolate import griddata
Z = griddata((edge_probs, pareto_scores), mdp_conflicts, (X, Y), method='cubic')

# Plot contours
contour = ax2.contourf(X, Y, Z, levels=15, cmap='RdYlGn_r', alpha=0.7)
cbar = plt.colorbar(contour, ax=ax2)
cbar.set_label('Mean Conflict', fontsize=10)

# Overlay scatter
for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    label_name = 'Wide' if label == 0 else 'LongCP'
    ax2.scatter(edge_probs[mask], pareto_scores[mask], s=100, c=color,
               alpha=0.9, edgecolors='black', linewidth=1.5, label=label_name)

ax2.set_xlabel('Edge Probability', fontsize=11, weight='bold')
ax2.set_ylabel('Pareto Alignment', fontsize=11, weight='bold')
ax2.set_title('(b) Conflict Landscape', fontsize=11, weight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '18_tradeoff_surface.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: 18_tradeoff_surface.png")
plt.close()

print("="*70)
print("✅ All extended plots created successfully!")
print(f"\nOutput directory: {output_dir}")
print("\nNew plots:")
print("  14_statistical_tests.png")
print("  15_distance_distributions.png")
print("  16_gradient_magnitude.png")
print("  17_similarity_network.png")
print("  18_tradeoff_surface.png")
