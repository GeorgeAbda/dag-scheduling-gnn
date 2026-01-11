"""
Create detailed Wasserstein distance visualizations
Shows how W1 distances capture gradient distribution differences
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import seaborn as sns
from pathlib import Path
from scipy.stats import wasserstein_distance, gaussian_kde
from scipy.interpolate import interp1d
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

clusters = np.array(results['clusters'])
edge_probs = np.array([cfg['edge_prob'] for cfg in results['mdp_configs']])

print("Creating Wasserstein distance visualizations...")

# ============================================================================
# FIGURE 1: Wasserstein Distance Concept
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Generate example gradient distributions
np.random.seed(42)

# Panel 1a: Two similar distributions (low W1)
ax = axes[0, 0]
dist1 = np.random.normal(0, 1, 1000)
dist2 = np.random.normal(0.2, 1.1, 1000)

ax.hist(dist1, bins=30, alpha=0.5, color='#32CD32', edgecolor='black', 
        density=True, label='Distribution A')
ax.hist(dist2, bins=30, alpha=0.5, color='#228B22', edgecolor='black', 
        density=True, label='Distribution B')

w1 = wasserstein_distance(dist1, dist2)
ax.set_title(f'(a) Similar Distributions\nW₁ = {w1:.3f}', fontsize=11, weight='bold')
ax.set_xlabel('Gradient Value', fontsize=10, weight='bold')
ax.set_ylabel('Density', fontsize=10, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 1b: Two different distributions (high W1)
ax = axes[0, 1]
dist3 = np.random.normal(-1, 0.8, 1000)
dist4 = np.random.normal(1.5, 0.9, 1000)

ax.hist(dist3, bins=30, alpha=0.5, color='#32CD32', edgecolor='black', 
        density=True, label='Distribution C')
ax.hist(dist4, bins=30, alpha=0.5, color='#FF6B6B', edgecolor='black', 
        density=True, label='Distribution D')

w2 = wasserstein_distance(dist3, dist4)
ax.set_title(f'(b) Different Distributions\nW₁ = {w2:.3f}', fontsize=11, weight='bold')
ax.set_xlabel('Gradient Value', fontsize=10, weight='bold')
ax.set_ylabel('Density', fontsize=10, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 1c: Earth Mover's Distance visualization
ax = axes[0, 2]

# Create simple discrete distributions
x = np.array([1, 2, 3, 4, 5])
p = np.array([0.1, 0.3, 0.4, 0.15, 0.05])
q = np.array([0.05, 0.1, 0.2, 0.4, 0.25])

ax.bar(x - 0.15, p, width=0.3, color='#32CD32', alpha=0.7, 
       edgecolor='black', linewidth=1.5, label='Source')
ax.bar(x + 0.15, q, width=0.3, color='#FF6B6B', alpha=0.7, 
       edgecolor='black', linewidth=1.5, label='Target')

# Draw arrows showing "earth moving"
for i in range(len(x)):
    if p[i] > q[i]:
        ax.arrow(x[i] - 0.15, p[i], 0.3, q[i] - p[i], 
                head_width=0.15, head_length=0.02, fc='gray', ec='gray', alpha=0.5)

ax.set_title('(c) Earth Mover\'s Distance\n(Optimal Transport)', fontsize=11, weight='bold')
ax.set_xlabel('Gradient Bin', fontsize=10, weight='bold')
ax.set_ylabel('Probability Mass', fontsize=10, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel 2a: CDF comparison (low W1)
ax = axes[1, 0]
sorted1 = np.sort(dist1)
sorted2 = np.sort(dist2)
cdf1 = np.arange(1, len(sorted1)+1) / len(sorted1)
cdf2 = np.arange(1, len(sorted2)+1) / len(sorted2)

ax.plot(sorted1, cdf1, color='#32CD32', linewidth=2.5, label='CDF A')
ax.plot(sorted2, cdf2, color='#228B22', linewidth=2.5, label='CDF B')
ax.fill_between(sorted1, cdf1, interp1d(sorted2, cdf2, bounds_error=False, 
                fill_value=(0, 1))(sorted1), alpha=0.3, color='yellow', 
                label=f'W₁ area = {w1:.3f}')

ax.set_title('(d) CDF Comparison (Similar)', fontsize=11, weight='bold')
ax.set_xlabel('Gradient Value', fontsize=10, weight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=10, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2b: CDF comparison (high W1)
ax = axes[1, 1]
sorted3 = np.sort(dist3)
sorted4 = np.sort(dist4)
cdf3 = np.arange(1, len(sorted3)+1) / len(sorted3)
cdf4 = np.arange(1, len(sorted4)+1) / len(sorted4)

ax.plot(sorted3, cdf3, color='#32CD32', linewidth=2.5, label='CDF C')
ax.plot(sorted4, cdf4, color='#FF6B6B', linewidth=2.5, label='CDF D')
ax.fill_between(sorted3, cdf3, interp1d(sorted4, cdf4, bounds_error=False, 
                fill_value=(0, 1))(sorted3), alpha=0.3, color='orange', 
                label=f'W₁ area = {w2:.3f}')

ax.set_title('(e) CDF Comparison (Different)', fontsize=11, weight='bold')
ax.set_xlabel('Gradient Value', fontsize=10, weight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=10, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2c: W1 interpretation
ax = axes[1, 2]
ax.axis('off')

explanation = """
Wasserstein Distance (W₁)

Definition:
  Minimum "work" to transform one 
  distribution into another

Properties:
  • Metric: satisfies triangle inequality
  • Sensitive to location shifts
  • Robust to noise
  • Captures distribution shape

Interpretation:
  • Low W₁: Similar gradient patterns
    → Same domain
  
  • High W₁: Different gradient patterns
    → Different domains

Why W₁ for WGDD?
  ✓ Captures full distribution shape
  ✓ More informative than moments
  ✓ Natural metric for gradients
  ✓ Enables spectral clustering
"""

ax.text(0.1, 0.9, explanation, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#E8F4F8', alpha=0.9, 
                  edgecolor='#2E86AB', linewidth=2))

plt.suptitle('Wasserstein Distance: Concept and Computation', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(output_dir / '22_wasserstein_concept.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: 22_wasserstein_concept.png")
plt.close()

# ============================================================================
# FIGURE 2: Actual Wasserstein Distances in WGDD
# ============================================================================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Panel 1: Distance matrix with annotations
ax1 = fig.add_subplot(gs[0, :])

# Reorder by cluster
order = np.argsort(clusters)
distance_sorted = distance_matrix[np.ix_(order, order)]

im = ax1.imshow(distance_sorted, cmap='viridis', aspect='auto')

# Add domain boundaries
ax1.axhline(20, color='white', linewidth=3, linestyle='--')
ax1.axvline(20, color='white', linewidth=3, linestyle='--')

# Add rectangles for within-domain blocks
rect1 = Rectangle((0, 0), 20, 20, fill=False, edgecolor='#32CD32', linewidth=3)
rect2 = Rectangle((20, 20), 20, 20, fill=False, edgecolor='#FF6B6B', linewidth=3)
ax1.add_patch(rect1)
ax1.add_patch(rect2)

ax1.set_xlabel('MDP Index (sorted by cluster)', fontsize=11, weight='bold')
ax1.set_ylabel('MDP Index (sorted by cluster)', fontsize=11, weight='bold')
ax1.set_title('(a) Wasserstein Distance Matrix (40×40 MDPs)', fontsize=12, weight='bold')

cbar = plt.colorbar(im, ax=ax1, fraction=0.02)
cbar.set_label('W₁ Distance', fontsize=10)

# Add annotations
ax1.text(10, -3, 'Wide Domain', ha='center', fontsize=11, weight='bold', color='#32CD32')
ax1.text(30, -3, 'LongCP Domain', ha='center', fontsize=11, weight='bold', color='#FF6B6B')
ax1.text(-3, 10, 'Wide', va='center', rotation=90, fontsize=11, weight='bold', color='#32CD32')
ax1.text(-3, 30, 'LongCP', va='center', rotation=90, fontsize=11, weight='bold', color='#FF6B6B')

# Panel 2: Distance distribution by category
ax2 = fig.add_subplot(gs[1, 0])

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

ax2.hist(within_wide, bins=20, alpha=0.6, color='#32CD32', edgecolor='black', 
         linewidth=1.5, label=f'Within Wide (n={len(within_wide)})', density=True)
ax2.hist(within_longcp, bins=20, alpha=0.6, color='#FF6B6B', edgecolor='black', 
         linewidth=1.5, label=f'Within LongCP (n={len(within_longcp)})', density=True)
ax2.hist(between, bins=20, alpha=0.6, color='#9370DB', edgecolor='black', 
         linewidth=1.5, label=f'Between (n={len(between)})', density=True)

ax2.set_xlabel('W₁ Distance', fontsize=10, weight='bold')
ax2.set_ylabel('Density', fontsize=10, weight='bold')
ax2.set_title('(b) Distance Distribution by Category', fontsize=11, weight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: Cumulative distributions
ax3 = fig.add_subplot(gs[1, 1])

for data, color, label in [(within_wide, '#32CD32', 'Within Wide'),
                            (within_longcp, '#FF6B6B', 'Within LongCP'),
                            (between, '#9370DB', 'Between')]:
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    ax3.plot(sorted_data, cdf, linewidth=2.5, color=color, label=label, alpha=0.8)

ax3.set_xlabel('W₁ Distance', fontsize=10, weight='bold')
ax3.set_ylabel('Cumulative Probability', fontsize=10, weight='bold')
ax3.set_title('(c) Cumulative Distribution Functions', fontsize=11, weight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: Statistics summary
ax4 = fig.add_subplot(gs[1, 2])
ax4.axis('off')

stats_text = f"""
Wasserstein Distance Statistics

Within Wide Domain:
  Mean:   {np.mean(within_wide):.4f}
  Std:    {np.std(within_wide):.4f}
  Median: {np.median(within_wide):.4f}
  Range:  [{np.min(within_wide):.4f}, {np.max(within_wide):.4f}]

Within LongCP Domain:
  Mean:   {np.mean(within_longcp):.4f}
  Std:    {np.std(within_longcp):.4f}
  Median: {np.median(within_longcp):.4f}
  Range:  [{np.min(within_longcp):.4f}, {np.max(within_longcp):.4f}]

Between Domains:
  Mean:   {np.mean(between):.4f}
  Std:    {np.std(between):.4f}
  Median: {np.median(between):.4f}
  Range:  [{np.min(between):.4f}, {np.max(between):.4f}]

Separation Metrics:
  Ratio:  {np.mean(between) / np.mean(within_wide + within_longcp):.2f}×
  Gap:    {np.mean(between) - np.mean(within_wide + within_longcp):.4f}
"""

ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#FFF8DC', alpha=0.9, 
                   edgecolor='#F77F00', linewidth=2))

# Panel 5: Distance vs edge probability
ax5 = fig.add_subplot(gs[2, 0])

# Compute mean distance to other domain
mean_dist_to_other = []
for i in range(len(clusters)):
    if clusters[i] == 0:
        # Mean distance to LongCP
        mean_dist_to_other.append(distance_matrix[i, clusters == 1].mean())
    else:
        # Mean distance to Wide
        mean_dist_to_other.append(distance_matrix[i, clusters == 0].mean())

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    label_name = 'Wide' if label == 0 else 'LongCP'
    ax5.scatter(edge_probs[mask], np.array(mean_dist_to_other)[mask], 
               s=100, c=color, alpha=0.7, edgecolors='black', linewidth=1.5, 
               label=label_name)

ax5.set_xlabel('Edge Probability', fontsize=10, weight='bold')
ax5.set_ylabel('Mean W₁ Distance to Other Domain', fontsize=10, weight='bold')
ax5.set_title('(d) Cross-Domain Distance vs Structure', fontsize=11, weight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# Panel 6: Pairwise distance scatter
ax6 = fig.add_subplot(gs[2, 1])

# Sample pairs for visualization
np.random.seed(42)
sample_indices = np.random.choice(len(within_wide), min(100, len(within_wide)), replace=False)

ax6.scatter(range(len(sample_indices)), np.array(within_wide)[sample_indices], 
           s=50, c='#32CD32', alpha=0.6, label='Within Wide')
ax6.scatter(range(len(sample_indices)), np.array(within_longcp)[sample_indices], 
           s=50, c='#FF6B6B', alpha=0.6, label='Within LongCP')
ax6.scatter(range(len(sample_indices)), np.array(between)[sample_indices], 
           s=50, c='#9370DB', alpha=0.6, label='Between')

ax6.axhline(np.mean(within_wide), color='#32CD32', linestyle='--', linewidth=2, alpha=0.7)
ax6.axhline(np.mean(within_longcp), color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.7)
ax6.axhline(np.mean(between), color='#9370DB', linestyle='--', linewidth=2, alpha=0.7)

ax6.set_xlabel('Sample Index', fontsize=10, weight='bold')
ax6.set_ylabel('W₁ Distance', fontsize=10, weight='bold')
ax6.set_title('(e) Sample Pairwise Distances', fontsize=11, weight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# Panel 7: Affinity matrix (from distances)
ax7 = fig.add_subplot(gs[2, 2])

# Compute affinity
sigma = np.median(distance_matrix)
affinity = np.exp(-distance_matrix**2 / (2 * sigma**2))
affinity_sorted = affinity[np.ix_(order, order)]

im7 = ax7.imshow(affinity_sorted, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax7.axhline(20, color='white', linewidth=2, linestyle='--')
ax7.axvline(20, color='white', linewidth=2, linestyle='--')

ax7.set_xlabel('MDP Index (sorted)', fontsize=10, weight='bold')
ax7.set_ylabel('MDP Index (sorted)', fontsize=10, weight='bold')
ax7.set_title(f'(f) Affinity Matrix (σ={sigma:.3f})', fontsize=11, weight='bold')

cbar7 = plt.colorbar(im7, ax=ax7, fraction=0.046)
cbar7.set_label('Affinity', fontsize=9)

plt.suptitle('Wasserstein Distances in WGDD: Analysis and Interpretation', 
             fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(output_dir / '23_wasserstein_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: 23_wasserstein_analysis.png")
plt.close()

print("="*70)
print("✅ Wasserstein distance visualizations created!")
print(f"\nOutput directory: {output_dir}")
print("\nNew figures:")
print("  22_wasserstein_concept.png - Concept and computation")
print("  23_wasserstein_analysis.png - Actual distances in WGDD")
print("\nKey insights:")
print(f"  • Within-domain distance: {np.mean(within_wide + within_longcp):.4f}")
print(f"  • Between-domain distance: {np.mean(between):.4f}")
print(f"  • Separation ratio: {np.mean(between) / np.mean(within_wide + within_longcp):.2f}×")
