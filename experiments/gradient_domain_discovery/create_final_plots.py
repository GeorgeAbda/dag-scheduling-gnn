"""
Create final set of diverse and interesting WGDD visualizations
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Wedge, Polygon
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde, entropy
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import SpectralEmbedding
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
pareto_scores = np.array(results['pareto_scores'])
objectives = np.array(results['objectives'])
k_scores = results['k_scores']

edge_probs = np.array([cfg['edge_prob'] for cfg in results['mdp_configs']])
task_counts = np.array([(cfg['min_tasks'] + cfg['max_tasks']) / 2 for cfg in results['mdp_configs']])
task_lengths = np.array([(cfg['min_length'] + cfg['max_length']) / 2 for cfg in results['mdp_configs']])

n_mdp = len(clusters)
n_obj = len(objectives)

print("Creating final visualization set...")

# ============================================================================
# FIGURE 24: Spectral Clustering Process
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Compute affinity and Laplacian
sigma = np.median(distance_matrix)
affinity = np.exp(-distance_matrix**2 / (2 * sigma**2))
np.fill_diagonal(affinity, 0)
degree = np.diag(affinity.sum(axis=1))
laplacian = degree - affinity

# Panel 1: Affinity matrix
ax = axes[0, 0]
order = np.argsort(clusters)
affinity_sorted = affinity[np.ix_(order, order)]
im = ax.imshow(affinity_sorted, cmap='YlOrRd', aspect='auto')
ax.axhline(20, color='white', linewidth=2, linestyle='--')
ax.axvline(20, color='white', linewidth=2, linestyle='--')
ax.set_title('(a) Affinity Matrix', fontsize=11, weight='bold')
ax.set_xlabel('MDP Index (sorted)', fontsize=10, weight='bold')
ax.set_ylabel('MDP Index (sorted)', fontsize=10, weight='bold')
plt.colorbar(im, ax=ax, fraction=0.046)

# Panel 2: Degree matrix
ax = axes[0, 1]
degree_sorted = degree[np.ix_(order, order)]
im = ax.imshow(degree_sorted, cmap='Blues', aspect='auto')
ax.set_title('(b) Degree Matrix', fontsize=11, weight='bold')
ax.set_xlabel('MDP Index (sorted)', fontsize=10, weight='bold')
ax.set_ylabel('MDP Index (sorted)', fontsize=10, weight='bold')
plt.colorbar(im, ax=ax, fraction=0.046)

# Panel 3: Laplacian matrix
ax = axes[0, 2]
laplacian_sorted = laplacian[np.ix_(order, order)]
im = ax.imshow(laplacian_sorted, cmap='RdBu_r', aspect='auto')
ax.axhline(20, color='white', linewidth=2, linestyle='--')
ax.axvline(20, color='white', linewidth=2, linestyle='--')
ax.set_title('(c) Graph Laplacian', fontsize=11, weight='bold')
ax.set_xlabel('MDP Index (sorted)', fontsize=10, weight='bold')
ax.set_ylabel('MDP Index (sorted)', fontsize=10, weight='bold')
plt.colorbar(im, ax=ax, fraction=0.046)

# Panel 4: Eigenvalue spectrum
ax = axes[1, 0]
eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
eigenvalues = np.sort(eigenvalues)
ax.plot(eigenvalues[:20], 'o-', color='#2E86AB', linewidth=2.5, markersize=8)
ax.axvline(2, color='red', linestyle='--', linewidth=2, label='k=2 (optimal)')
ax.set_title('(d) Laplacian Eigenvalue Spectrum', fontsize=11, weight='bold')
ax.set_xlabel('Eigenvalue Index', fontsize=10, weight='bold')
ax.set_ylabel('Eigenvalue', fontsize=10, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 5: First two eigenvectors
ax = axes[1, 1]
embedding = SpectralEmbedding(n_components=2, affinity='precomputed')
coords = embedding.fit_transform(affinity)

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    marker = 'o' if label == 0 else 's'
    label_name = 'Wide' if label == 0 else 'LongCP'
    ax.scatter(coords[mask, 0], coords[mask, 1], s=120, c=color,
               marker=marker, alpha=0.8, edgecolors='black', linewidth=2, label=label_name)

ax.set_title('(e) Spectral Embedding', fontsize=11, weight='bold')
ax.set_xlabel('Eigenvector 1', fontsize=10, weight='bold')
ax.set_ylabel('Eigenvector 2', fontsize=10, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 6: Eigengap
ax = axes[1, 2]
eigengaps = np.diff(eigenvalues[:10])
ax.bar(range(len(eigengaps)), eigengaps, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(eigengaps[1], color='red', linestyle='--', linewidth=2, label='Largest gap')
ax.set_title('(f) Eigengap Analysis', fontsize=11, weight='bold')
ax.set_xlabel('Gap Index', fontsize=10, weight='bold')
ax.set_ylabel('Eigenvalue Gap', fontsize=10, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Spectral Clustering: From Affinity to Clusters', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(output_dir / '24_spectral_clustering.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: 24_spectral_clustering.png")
plt.close()

# ============================================================================
# FIGURE 25: Multi-Objective Gradient Analysis
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Panel 1: Objective weight distribution
ax = axes[0, 0]
alphas = objectives[:, 0]
ax.scatter(alphas, 1 - alphas, s=150, c=alphas, cmap='RdYlGn_r', 
           edgecolors='black', linewidth=1.5, alpha=0.8)
ax.plot([0, 1], [1, 0], 'k--', linewidth=2, alpha=0.5)
ax.set_xlabel('α (Makespan Weight)', fontsize=10, weight='bold')
ax.set_ylabel('1-α (Energy Weight)', fontsize=10, weight='bold')
ax.set_title('(a) Objective Weight Sampling', fontsize=11, weight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)

# Panel 2: Conflict by objective type
ax = axes[0, 1]
obj_types = []
for alpha in alphas:
    if alpha > 0.8:
        obj_types.append('Makespan\nFocused')
    elif alpha < 0.2:
        obj_types.append('Energy\nFocused')
    else:
        obj_types.append('Balanced')

# Compute mean conflict for each objective
obj_conflicts = []
for obj_idx in range(n_obj):
    conflicts = []
    for mdp_idx in range(n_mdp):
        conflicts.append(np.mean(np.abs(conflict_tensor[mdp_idx, obj_idx, :])))
    obj_conflicts.append(np.mean(conflicts))

colors_obj = ['#DC143C' if t == 'Makespan\nFocused' else '#228B22' if t == 'Energy\nFocused' else '#F77F00' 
              for t in obj_types]
ax.scatter(alphas, obj_conflicts, s=120, c=colors_obj, edgecolors='black', linewidth=1.5, alpha=0.8)
ax.set_xlabel('α (Makespan Weight)', fontsize=10, weight='bold')
ax.set_ylabel('Mean Conflict Magnitude', fontsize=10, weight='bold')
ax.set_title('(b) Conflict vs Objective Type', fontsize=11, weight='bold')
ax.grid(True, alpha=0.3)

# Panel 3: Conflict correlation matrix
ax = axes[0, 2]
# Compute correlation between objectives
obj_correlations = np.zeros((n_obj, n_obj))
for i in range(n_obj):
    for j in range(n_obj):
        # Correlation of conflicts across MDPs
        conflicts_i = [conflict_tensor[m, i, :].mean() for m in range(n_mdp)]
        conflicts_j = [conflict_tensor[m, j, :].mean() for m in range(n_mdp)]
        obj_correlations[i, j] = np.corrcoef(conflicts_i, conflicts_j)[0, 1]

im = ax.imshow(obj_correlations, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax.set_title('(c) Objective Conflict Correlation', fontsize=11, weight='bold')
ax.set_xlabel('Objective Index', fontsize=10, weight='bold')
ax.set_ylabel('Objective Index', fontsize=10, weight='bold')
plt.colorbar(im, ax=ax, fraction=0.046)

# Panel 4: Gradient diversity (entropy)
ax = axes[1, 0]
gradient_entropies = []
for mdp_idx in range(n_mdp):
    # Compute entropy of conflict distribution
    conflicts = conflict_tensor[mdp_idx, :, :].flatten()
    hist, _ = np.histogram(conflicts, bins=20, density=True)
    hist = hist[hist > 0]  # Remove zeros
    gradient_entropies.append(entropy(hist))

gradient_entropies = np.array(gradient_entropies)

for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    label_name = 'Wide' if label == 0 else 'LongCP'
    ax.scatter(edge_probs[mask], gradient_entropies[mask], s=100, c=color,
               alpha=0.7, edgecolors='black', linewidth=1.5, label=label_name)

ax.set_xlabel('Edge Probability', fontsize=10, weight='bold')
ax.set_ylabel('Gradient Diversity (Entropy)', fontsize=10, weight='bold')
ax.set_title('(d) Gradient Diversity by Structure', fontsize=11, weight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 5: PCA of objectives
ax = axes[1, 1]
# Treat each objective as a feature vector (its conflicts across MDPs)
obj_features = np.array([[conflict_tensor[m, o, :].mean() for m in range(n_mdp)] 
                         for o in range(n_obj)])
pca_obj = PCA(n_components=2)
obj_pca = pca_obj.fit_transform(obj_features)

scatter = ax.scatter(obj_pca[:, 0], obj_pca[:, 1], s=150, c=alphas, cmap='RdYlGn_r',
                     edgecolors='black', linewidth=1.5, alpha=0.8)
ax.set_xlabel(f'PC1 ({pca_obj.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10, weight='bold')
ax.set_ylabel(f'PC2 ({pca_obj.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10, weight='bold')
ax.set_title('(e) PCA of Objective Signatures', fontsize=11, weight='bold')
plt.colorbar(scatter, ax=ax, label='α', fraction=0.046)
ax.grid(True, alpha=0.3)

# Panel 6: Objective clustering dendrogram
ax = axes[1, 2]
from scipy.cluster.hierarchy import dendrogram, linkage
Z_obj = linkage(obj_features, method='ward')
dendro = dendrogram(Z_obj, ax=ax, color_threshold=0, above_threshold_color='#2E86AB')
ax.set_title('(f) Objective Hierarchy', fontsize=11, weight='bold')
ax.set_xlabel('Objective Index', fontsize=10, weight='bold')
ax.set_ylabel('Distance', fontsize=10, weight='bold')

plt.suptitle('Multi-Objective Gradient Analysis', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(output_dir / '25_multiobjective_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: 25_multiobjective_analysis.png")
plt.close()

# ============================================================================
# FIGURE 26: Domain Characterization
# ============================================================================
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.35)

# Panel 1: Radar chart of domain characteristics
ax1 = fig.add_subplot(gs[0, :2], projection='polar')

categories = ['Edge\nProb', 'Task\nCount', 'Task\nLength\n(log)', 'Pareto\nAlign', 'Distance\nto Other']
N = len(categories)

# Normalize features
features_wide = [
    np.mean(edge_probs[clusters == 0]),
    np.mean(task_counts[clusters == 0]) / 50,  # Normalize
    np.mean(np.log10(task_lengths[clusters == 0])) / 5,
    (np.mean(pareto_scores[clusters == 0]) + 1) / 2,  # Scale to [0,1]
    np.mean([distance_matrix[i, clusters == 1].mean() for i in range(n_mdp) if clusters[i] == 0]) / 0.01
]

features_longcp = [
    np.mean(edge_probs[clusters == 1]),
    np.mean(task_counts[clusters == 1]) / 50,
    np.mean(np.log10(task_lengths[clusters == 1])) / 5,
    (np.mean(pareto_scores[clusters == 1]) + 1) / 2,
    np.mean([distance_matrix[i, clusters == 0].mean() for i in range(n_mdp) if clusters[i] == 1]) / 0.01
]

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
features_wide += features_wide[:1]
features_longcp += features_longcp[:1]
angles += angles[:1]

ax1.plot(angles, features_wide, 'o-', linewidth=2.5, color='#32CD32', label='Wide', markersize=8)
ax1.fill(angles, features_wide, alpha=0.25, color='#32CD32')
ax1.plot(angles, features_longcp, 's-', linewidth=2.5, color='#FF6B6B', label='LongCP', markersize=8)
ax1.fill(angles, features_longcp, alpha=0.25, color='#FF6B6B')

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(categories, fontsize=10)
ax1.set_ylim(0, 1)
ax1.set_title('(a) Domain Characteristic Profiles', fontsize=12, weight='bold', pad=20)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax1.grid(True)

# Panel 2: Violin plots
ax2 = fig.add_subplot(gs[0, 2:])

data_to_plot = [
    edge_probs[clusters == 0],
    edge_probs[clusters == 1],
    pareto_scores[clusters == 0],
    pareto_scores[clusters == 1]
]
positions = [1, 2, 4, 5]
colors = ['#32CD32', '#FF6B6B', '#32CD32', '#FF6B6B']

parts = ax2.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.6)

ax2.set_xticks([1.5, 4.5])
ax2.set_xticklabels(['Edge Probability', 'Pareto Alignment'], fontsize=10, weight='bold')
ax2.set_ylabel('Value', fontsize=11, weight='bold')
ax2.set_title('(b) Distribution Comparison', fontsize=12, weight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#32CD32', alpha=0.6, label='Wide'),
                   Patch(facecolor='#FF6B6B', alpha=0.6, label='LongCP')]
ax2.legend(handles=legend_elements, fontsize=10)

# Panel 3: 2D density plot
ax3 = fig.add_subplot(gs[1, 0])

from scipy.stats import gaussian_kde
for label in [0, 1]:
    mask = clusters == label
    if np.sum(mask) > 1:
        xy = np.vstack([edge_probs[mask], pareto_scores[mask]])
        z = gaussian_kde(xy)(xy)
        color = '#32CD32' if label == 0 else '#FF6B6B'
        label_name = 'Wide' if label == 0 else 'LongCP'
        ax3.scatter(edge_probs[mask], pareto_scores[mask], c=z, s=100, 
                   cmap='Greens' if label == 0 else 'Reds', 
                   edgecolors='black', linewidth=1.5, alpha=0.8, label=label_name)

ax3.set_xlabel('Edge Probability', fontsize=10, weight='bold')
ax3.set_ylabel('Pareto Alignment', fontsize=10, weight='bold')
ax3.set_title('(c) Density-Weighted Scatter', fontsize=11, weight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: Parallel coordinates
ax4 = fig.add_subplot(gs[1, 1])

# Normalize all features
features_norm = np.column_stack([
    (edge_probs - edge_probs.min()) / (edge_probs.max() - edge_probs.min()),
    (task_counts - task_counts.min()) / (task_counts.max() - task_counts.min()),
    (pareto_scores - pareto_scores.min()) / (pareto_scores.max() - pareto_scores.min())
])

x = [0, 1, 2]
for i in range(n_mdp):
    color = '#32CD32' if clusters[i] == 0 else '#FF6B6B'
    ax4.plot(x, features_norm[i], color=color, alpha=0.3, linewidth=1.5)

ax4.set_xticks(x)
ax4.set_xticklabels(['Edge\nProb', 'Task\nCount', 'Pareto\nAlign'], fontsize=9)
ax4.set_ylabel('Normalized Value', fontsize=10, weight='bold')
ax4.set_title('(d) Parallel Coordinates', fontsize=11, weight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Panel 5: Centroid comparison
ax5 = fig.add_subplot(gs[1, 2])

metrics = ['Edge\nProb', 'Tasks', 'Length\n(log)', 'Pareto']
wide_vals = [
    np.mean(edge_probs[clusters == 0]),
    np.mean(task_counts[clusters == 0]),
    np.mean(np.log10(task_lengths[clusters == 0])),
    np.mean(pareto_scores[clusters == 0])
]
longcp_vals = [
    np.mean(edge_probs[clusters == 1]),
    np.mean(task_counts[clusters == 1]),
    np.mean(np.log10(task_lengths[clusters == 1])),
    np.mean(pareto_scores[clusters == 1])
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax5.bar(x - width/2, wide_vals, width, label='Wide', color='#32CD32', 
                alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax5.bar(x + width/2, longcp_vals, width, label='LongCP', color='#FF6B6B', 
                alpha=0.7, edgecolor='black', linewidth=1.5)

ax5.set_xticks(x)
ax5.set_xticklabels(metrics, fontsize=9)
ax5.set_ylabel('Mean Value', fontsize=10, weight='bold')
ax5.set_title('(e) Domain Centroids', fontsize=11, weight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Panel 6: Summary statistics
ax6 = fig.add_subplot(gs[1, 3])
ax6.axis('off')

summary = f"""
Domain Characterization Summary

Wide Domain (n=20):
  Edge Prob:    {np.mean(edge_probs[clusters==0]):.3f} ± {np.std(edge_probs[clusters==0]):.3f}
  Task Count:   {np.mean(task_counts[clusters==0]):.1f} ± {np.std(task_counts[clusters==0]):.1f}
  Task Length:  {np.mean(task_lengths[clusters==0]):.0f} ± {np.std(task_lengths[clusters==0]):.0f}
  Pareto Align: {np.mean(pareto_scores[clusters==0]):.3f} ± {np.std(pareto_scores[clusters==0]):.3f}

LongCP Domain (n=20):
  Edge Prob:    {np.mean(edge_probs[clusters==1]):.3f} ± {np.std(edge_probs[clusters==1]):.3f}
  Task Count:   {np.mean(task_counts[clusters==1]):.1f} ± {np.std(task_counts[clusters==1]):.1f}
  Task Length:  {np.mean(task_lengths[clusters==1]):.0f} ± {np.std(task_lengths[clusters==1]):.0f}
  Pareto Align: {np.mean(pareto_scores[clusters==1]):.3f} ± {np.std(pareto_scores[clusters==1]):.3f}

Key Differences:
  ✓ Edge probability
  ✓ Computational depth
  ✓ Objective alignment
"""

ax6.text(0.1, 0.9, summary, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#E8F4F8', alpha=0.9, 
                   edgecolor='#2E86AB', linewidth=2))

plt.suptitle('Comprehensive Domain Characterization', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(output_dir / '26_domain_characterization.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: 26_domain_characterization.png")
plt.close()

print("="*70)
print("✅ Final visualization set created!")
print(f"\nOutput directory: {output_dir}")
print("\nNew figures:")
print("  24_spectral_clustering.png - Complete spectral clustering process")
print("  25_multiobjective_analysis.png - Multi-objective gradient analysis")
print("  26_domain_characterization.png - Comprehensive domain profiles")
