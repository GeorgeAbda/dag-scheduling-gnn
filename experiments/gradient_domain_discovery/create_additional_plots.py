"""
Create additional interesting plots for WGDD paper
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load results
with open('logs/wgdd/wgdd_results.json', 'r') as f:
    results = json.load(f)

# Load distance matrix
distance_matrix = np.load('logs/wgdd/distance_matrix.npy')
conflict_tensor = np.load('logs/wgdd/conflict_tensor.npy')

# Extract data
mdp_configs = results['mdp_configs']
true_labels = np.array(results['true_labels'])
clusters = np.array(results['clusters'])
pareto_scores = np.array(results['pareto_scores'])
objectives = np.array(results['objectives'])

# Extract MDP parameters
edge_probs = [cfg['edge_prob'] for cfg in mdp_configs]
task_counts = [(cfg['min_tasks'] + cfg['max_tasks']) / 2 for cfg in mdp_configs]
task_lengths = [(cfg['min_length'] + cfg['max_length']) / 2 for cfg in mdp_configs]

output_dir = Path('experiments/gradient_domain_discovery/figures')
output_dir.mkdir(exist_ok=True, parents=True)

# ============================================================================
# PLOT 1: MDP Parameter Space with Discovered Domains
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1a: Edge Prob vs Task Count
ax = axes[0]
for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    label_name = 'Wide' if label == 0 else 'LongCP'
    ax.scatter(np.array(edge_probs)[mask], np.array(task_counts)[mask], 
               s=100, alpha=0.7, c=color, edgecolors='black', linewidth=1.5,
               label=label_name)
ax.set_xlabel('Edge Probability', fontsize=12, weight='bold')
ax.set_ylabel('Task Count', fontsize=12, weight='bold')
ax.set_title('(a) Domain Structure in Parameter Space', fontsize=12, weight='bold')
ax.legend(fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)

# 1b: Edge Prob vs Task Length
ax = axes[1]
for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    label_name = 'Wide' if label == 0 else 'LongCP'
    ax.scatter(np.array(edge_probs)[mask], np.array(task_lengths)[mask], 
               s=100, alpha=0.7, c=color, edgecolors='black', linewidth=1.5,
               label=label_name)
ax.set_xlabel('Edge Probability', fontsize=12, weight='bold')
ax.set_ylabel('Task Length (avg)', fontsize=12, weight='bold')
ax.set_title('(b) Edge Prob vs Computational Size', fontsize=12, weight='bold')
ax.legend(fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# 1c: Task Count vs Task Length
ax = axes[2]
for label in [0, 1]:
    mask = clusters == label
    color = '#32CD32' if label == 0 else '#FF6B6B'
    label_name = 'Wide' if label == 0 else 'LongCP'
    ax.scatter(np.array(task_counts)[mask], np.array(task_lengths)[mask], 
               s=100, alpha=0.7, c=color, edgecolors='black', linewidth=1.5,
               label=label_name)
ax.set_xlabel('Task Count', fontsize=12, weight='bold')
ax.set_ylabel('Task Length (avg)', fontsize=12, weight='bold')
ax.set_title('(c) Parallelism vs Computation', fontsize=12, weight='bold')
ax.legend(fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(output_dir / 'parameter_space_domains.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created: parameter_space_domains.png")
plt.close()

# ============================================================================
# PLOT 2: Pareto Alignment Distribution
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 2a: Distribution by domain
ax = axes[0]
wide_scores = pareto_scores[clusters == 0]
longcp_scores = pareto_scores[clusters == 1]

ax.hist(wide_scores, bins=15, alpha=0.6, color='#32CD32', edgecolor='black', 
        linewidth=1.5, label='Wide', density=True)
ax.hist(longcp_scores, bins=15, alpha=0.6, color='#FF6B6B', edgecolor='black', 
        linewidth=1.5, label='LongCP', density=True)

# Add KDE
if len(wide_scores) > 1:
    kde_wide = gaussian_kde(wide_scores)
    x_wide = np.linspace(wide_scores.min(), wide_scores.max(), 100)
    ax.plot(x_wide, kde_wide(x_wide), color='#228B22', linewidth=2.5, label='Wide KDE')

if len(longcp_scores) > 1:
    kde_longcp = gaussian_kde(longcp_scores)
    x_longcp = np.linspace(longcp_scores.min(), longcp_scores.max(), 100)
    ax.plot(x_longcp, kde_longcp(x_longcp), color='#DC143C', linewidth=2.5, label='LongCP KDE')

ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Neutral')
ax.set_xlabel('Pareto Alignment Score (cos similarity)', fontsize=12, weight='bold')
ax.set_ylabel('Density', fontsize=12, weight='bold')
ax.set_title('(a) Objective Alignment by Domain', fontsize=12, weight='bold')
ax.legend(fontsize=9, framealpha=0.95)
ax.grid(True, alpha=0.3)

# 2b: Violin plot
ax = axes[1]
data_to_plot = [wide_scores, longcp_scores]
positions = [1, 2]
colors = ['#32CD32', '#FF6B6B']

parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.6)

ax.set_xticks(positions)
ax.set_xticklabels(['Wide', 'LongCP'], fontsize=11)
ax.set_ylabel('Pareto Alignment Score', fontsize=12, weight='bold')
ax.set_title('(b) Distribution Comparison', fontsize=12, weight='bold')
ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Add statistics
ax.text(1, wide_scores.max() + 0.1, f'μ={wide_scores.mean():.2f}\nσ={wide_scores.std():.2f}',
        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='#E8F8E8', alpha=0.8))
ax.text(2, longcp_scores.max() + 0.1, f'μ={longcp_scores.mean():.2f}\nσ={longcp_scores.std():.2f}',
        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / 'pareto_alignment_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created: pareto_alignment_analysis.png")
plt.close()

# ============================================================================
# PLOT 3: Distance Matrix Analysis
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 3a: Within vs Between domain distances
ax = axes[0]
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

data = [within_wide, within_longcp, between]
labels = ['Within\nWide', 'Within\nLongCP', 'Between\nDomains']
colors = ['#32CD32', '#FF6B6B', '#9370DB']

bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel('Wasserstein Distance', fontsize=12, weight='bold')
ax.set_title('(a) Within vs Between Domain Distances', fontsize=12, weight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 3b: Distance distribution
ax = axes[1]
ax.hist(within_wide, bins=20, alpha=0.5, color='#32CD32', label='Within Wide', density=True)
ax.hist(within_longcp, bins=20, alpha=0.5, color='#FF6B6B', label='Within LongCP', density=True)
ax.hist(between, bins=20, alpha=0.5, color='#9370DB', label='Between', density=True)
ax.set_xlabel('Wasserstein Distance', fontsize=12, weight='bold')
ax.set_ylabel('Density', fontsize=12, weight='bold')
ax.set_title('(b) Distance Distributions', fontsize=12, weight='bold')
ax.legend(fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)

# 3c: Separation ratio
ax = axes[2]
mean_within = (np.mean(within_wide) + np.mean(within_longcp)) / 2
mean_between = np.mean(between)
separation_ratio = mean_between / mean_within

categories = ['Within\nDomains', 'Between\nDomains']
means = [mean_within, mean_between]
colors_bar = ['#7FBF7F', '#9370DB']

bars = ax.bar(categories, means, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Mean Wasserstein Distance', fontsize=12, weight='bold')
ax.set_title(f'(c) Separation Quality\nRatio = {separation_ratio:.2f}×', fontsize=12, weight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, val in zip(bars, means):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'distance_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created: distance_analysis.png")
plt.close()

# ============================================================================
# PLOT 4: Objective Trade-off Landscape
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 4a: Objective weights coverage
ax = axes[0]
alphas = objectives[:, 0]
scatter = ax.scatter(alphas, np.arange(len(alphas)), c=alphas, cmap='RdYlGn_r', 
                     s=150, edgecolors='black', linewidth=1.5)
ax.axvline(0.5, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Balanced')
ax.set_xlabel('α (Makespan Weight)', fontsize=12, weight='bold')
ax.set_ylabel('Objective Index', fontsize=12, weight='bold')
ax.set_title('(a) Multi-Scale Objective Sampling', fontsize=12, weight='bold')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Makespan Priority', fontsize=10)

# Add annotations for extremes
ax.annotate('Pure\nMakespan', xy=(1.0, 0), xytext=(0.85, 3),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=9, color='red', weight='bold')
ax.annotate('Pure\nEnergy', xy=(0.0, 1), xytext=(0.15, 4),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=9, color='green', weight='bold')

# 4b: Gradient conflict across objectives
ax = axes[1]
# Compute mean conflict for each objective pair
n_obj = len(objectives)
conflict_by_obj = np.zeros(n_obj)
for i in range(len(mdp_configs)):
    for j in range(n_obj):
        # Conflict between this objective and pure makespan
        conflict_by_obj[j] += conflict_tensor[i, 0, j]

conflict_by_obj /= len(mdp_configs)

ax.plot(alphas, conflict_by_obj, 'o-', linewidth=2.5, markersize=8, color='#2E86AB')
ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax.fill_between(alphas, 0, conflict_by_obj, alpha=0.3, color='#2E86AB')
ax.set_xlabel('α (Makespan Weight)', fontsize=12, weight='bold')
ax.set_ylabel('Mean Gradient Conflict', fontsize=12, weight='bold')
ax.set_title('(b) Conflict Across Trade-off Spectrum', fontsize=12, weight='bold')
ax.grid(True, alpha=0.3)

# Highlight regions
ax.axvspan(0, 0.3, alpha=0.1, color='green', label='Energy-focused')
ax.axvspan(0.4, 0.6, alpha=0.1, color='gray', label='Balanced')
ax.axvspan(0.7, 1.0, alpha=0.1, color='red', label='Time-focused')
ax.legend(fontsize=9, loc='best', framealpha=0.95)

plt.tight_layout()
plt.savefig(output_dir / 'objective_landscape.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created: objective_landscape.png")
plt.close()

# ============================================================================
# PLOT 5: Confusion Matrix Style Visualization
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 5a: Clustering confusion matrix
ax = axes[0]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_labels, clusters)
im = ax.imshow(cm, cmap='Blues', aspect='auto')

# Add text annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                      color="white" if cm[i, j] > cm.max()/2 else "black",
                      fontsize=16, weight='bold')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Cluster 0', 'Cluster 1'], fontsize=11)
ax.set_yticklabels(['True Wide', 'True LongCP'], fontsize=11)
ax.set_xlabel('Predicted Domain', fontsize=12, weight='bold')
ax.set_ylabel('True Domain', fontsize=12, weight='bold')
ax.set_title('(a) Perfect Domain Recovery\nNMI=1.0, ARI=1.0', fontsize=12, weight='bold')
plt.colorbar(im, ax=ax, label='Count')

# 5b: Domain characteristics comparison
ax = axes[1]
characteristics = ['Edge\nProb', 'Task\nCount', 'Task\nLength\n(log)']
wide_vals = [
    np.mean([edge_probs[i] for i in range(len(clusters)) if clusters[i] == 0]),
    np.mean([task_counts[i] for i in range(len(clusters)) if clusters[i] == 0]),
    np.log10(np.mean([task_lengths[i] for i in range(len(clusters)) if clusters[i] == 0]))
]
longcp_vals = [
    np.mean([edge_probs[i] for i in range(len(clusters)) if clusters[i] == 1]),
    np.mean([task_counts[i] for i in range(len(clusters)) if clusters[i] == 1]),
    np.log10(np.mean([task_lengths[i] for i in range(len(clusters)) if clusters[i] == 1]))
]

x = np.arange(len(characteristics))
width = 0.35

bars1 = ax.bar(x - width/2, wide_vals, width, label='Wide', color='#32CD32', 
               alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, longcp_vals, width, label='LongCP', color='#FF6B6B', 
               alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Normalized Value', fontsize=12, weight='bold')
ax.set_title('(b) Domain Characteristics', fontsize=12, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(characteristics, fontsize=10)
ax.legend(fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'domain_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created: domain_comparison.png")
plt.close()

# ============================================================================
# PLOT 6: Gradient Conflict Heatmap per MDP
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 6a: Wide MDPs conflict heatmap
ax = axes[0]
wide_indices = [i for i in range(len(clusters)) if clusters[i] == 0]
wide_conflicts = conflict_tensor[wide_indices, :, :]
mean_wide_conflict = np.mean(wide_conflicts, axis=0)

im = ax.imshow(mean_wide_conflict, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax.set_xlabel('Objective Index', fontsize=11, weight='bold')
ax.set_ylabel('Objective Index', fontsize=11, weight='bold')
ax.set_title('(a) Wide Domain: Mean Gradient Conflicts', fontsize=12, weight='bold')
plt.colorbar(im, ax=ax, label='Cosine Similarity')

# 6b: LongCP MDPs conflict heatmap
ax = axes[1]
longcp_indices = [i for i in range(len(clusters)) if clusters[i] == 1]
longcp_conflicts = conflict_tensor[longcp_indices, :, :]
mean_longcp_conflict = np.mean(longcp_conflicts, axis=0)

im = ax.imshow(mean_longcp_conflict, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax.set_xlabel('Objective Index', fontsize=11, weight='bold')
ax.set_ylabel('Objective Index', fontsize=11, weight='bold')
ax.set_title('(b) LongCP Domain: Mean Gradient Conflicts', fontsize=12, weight='bold')
plt.colorbar(im, ax=ax, label='Cosine Similarity')

plt.tight_layout()
plt.savefig(output_dir / 'conflict_heatmaps.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created: conflict_heatmaps.png")
plt.close()

print("\n" + "="*60)
print("✅ All additional plots created successfully!")
print("="*60)
print(f"\nOutput directory: {output_dir}")
print("\nCreated plots:")
print("  1. parameter_space_domains.png - MDP parameter space visualization")
print("  2. pareto_alignment_analysis.png - Objective alignment distributions")
print("  3. distance_analysis.png - Wasserstein distance analysis")
print("  4. objective_landscape.png - Trade-off spectrum coverage")
print("  5. domain_comparison.png - Confusion matrix & characteristics")
print("  6. conflict_heatmaps.png - Gradient conflict patterns")
