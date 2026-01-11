"""
Extract k-selection subplot from WGDD results
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('logs/wgdd/wgdd_results.json', 'r') as f:
    results = json.load(f)

# Extract k-selection data
k_scores = results['k_scores']
k_values = sorted([int(k) for k in k_scores.keys()])
silhouette_scores = [k_scores[str(k)] for k in k_values]
optimal_k = results['optimal_k']

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# Plot silhouette scores
ax.plot(k_values, silhouette_scores, 'o-', linewidth=2.5, markersize=10, 
        color='#2E86AB', label='Silhouette Score')

# Highlight optimal k
optimal_idx = k_values.index(optimal_k)
ax.plot(optimal_k, silhouette_scores[optimal_idx], 'o', markersize=15, 
        color='#D62828', label=f'Optimal k*={optimal_k}', zorder=10)

# Add vertical line at optimal k
ax.axvline(optimal_k, color='#D62828', linestyle='--', linewidth=1.5, alpha=0.5)

# Styling
ax.set_xlabel('Number of Clusters (k)', fontsize=12, weight='bold')
ax.set_ylabel('Silhouette Score', fontsize=12, weight='bold')
ax.set_title('Automatic Domain Count Selection', fontsize=13, weight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(k_values)
ax.legend(fontsize=10, loc='upper right', framealpha=0.95)

# Add annotation
ax.annotate(f'Best: k={optimal_k}\nScore={silhouette_scores[optimal_idx]:.3f}',
            xy=(optimal_k, silhouette_scores[optimal_idx]),
            xytext=(optimal_k + 0.8, silhouette_scores[optimal_idx] - 0.05),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', edgecolor='#D62828', linewidth=1.5),
            arrowprops=dict(arrowstyle='->', color='#D62828', lw=1.5))

# Set y-axis limits for better visualization
y_min = min(silhouette_scores) - 0.05
y_max = max(silhouette_scores) + 0.05
ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('experiments/gradient_domain_discovery/figures/k_selection.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created: experiments/gradient_domain_discovery/figures/k_selection.png")
plt.close()

# Print summary
print("\nk-Selection Summary:")
print("=" * 40)
for k, score in zip(k_values, silhouette_scores):
    marker = " ← SELECTED" if k == optimal_k else ""
    print(f"k={k}: {score:.4f}{marker}")
