"""
Analyze the discovered strategy domains in detail.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open('logs/strategy_domains/strategy_domains.json', 'r') as f:
    results = json.load(f)

mdp_configs = results['mdp_configs']
conflicts = np.array(results['conflicts'])
clusters = np.array(results['clusters'])

print("="*70)
print("DETAILED STRATEGY DOMAIN ANALYSIS")
print("="*70)

# Analyze each MDP
print("\n## MDP-by-MDP Conflict Analysis\n")
print(f"{'MDP':<5} {'Edge Prob':<12} {'Conflict':<12} {'Cluster':<10} {'Interpretation'}")
print("-" * 70)

for i, (cfg, conflict, cluster) in enumerate(zip(mdp_configs, conflicts, clusters)):
    edge_prob = cfg['edge_prob']
    
    if conflict < -0.3:
        interp = "SEVERE CONFLICT"
    elif conflict < -0.1:
        interp = "Moderate conflict"
    elif conflict < 0.1:
        interp = "Weak/no conflict"
    else:
        interp = "Aligned objectives"
    
    print(f"{i+1:<5} {edge_prob:<12.3f} {conflict:<12.4f} {cluster:<10} {interp}")

# Analyze clusters
print("\n" + "="*70)
print("CLUSTER ANALYSIS")
print("="*70)

for cluster_id in [0, 1]:
    cluster_mask = clusters == cluster_id
    cluster_mdps = np.where(cluster_mask)[0]
    cluster_conflicts = conflicts[cluster_mask]
    cluster_edge_probs = [mdp_configs[i]['edge_prob'] for i in cluster_mdps]
    
    print(f"\n## Cluster {cluster_id} ({len(cluster_mdps)} MDPs)")
    print(f"  Edge probability: {np.mean(cluster_edge_probs):.3f} ± {np.std(cluster_edge_probs):.3f}")
    print(f"  Conflict: {np.mean(cluster_conflicts):.4f} ± {np.std(cluster_conflicts):.4f}")
    print(f"  Range: [{np.min(cluster_conflicts):.4f}, {np.max(cluster_conflicts):.4f}]")
    print(f"  MDPs: {list(cluster_mdps + 1)}")

# Correlation analysis
print("\n" + "="*70)
print("CORRELATION ANALYSIS")
print("="*70)

edge_probs = [cfg['edge_prob'] for cfg in mdp_configs]
correlation = np.corrcoef(edge_probs, conflicts)[0, 1]

print(f"\nCorrelation between edge_prob and conflict: {correlation:.4f}")

if abs(correlation) < 0.3:
    print("  → WEAK correlation: Edge probability doesn't strongly predict conflict")
    print("  → Other factors (task count, task length) may be important")
elif correlation < -0.3:
    print("  → NEGATIVE correlation: Higher edge_prob → more conflict")
    print("  → Long CP MDPs have opposing makespan/energy objectives")
else:
    print("  → POSITIVE correlation: Higher edge_prob → less conflict")
    print("  → Long CP MDPs have aligned makespan/energy objectives")

# Find most conflicting MDPs
print("\n" + "="*70)
print("MOST CONFLICTING MDPs")
print("="*70)

most_conflicting_idx = np.argmin(conflicts)
least_conflicting_idx = np.argmax(conflicts)

print(f"\nMost conflict (MDP {most_conflicting_idx + 1}):")
print(f"  Conflict: {conflicts[most_conflicting_idx]:.4f}")
print(f"  Edge prob: {mdp_configs[most_conflicting_idx]['edge_prob']:.3f}")
print(f"  Tasks: {mdp_configs[most_conflicting_idx]['min_tasks']}-{mdp_configs[most_conflicting_idx]['max_tasks']}")
print(f"  Length: {mdp_configs[most_conflicting_idx]['min_length']}-{mdp_configs[most_conflicting_idx]['max_length']}")

print(f"\nLeast conflict (MDP {least_conflicting_idx + 1}):")
print(f"  Conflict: {conflicts[least_conflicting_idx]:.4f}")
print(f"  Edge prob: {mdp_configs[least_conflicting_idx]['edge_prob']:.3f}")
print(f"  Tasks: {mdp_configs[least_conflicting_idx]['min_tasks']}-{mdp_configs[least_conflicting_idx]['max_tasks']}")
print(f"  Length: {mdp_configs[least_conflicting_idx]['min_length']}-{mdp_configs[least_conflicting_idx]['max_length']}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Conflict vs edge probability
ax = axes[0, 0]
colors = ['red' if c == 0 else 'blue' for c in clusters]
ax.scatter(edge_probs, conflicts, c=colors, alpha=0.6, s=100)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Edge Probability', fontsize=12)
ax.set_ylabel('Gradient Conflict (cos_sim)', fontsize=12)
ax.set_title('Gradient Conflict vs MDP Structure', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add annotations for extreme points
ax.annotate(f'Most conflict\n({most_conflicting_idx+1})', 
            xy=(edge_probs[most_conflicting_idx], conflicts[most_conflicting_idx]),
            xytext=(10, -20), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Plot 2: Histogram of conflicts
ax = axes[0, 1]
ax.hist(conflicts, bins=15, alpha=0.7, color='green', edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No conflict')
ax.set_xlabel('Gradient Conflict', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Gradient Conflicts', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Edge probability distribution by cluster
ax = axes[1, 0]
cluster_0_edge_probs = [edge_probs[i] for i in range(len(edge_probs)) if clusters[i] == 0]
cluster_1_edge_probs = [edge_probs[i] for i in range(len(edge_probs)) if clusters[i] == 1]
ax.hist(cluster_0_edge_probs, bins=10, alpha=0.5, label='Cluster 0', color='red')
ax.hist(cluster_1_edge_probs, bins=10, alpha=0.5, label='Cluster 1', color='blue')
ax.set_xlabel('Edge Probability', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Cluster Distribution by Edge Probability', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Conflict by cluster
ax = axes[1, 1]
cluster_0_conflicts = conflicts[clusters == 0]
cluster_1_conflicts = conflicts[clusters == 1]
ax.boxplot([cluster_0_conflicts, cluster_1_conflicts], labels=['Cluster 0', 'Cluster 1'])
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.set_ylabel('Gradient Conflict', fontsize=12)
ax.set_title('Conflict Distribution by Cluster', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('logs/strategy_domains/detailed_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved detailed analysis plot to logs/strategy_domains/detailed_analysis.png")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nKey findings:")
print(f"  1. Conflict range: [{conflicts.min():.4f}, {conflicts.max():.4f}]")
print(f"  2. {np.sum(conflicts < -0.1)} MDPs have moderate-to-severe conflict")
print(f"  3. {np.sum(np.abs(conflicts) < 0.1)} MDPs have weak/no conflict")
print(f"  4. {np.sum(conflicts > 0.1)} MDPs have aligned objectives")
print(f"  5. Correlation with edge_prob: {correlation:.4f}")
