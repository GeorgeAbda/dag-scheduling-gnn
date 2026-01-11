"""
Detailed analysis of the similarity matrix from MO-Gymnasium clustering
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mo_gym_experiment import run_mo_gym_clustering
import os


def analyze_similarity_matrix():
    """Run experiment and analyze similarity matrix in detail"""
    
    print("Running MO-Gymnasium clustering experiment...")
    metrics, similarity_matrix, predicted_labels, true_labels = run_mo_gym_clustering()
    
    domain_names = [
        'dst_seed0', 'dst_seed1', 'dst_seed2',
        'fourroom_seed0', 'fourroom_seed1', 'fourroom_seed2',
        'minecart_seed0', 'minecart_seed1', 'minecart_seed2'
    ]
    
    print("\n" + "=" * 80)
    print("DETAILED SIMILARITY ANALYSIS")
    print("=" * 80)
    
    # 1. Within-group vs between-group similarity
    print("\n[1] WITHIN-GROUP VS BETWEEN-GROUP SIMILARITY")
    print("-" * 80)
    
    groups = {
        'Deep Sea Treasure': [0, 1, 2],
        'Four Room': [3, 4, 5],
        'Minecart': [6, 7, 8]
    }
    
    within_group_sims = []
    between_group_sims = []
    
    for group_name, indices in groups.items():
        # Within-group similarities
        group_sims = []
        for i in indices:
            for j in indices:
                if i != j:
                    group_sims.append(similarity_matrix[i, j])
        
        avg_within = np.mean(group_sims) if group_sims else 0
        within_group_sims.extend(group_sims)
        print(f"\n{group_name}:")
        print(f"  Within-group avg similarity: {avg_within:.4f}")
        print(f"  Within-group similarities: {[f'{s:.3f}' for s in group_sims]}")
    
    # Between-group similarities
    print("\nBetween-group similarities:")
    for g1_name, g1_indices in groups.items():
        for g2_name, g2_indices in groups.items():
            if g1_name < g2_name:
                cross_sims = []
                for i in g1_indices:
                    for j in g2_indices:
                        cross_sims.append(similarity_matrix[i, j])
                        between_group_sims.append(similarity_matrix[i, j])
                
                avg_cross = np.mean(cross_sims)
                print(f"  {g1_name} <-> {g2_name}: {avg_cross:.4f}")
    
    print(f"\nOverall within-group avg: {np.mean(within_group_sims):.4f}")
    print(f"Overall between-group avg: {np.mean(between_group_sims):.4f}")
    print(f"Separation ratio: {np.mean(within_group_sims) / (np.mean(between_group_sims) + 1e-8):.2f}x")
    
    # 2. Pairwise similarity table
    print("\n[2] FULL SIMILARITY MATRIX")
    print("-" * 80)
    
    # Print as table
    header = "            " + "  ".join([f"{n[:8]:>8}" for n in domain_names])
    print(header)
    for i, name_i in enumerate(domain_names):
        row = f"{name_i[:10]:>10}  "
        for j in range(len(domain_names)):
            val = similarity_matrix[i, j]
            if i == j:
                row += f"{'1.000':>8}  "
            elif true_labels[i] == true_labels[j]:
                row += f"{val:>8.3f}* "  # Same group
            else:
                row += f"{val:>8.3f}  "
        print(row)
    print("(* = same true group)")
    
    # 3. Misclassification analysis
    print("\n[3] MISCLASSIFICATION ANALYSIS")
    print("-" * 80)
    
    for i, (name, true_label, pred_label) in enumerate(zip(domain_names, true_labels, predicted_labels)):
        if true_label != pred_label:
            # Find which domains it's most similar to
            sims = similarity_matrix[i, :]
            sorted_indices = np.argsort(sims)[::-1]
            
            print(f"\n{name} (true={true_label}, pred={pred_label}):")
            print(f"  Most similar domains:")
            for idx in sorted_indices[1:4]:  # Top 3 (excluding self)
                print(f"    {domain_names[idx]}: {sims[idx]:.4f} (group {true_labels[idx]})")
    
    # 4. Create detailed visualization
    print("\n[4] GENERATING DETAILED VISUALIZATION")
    print("-" * 80)
    
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Similarity matrix with group annotations
    ax1 = fig.add_subplot(2, 3, 1)
    
    # Reorder by true groups
    order = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Already in group order
    reordered_sim = similarity_matrix[np.ix_(order, order)]
    reordered_names = [domain_names[i] for i in order]
    
    sns.heatmap(
        reordered_sim,
        xticklabels=[n.replace('_seed', '\n') for n in reordered_names],
        yticklabels=[n.replace('_seed', '\n') for n in reordered_names],
        cmap='RdBu_r',
        center=0,
        ax=ax1,
        cbar_kws={'label': 'Cosine Similarity'},
        vmin=-0.2,
        vmax=1.0
    )
    ax1.set_title('Similarity Matrix\n(ordered by true groups)', fontsize=12, fontweight='bold')
    
    # Add group boundaries
    ax1.axhline(y=3, color='black', linewidth=2)
    ax1.axhline(y=6, color='black', linewidth=2)
    ax1.axvline(x=3, color='black', linewidth=2)
    ax1.axvline(x=6, color='black', linewidth=2)
    
    # Plot 2: Within vs between group distribution
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(within_group_sims, bins=15, alpha=0.7, label='Within-group', color='green', edgecolor='black')
    ax2.hist(between_group_sims, bins=15, alpha=0.7, label='Between-group', color='red', edgecolor='black')
    ax2.axvline(np.mean(within_group_sims), color='green', linestyle='--', linewidth=2, label=f'Within avg: {np.mean(within_group_sims):.3f}')
    ax2.axvline(np.mean(between_group_sims), color='red', linestyle='--', linewidth=2, label=f'Between avg: {np.mean(between_group_sims):.3f}')
    ax2.set_xlabel('Cosine Similarity', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Within vs Between Group\nSimilarity Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Group-level similarity heatmap
    ax3 = fig.add_subplot(2, 3, 3)
    group_names = list(groups.keys())
    group_sim_matrix = np.zeros((3, 3))
    
    for i, (g1_name, g1_indices) in enumerate(groups.items()):
        for j, (g2_name, g2_indices) in enumerate(groups.items()):
            sims = []
            for idx1 in g1_indices:
                for idx2 in g2_indices:
                    if idx1 != idx2:
                        sims.append(similarity_matrix[idx1, idx2])
            group_sim_matrix[i, j] = np.mean(sims) if sims else 1.0
    
    sns.heatmap(
        group_sim_matrix,
        xticklabels=group_names,
        yticklabels=group_names,
        cmap='RdBu_r',
        center=0,
        ax=ax3,
        annot=True,
        fmt='.3f',
        cbar_kws={'label': 'Avg Similarity'}
    )
    ax3.set_title('Group-Level Similarity', fontsize=12, fontweight='bold')
    
    # Plot 4: Clustering result comparison
    ax4 = fig.add_subplot(2, 3, 4)
    x = np.arange(len(domain_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, true_labels, width, label='True', alpha=0.8, color='blue', edgecolor='black')
    bars2 = ax4.bar(x + width/2, predicted_labels, width, label='Predicted', alpha=0.8, color='orange', edgecolor='black')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels([n.replace('_seed', '\n') for n in domain_names], fontsize=8)
    ax4.set_ylabel('Cluster ID', fontsize=11)
    ax4.set_title('True vs Predicted Clusters', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Highlight mismatches
    for i, (t, p) in enumerate(zip(true_labels, predicted_labels)):
        if t != p:
            ax4.annotate('✗', (i, max(t, p) + 0.2), ha='center', fontsize=14, color='red')
    
    # Plot 5: Metrics summary
    ax5 = fig.add_subplot(2, 3, 5)
    metric_names = ['ARI', 'NMI', 'Silhouette']
    metric_values = [
        metrics['adjusted_rand_index'],
        metrics['normalized_mutual_info'],
        metrics.get('silhouette_score', 0)
    ]
    colors = ['green' if v > 0.5 else 'orange' if v > 0.2 else 'red' for v in metric_values]
    
    bars = ax5.barh(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
    ax5.set_xlim(-0.1, 1.0)
    ax5.axvline(x=0.5, color='green', linestyle='--', alpha=0.5)
    ax5.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax5.set_title('Clustering Quality Metrics', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars, metric_values):
        ax5.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontweight='bold')
    
    # Plot 6: Environment characteristics
    ax6 = fig.add_subplot(2, 3, 6)
    env_info = {
        'Deep Sea Treasure': {'obs_dim': 2, 'n_actions': 4, 'type': 'Grid Navigation'},
        'Four Room': {'obs_dim': 14, 'n_actions': 4, 'type': 'Multi-goal Navigation'},
        'Minecart': {'obs_dim': 7, 'n_actions': 6, 'type': 'Resource Collection'}
    }
    
    table_data = []
    for env_name, info in env_info.items():
        table_data.append([env_name, info['obs_dim'], info['n_actions'], info['type']])
    
    ax6.axis('off')
    table = ax6.table(
        cellText=table_data,
        colLabels=['Environment', 'Obs Dim', 'Actions', 'Type'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax6.set_title('Environment Characteristics', fontsize=12, fontweight='bold', y=0.8)
    
    plt.tight_layout()
    plt.savefig('results/mo_gym_detailed_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("  Saved to results/mo_gym_detailed_analysis.png")
    
    # 5. Summary statistics
    print("\n[5] SUMMARY STATISTICS")
    print("-" * 80)
    
    print(f"\nClustering Performance:")
    print(f"  Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
    print(f"  Normalized Mutual Info: {metrics['normalized_mutual_info']:.4f}")
    print(f"  Silhouette Score: {metrics.get('silhouette_score', 'N/A')}")
    
    print(f"\nSimilarity Statistics:")
    print(f"  Within-group mean: {np.mean(within_group_sims):.4f}")
    print(f"  Within-group std: {np.std(within_group_sims):.4f}")
    print(f"  Between-group mean: {np.mean(between_group_sims):.4f}")
    print(f"  Between-group std: {np.std(between_group_sims):.4f}")
    
    # Separation quality
    sep_ratio = np.mean(within_group_sims) / (np.mean(between_group_sims) + 1e-8)
    print(f"\nSeparation Quality:")
    print(f"  Within/Between ratio: {sep_ratio:.2f}x")
    if sep_ratio > 2:
        print("  → Good separation (ratio > 2)")
    elif sep_ratio > 1.5:
        print("  → Moderate separation (1.5 < ratio < 2)")
    else:
        print("  → Weak separation (ratio < 1.5)")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return metrics, similarity_matrix


if __name__ == '__main__':
    analyze_similarity_matrix()
