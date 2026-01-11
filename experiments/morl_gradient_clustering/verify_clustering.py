"""
Verification script to analyze clustering results against ground truth
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gradient_domain_clustering import GradientDomainClustering, ClusteringConfig
from mo_gymnasium_loader import create_synthetic_mo_domains, get_domain_info


def analyze_clustering_results(
    domains,
    true_labels,
    predicted_labels,
    domain_info
):
    """
    Detailed analysis of clustering results
    """
    print("=" * 80)
    print("CLUSTERING VERIFICATION AND ANALYSIS")
    print("=" * 80)
    
    # 1. Domain characteristics by true cluster
    print("\n[1] TRUE CLUSTER STRUCTURE (Ground Truth)")
    print("-" * 80)
    
    for true_cluster_id in np.unique(true_labels):
        domain_indices = np.where(true_labels == true_cluster_id)[0]
        print(f"\nTrue Cluster {true_cluster_id}:")
        
        weights_list = []
        for idx in domain_indices:
            domain_name = f'domain_{idx}'
            info = domain_info[domain_name]
            weights = info['reward_weights']
            weights_list.append(weights)
            print(f"  {domain_name}: weights=[{weights[0]:.3f}, {weights[1]:.3f}]")
        
        # Average weights for this cluster
        avg_weights = np.mean(weights_list, axis=0)
        print(f"  → Average: [{avg_weights[0]:.3f}, {avg_weights[1]:.3f}]")
        
        if avg_weights[0] > 0.7:
            print(f"  → Interpretation: Strongly favors Objective 1")
        elif avg_weights[0] < 0.3:
            print(f"  → Interpretation: Strongly favors Objective 2")
        else:
            print(f"  → Interpretation: Balanced objectives")
    
    # 2. Predicted cluster structure
    print("\n[2] PREDICTED CLUSTER STRUCTURE (From Gradient Clustering)")
    print("-" * 80)
    
    for pred_cluster_id in np.unique(predicted_labels):
        domain_indices = np.where(predicted_labels == pred_cluster_id)[0]
        print(f"\nPredicted Cluster {pred_cluster_id}:")
        
        weights_list = []
        for idx in domain_indices:
            domain_name = f'domain_{idx}'
            info = domain_info[domain_name]
            weights = info['reward_weights']
            weights_list.append(weights)
            true_cluster = true_labels[idx]
            print(f"  {domain_name}: weights=[{weights[0]:.3f}, {weights[1]:.3f}] (true cluster: {true_cluster})")
        
        # Average weights for this cluster
        avg_weights = np.mean(weights_list, axis=0)
        print(f"  → Average: [{avg_weights[0]:.3f}, {avg_weights[1]:.3f}]")
    
    # 3. Confusion matrix
    print("\n[3] CONFUSION MATRIX")
    print("-" * 80)
    
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(conf_matrix)
    
    # 4. Detailed metrics
    print("\n[4] CLUSTERING QUALITY METRICS")
    print("-" * 80)
    
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    
    print(f"\nAdjusted Rand Index (ARI): {ari:.4f}")
    print("  Interpretation:")
    if ari > 0.8:
        print("    ✓ Excellent clustering (>0.8)")
    elif ari > 0.6:
        print("    ✓ Good clustering (0.6-0.8)")
    elif ari > 0.4:
        print("    ~ Moderate clustering (0.4-0.6)")
    elif ari > 0.2:
        print("    ⚠ Poor clustering (0.2-0.4)")
    else:
        print("    ✗ Very poor clustering (<0.2)")
    
    print(f"\nNormalized Mutual Information (NMI): {nmi:.4f}")
    print("  Interpretation:")
    if nmi > 0.7:
        print("    ✓ Strong information sharing (>0.7)")
    elif nmi > 0.5:
        print("    ~ Moderate information sharing (0.5-0.7)")
    else:
        print("    ⚠ Weak information sharing (<0.5)")
    
    # 5. Misclassification analysis
    print("\n[5] MISCLASSIFICATION ANALYSIS")
    print("-" * 80)
    
    misclassified = []
    for idx in range(len(true_labels)):
        domain_name = f'domain_{idx}'
        info = domain_info[domain_name]
        weights = info['reward_weights']
        
        # Find most common predicted cluster for this true cluster
        true_cluster = true_labels[idx]
        same_true_cluster = np.where(true_labels == true_cluster)[0]
        predicted_for_true = predicted_labels[same_true_cluster]
        most_common_pred = np.bincount(predicted_for_true).argmax()
        
        if predicted_labels[idx] != most_common_pred:
            misclassified.append({
                'domain': domain_name,
                'true_cluster': true_cluster,
                'predicted_cluster': predicted_labels[idx],
                'expected_cluster': most_common_pred,
                'weights': weights
            })
    
    if len(misclassified) > 0:
        print(f"\nFound {len(misclassified)} potential misclassifications:")
        for item in misclassified:
            print(f"  {item['domain']}: true={item['true_cluster']}, "
                  f"predicted={item['predicted_cluster']}, "
                  f"expected={item['expected_cluster']}, "
                  f"weights=[{item['weights'][0]:.3f}, {item['weights'][1]:.3f}]")
    else:
        print("\nNo obvious misclassifications detected.")
    
    # 6. Cluster purity
    print("\n[6] CLUSTER PURITY ANALYSIS")
    print("-" * 80)
    
    for pred_cluster_id in np.unique(predicted_labels):
        domain_indices = np.where(predicted_labels == pred_cluster_id)[0]
        true_clusters_in_pred = true_labels[domain_indices]
        
        # Calculate purity
        most_common_true = np.bincount(true_clusters_in_pred).argmax()
        purity = np.sum(true_clusters_in_pred == most_common_true) / len(true_clusters_in_pred)
        
        print(f"\nPredicted Cluster {pred_cluster_id}:")
        print(f"  Size: {len(domain_indices)} domains")
        print(f"  True cluster distribution: {np.bincount(true_clusters_in_pred)}")
        print(f"  Dominant true cluster: {most_common_true}")
        print(f"  Purity: {purity:.2%}")
    
    # 7. Visualization
    print("\n[7] GENERATING VISUALIZATION")
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Reward weights scatter
    ax = axes[0, 0]
    for true_cluster_id in np.unique(true_labels):
        domain_indices = np.where(true_labels == true_cluster_id)[0]
        weights = np.array([domain_info[f'domain_{idx}']['reward_weights'] for idx in domain_indices])
        ax.scatter(weights[:, 0], weights[:, 1], 
                  s=200, alpha=0.7, 
                  label=f'True Cluster {true_cluster_id}',
                  marker='o', edgecolors='black', linewidths=2)
    
    ax.set_xlabel('Weight for Objective 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight for Objective 2', fontsize=12, fontweight='bold')
    ax.set_title('Ground Truth: Reward Weight Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    
    # Plot 2: Predicted clusters
    ax = axes[0, 1]
    for pred_cluster_id in np.unique(predicted_labels):
        domain_indices = np.where(predicted_labels == pred_cluster_id)[0]
        weights = np.array([domain_info[f'domain_{idx}']['reward_weights'] for idx in domain_indices])
        ax.scatter(weights[:, 0], weights[:, 1], 
                  s=200, alpha=0.7, 
                  label=f'Predicted Cluster {pred_cluster_id}',
                  marker='s', edgecolors='black', linewidths=2)
    
    ax.set_xlabel('Weight for Objective 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight for Objective 2', fontsize=12, fontweight='bold')
    ax.set_title('Predicted: Gradient-Based Clustering', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    
    # Plot 3: Confusion matrix
    ax = axes[1, 0]
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Cluster', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Plot 4: Domain-by-domain comparison
    ax = axes[1, 1]
    x_pos = np.arange(len(true_labels))
    width = 0.35
    
    ax.bar(x_pos - width/2, true_labels, width, label='True Labels', alpha=0.8)
    ax.bar(x_pos + width/2, predicted_labels, width, label='Predicted Labels', alpha=0.8)
    
    ax.set_xlabel('Domain Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cluster ID', fontsize=12, fontweight='bold')
    ax.set_title('True vs Predicted Labels', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'D{i}' for i in range(len(true_labels))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = 'results/clustering_verification.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {save_path}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


def main():
    """Run verification on the last experiment"""
    
    # Recreate the experiment
    print("Recreating experiment to verify results...\n")
    
    # Create domains
    domains = create_synthetic_mo_domains(n_domains=9, obs_dim=4, action_dim=2, seed=42)
    domain_info = get_domain_info(domains)
    
    # Ground truth labels
    true_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    
    # Load or recompute clustering
    config = ClusteringConfig(
        n_clusters=3,
        n_gradient_samples=50,
        alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0],
        random_seed=42,
        policy_hidden_dims=[32, 32]
    )
    
    clustering = GradientDomainClustering(config)
    
    print("Computing gradients (this may take a minute)...")
    clustering.compute_domain_gradients(domains, obs_dim=4, action_dim=2)
    clustering.build_similarity_matrix()
    predicted_labels = clustering.apply_spectral_clustering()
    
    print("\n")
    
    # Analyze results
    analyze_clustering_results(
        domains,
        true_labels,
        predicted_labels,
        domain_info
    )


if __name__ == '__main__':
    main()
