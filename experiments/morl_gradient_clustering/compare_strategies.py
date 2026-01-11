"""
Compare different improvement strategies for gradient-based clustering
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from improved_clustering import run_improved_experiment


def compare_all_strategies():
    """Run and compare all improvement strategies"""
    
    strategies = [
        ("baseline", "Baseline (50 samples, 5 alphas)"),
        ("more_samples", "More Samples (200 samples)"),
        ("more_alphas", "More Alphas (11 values)"),
        ("shared_policy", "Shared Policy Init"),
        ("multiple_inits", "Multiple Inits (5x)"),
        ("all_improvements", "All Improvements Combined")
    ]
    
    results = {}
    
    print("=" * 80)
    print("COMPARING IMPROVEMENT STRATEGIES")
    print("=" * 80)
    
    for strategy_id, strategy_name in strategies:
        print(f"\n{'='*80}")
        print(f"Running: {strategy_name}")
        print(f"{'='*80}\n")
        
        try:
            clustering, metrics = run_improved_experiment(strategy_id)
            results[strategy_name] = metrics
            print(f"\n✓ {strategy_name} complete")
        except Exception as e:
            print(f"\n✗ {strategy_name} failed: {e}")
            results[strategy_name] = None
    
    # Create comparison visualization
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON VISUALIZATION")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    strategy_names = [name for _, name in strategies if results.get(name) is not None]
    
    # Plot 1: ARI comparison
    ax = axes[0]
    ari_values = [results[name]['adjusted_rand_index'] for name in strategy_names]
    colors = ['red' if v < 0.2 else 'orange' if v < 0.6 else 'green' for v in ari_values]
    bars = ax.barh(range(len(strategy_names)), ari_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(strategy_names)))
    ax.set_yticklabels(strategy_names, fontsize=10)
    ax.set_xlabel('Adjusted Rand Index', fontsize=12, fontweight='bold')
    ax.set_title('Clustering Quality (ARI)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0.6, color='green', linestyle='--', alpha=0.3, label='Good (>0.6)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, ari_values)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)
    
    # Plot 2: NMI comparison
    ax = axes[1]
    nmi_values = [results[name]['normalized_mutual_info'] for name in strategy_names]
    colors = ['red' if v < 0.3 else 'orange' if v < 0.7 else 'green' for v in nmi_values]
    bars = ax.barh(range(len(strategy_names)), nmi_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(strategy_names)))
    ax.set_yticklabels(strategy_names, fontsize=10)
    ax.set_xlabel('Normalized Mutual Information', fontsize=12, fontweight='bold')
    ax.set_title('Information Sharing (NMI)', fontsize=14, fontweight='bold')
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.3, label='Moderate (>0.5)')
    ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.3, label='Strong (>0.7)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, nmi_values)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)
    
    # Plot 3: Silhouette comparison
    ax = axes[2]
    sil_values = [results[name]['silhouette_score'] for name in strategy_names]
    colors = ['red' if v < 0.2 else 'orange' if v < 0.5 else 'green' for v in sil_values]
    bars = ax.barh(range(len(strategy_names)), sil_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(strategy_names)))
    ax.set_yticklabels(strategy_names, fontsize=10)
    ax.set_xlabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Separation', fontsize=14, fontweight='bold')
    ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.3, label='Reasonable (>0.3)')
    ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.3, label='Good (>0.5)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sil_values)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/strategy_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Comparison saved to results/strategy_comparison.png")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Strategy':<35} {'ARI':>8} {'NMI':>8} {'Silhouette':>12}")
    print("-" * 80)
    
    for name in strategy_names:
        metrics = results[name]
        print(f"{name:<35} {metrics['adjusted_rand_index']:>8.4f} "
              f"{metrics['normalized_mutual_info']:>8.4f} "
              f"{metrics['silhouette_score']:>12.4f}")
    
    # Find best strategy
    best_ari = max(strategy_names, key=lambda x: results[x]['adjusted_rand_index'])
    best_nmi = max(strategy_names, key=lambda x: results[x]['normalized_mutual_info'])
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print(f"\nBest ARI: {best_ari}")
    print(f"  Score: {results[best_ari]['adjusted_rand_index']:.4f}")
    print(f"\nBest NMI: {best_nmi}")
    print(f"  Score: {results[best_nmi]['normalized_mutual_info']:.4f}")
    
    return results


if __name__ == '__main__':
    results = compare_all_strategies()
