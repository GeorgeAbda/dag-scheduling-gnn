"""
Analyze comprehensive gradient conflict study results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
with open('logs/comprehensive_study/comprehensive_results.json', 'r') as f:
    results = json.load(f)

architectures = [
    'Tiny GNN (1 layer, 16 dim)',
    'Small GNN (2 layers, 32 dim)',
    'Medium GNN (2 layers, 64 dim)',
    'Large GNN (2 layers, 128 dim)',
    'Deep GNN (3 layers, 128 dim)'
]

metrics = ['cosine', 'wasserstein', 'euclidean']

print("="*80)
print("COMPREHENSIVE GRADIENT CONFLICT STUDY - ANALYSIS")
print("="*80)

# Extract key statistics
print("\n## Architecture Complexity vs Performance\n")
print(f"{'Architecture':<35} {'Params':<10} {'Best Metric':<15} {'NMI':<8} {'ARI':<8} {'Sil':<8}")
print("-" * 90)

best_results = []

for arch in architectures:
    num_params = results[arch]['num_params']
    
    # Find best metric for this architecture
    best_nmi = 0
    best_metric = None
    best_ari = 0
    best_sil = 0
    
    for metric in metrics:
        nmi = results[arch]['metrics'][metric]['nmi']
        ari = results[arch]['metrics'][metric]['ari']
        sil = results[arch]['metrics'][metric]['silhouette']
        
        if nmi > best_nmi:
            best_nmi = nmi
            best_metric = metric
            best_ari = ari
            best_sil = sil
    
    print(f"{arch:<35} {num_params:<10,} {best_metric:<15} {best_nmi:<8.3f} {best_ari:<8.3f} {best_sil:<8.3f}")
    best_results.append((arch, num_params, best_metric, best_nmi, best_ari, best_sil))

# Key findings
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

print("\n### 1. Clustering Quality (NMI)")
print("\nNMI measures how well the discovered clusters match true labels (0-1, higher is better)")
print("Results show VERY LOW NMI across all architectures:")

for arch, _, metric, nmi, _, _ in best_results:
    print(f"  • {arch:<35} NMI={nmi:.3f} (metric: {metric})")

print("\n⚠️  CRITICAL ISSUE: All NMI scores are near zero!")
print("   This means gradient conflict does NOT reliably separate wide vs long_cp MDPs")

print("\n### 2. Best Performing Combination")
best_combo = max(best_results, key=lambda x: x[3])
print(f"\n  Architecture: {best_combo[0]}")
print(f"  Metric: {best_combo[2]}")
print(f"  NMI: {best_combo[3]:.3f}")
print(f"  ARI: {best_combo[4]:.3f}")

if best_combo[3] < 0.2:
    print("\n  ⚠️  Even the best combination has very poor clustering quality!")
    print("     NMI < 0.2 indicates almost random clustering")

print("\n### 3. Distance Metric Comparison")

metric_performance = {metric: [] for metric in metrics}

for arch in architectures:
    for metric in metrics:
        nmi = results[arch]['metrics'][metric]['nmi']
        metric_performance[metric].append(nmi)

print("\nAverage NMI by distance metric:")
for metric in metrics:
    avg_nmi = np.mean(metric_performance[metric])
    std_nmi = np.std(metric_performance[metric])
    print(f"  {metric:<12}: {avg_nmi:.4f} ± {std_nmi:.4f}")

best_metric_overall = max(metrics, key=lambda m: np.mean(metric_performance[m]))
print(f"\n  Best metric overall: {best_metric_overall}")

print("\n### 4. Architecture Complexity Effect")

complexities = [results[arch]['complexity'] for arch in architectures]
best_nmis = [max(results[arch]['metrics'][m]['nmi'] for m in metrics) for arch in architectures]

print("\nNMI vs Complexity:")
for arch, complexity, nmi in zip(architectures, complexities, best_nmis):
    print(f"  Complexity {complexity}: {nmi:.4f} ({arch})")

correlation = np.corrcoef(complexities, best_nmis)[0, 1]
print(f"\nCorrelation (complexity vs NMI): {correlation:.3f}")

if abs(correlation) < 0.3:
    print("  → WEAK: Architecture complexity doesn't improve clustering")
elif correlation > 0:
    print("  → POSITIVE: Larger architectures perform slightly better")
else:
    print("  → NEGATIVE: Simpler architectures perform better")

print("\n### 5. Conflict Statistics")

print("\nConflict range by architecture (cosine similarity):")
for arch in architectures:
    conflicts = results[arch]['metrics']['cosine']['conflicts']
    mean_c = np.mean(conflicts)
    std_c = np.std(conflicts)
    min_c = np.min(conflicts)
    max_c = np.max(conflicts)
    
    print(f"  {arch:<35} mean={mean_c:>7.3f}, std={std_c:>6.3f}, range=[{min_c:>6.3f}, {max_c:>6.3f}]")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

print("\n## Why is NMI so low?")
print("\n1. **Random Agent Problem**")
print("   • All agents are randomly initialized (untrained)")
print("   • Random policies don't have learned preferences")
print("   • Gradients reflect random exploration, not task structure")

print("\n2. **Gradient Noise**")
print("   • REINFORCE gradients are high-variance")
print("   • Only 3 replicates per MDP may not be enough")
print("   • Stochastic environments add more noise")

print("\n3. **Weak Signal**")
print("   • Makespan and energy objectives may not conflict strongly")
print("   • Both objectives favor similar scheduling strategies")
print("   • Need more extreme objective trade-offs")

print("\n4. **Architecture Independence**")
print("   • All architectures perform similarly poorly")
print("   • Suggests the problem is fundamental, not architecture-specific")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("\n### To Improve Gradient-Based Domain Discovery:")

print("\n1. **Use Trained Agents**")
print("   • Train agents on mixed data first")
print("   • Trained agents have learned preferences")
print("   • Gradients will reflect actual task conflicts")

print("\n2. **Increase Gradient Samples**")
print("   • Use 10+ replicates instead of 3")
print("   • Longer rollouts (512+ steps)")
print("   • Average over multiple random seeds")

print("\n3. **Stronger Objective Conflicts**")
print("   • Use extreme weight differences:")
print("     - Wide MDP: -0.1*makespan - 0.9*energy")
print("     - Long CP: -0.9*makespan - 0.1*energy")
print("   • This creates explicit opposing gradients")

print("\n4. **Alternative Approaches**")
print("   • Use performance-based clustering instead of gradients")
print("   • Cluster by actual makespan/energy trade-offs")
print("   • Use domain adaptation metrics (e.g., transfer learning)")

print("\n5. **Validation**")
print("   • Test discovered domains by training separate policies")
print("   • Measure performance improvement vs single policy")
print("   • Use cross-validation across MDP types")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("\n**Current Results: Gradient conflict with random agents does NOT work**")
print("\nThe comprehensive study reveals:")
print("  ✗ NMI scores near zero across all configurations")
print("  ✗ No distance metric significantly outperforms others")
print("  ✗ Architecture complexity doesn't help")
print("  ✗ Gradient conflict fails to identify MDP structure")

print("\n**Next Steps:**")
print("  1. Repeat experiment with TRAINED agents")
print("  2. Use explicit multi-objective rewards with opposing weights")
print("  3. Increase gradient sampling (10+ replicates)")
print("  4. Consider alternative domain discovery methods")

print("\n" + "="*80)
