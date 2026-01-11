"""
Quick test script to verify the gradient-based clustering implementation
"""

import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from gradient_domain_clustering import (
    GradientDomainClustering,
    ClusteringConfig,
    SimplePolicy
)
from mo_gymnasium_loader import (
    create_synthetic_mo_domains,
    get_domain_info
)


def test_synthetic_domains():
    """Test clustering on synthetic domains"""
    print("=" * 80)
    print("TESTING GRADIENT-BASED DOMAIN CLUSTERING")
    print("=" * 80)
    
    # Test 1: Create synthetic domains
    print("\n[Test 1] Creating synthetic domains...")
    try:
        domains = create_synthetic_mo_domains(n_domains=6, obs_dim=4, action_dim=2)
        print(f"✓ Created {len(domains)} synthetic domains")
        
        domain_info = get_domain_info(domains)
        print("  Domain characteristics:")
        for name, info in domain_info.items():
            if 'reward_weights' in info:
                weights = info['reward_weights']
                print(f"    {name}: weights=[{weights[0]:.3f}, {weights[1]:.3f}]")
    except Exception as e:
        print(f"✗ Failed to create domains: {e}")
        return False
    
    # Test 2: Initialize clustering
    print("\n[Test 2] Initializing clustering...")
    try:
        config = ClusteringConfig(
            n_clusters=3,
            n_gradient_samples=10,  # Small number for quick test
            alpha_values=[0.0, 0.5, 1.0],
            random_seed=42,
            policy_hidden_dims=[16, 16]  # Small network for quick test
        )
        clustering = GradientDomainClustering(config)
        print("✓ Clustering initialized")
        print(f"  Config: {config.n_clusters} clusters, {config.n_gradient_samples} samples")
    except Exception as e:
        print(f"✗ Failed to initialize clustering: {e}")
        return False
    
    # Test 3: Compute gradients
    print("\n[Test 3] Computing gradients...")
    try:
        clustering.compute_domain_gradients(domains, obs_dim=4, action_dim=2)
        print(f"✓ Computed gradients for {len(clustering.domain_gradients)} domains")
        
        # Check gradient shapes
        first_grad = list(clustering.domain_gradients.values())[0]
        print(f"  Gradient vector size: {len(first_grad)}")
    except Exception as e:
        print(f"✗ Failed to compute gradients: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Build similarity matrix
    print("\n[Test 4] Building similarity matrix...")
    try:
        similarity_matrix = clustering.build_similarity_matrix()
        print(f"✓ Built similarity matrix: shape={similarity_matrix.shape}")
        print(f"  Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
        
        # Check diagonal is 1.0
        diag_values = np.diag(similarity_matrix)
        if np.allclose(diag_values, 1.0, atol=1e-6):
            print("  ✓ Diagonal values are 1.0 (self-similarity)")
        else:
            print(f"  ⚠ Diagonal values: {diag_values}")
    except Exception as e:
        print(f"✗ Failed to build similarity matrix: {e}")
        return False
    
    # Test 5: Apply spectral clustering
    print("\n[Test 5] Applying spectral clustering...")
    try:
        cluster_labels = clustering.apply_spectral_clustering()
        print(f"✓ Clustering complete: {len(cluster_labels)} labels")
        print(f"  Cluster labels: {cluster_labels}")
        print(f"  Unique clusters: {np.unique(cluster_labels)}")
    except Exception as e:
        print(f"✗ Failed to apply clustering: {e}")
        return False
    
    # Test 6: Evaluate clustering
    print("\n[Test 6] Evaluating clustering...")
    try:
        # Create ground truth labels
        true_labels = np.array([0, 0, 1, 1, 2, 2])  # 3 groups of 2
        
        metrics = clustering.evaluate_clustering(true_labels)
        print("✓ Evaluation complete")
        print("  Metrics:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")
    except Exception as e:
        print(f"✗ Failed to evaluate: {e}")
        return False
    
    # Test 7: Get cluster summary
    print("\n[Test 7] Getting cluster summary...")
    try:
        cluster_summary = clustering.get_cluster_summary(list(domains.keys()))
        print("✓ Cluster summary:")
        for cluster_id, domain_names in cluster_summary.items():
            print(f"    Cluster {cluster_id}: {domain_names}")
    except Exception as e:
        print(f"✗ Failed to get summary: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    return True


def test_policy_network():
    """Test policy network"""
    print("\n[Bonus Test] Testing policy network...")
    try:
        policy = SimplePolicy(obs_dim=4, action_dim=2, hidden_dims=[16, 16])
        
        # Test forward pass
        import torch
        obs = torch.randn(1, 4)
        action = policy(obs)
        
        print(f"✓ Policy network works")
        print(f"  Input shape: {obs.shape}")
        print(f"  Output shape: {action.shape}")
        print(f"  Output range: [{action.min():.3f}, {action.max():.3f}]")
        
        # Test gradient computation
        loss = action.sum()
        loss.backward()
        
        has_gradients = any(p.grad is not None for p in policy.parameters())
        if has_gradients:
            print("  ✓ Gradients computed successfully")
        else:
            print("  ⚠ No gradients computed")
        
        return True
    except Exception as e:
        print(f"✗ Policy network test failed: {e}")
        return False


if __name__ == '__main__':
    print("\nRunning implementation tests...\n")
    
    # Test policy network
    policy_ok = test_policy_network()
    
    # Test main clustering
    clustering_ok = test_synthetic_domains()
    
    if policy_ok and clustering_ok:
        print("\n✓ All tests passed! Implementation is working correctly.")
        print("\nYou can now run the full experiment with:")
        print("  python run_clustering_experiment.py --experiment synthetic")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
