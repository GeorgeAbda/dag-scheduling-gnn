"""
Improved gradient-based clustering with better gradient estimation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict
import copy

from gradient_domain_clustering import (
    GradientDomainClustering, 
    ClusteringConfig, 
    SimplePolicy
)
from mo_gymnasium_loader import create_synthetic_mo_domains, get_domain_info


class ImprovedGradientClustering(GradientDomainClustering):
    """
    Enhanced version with better gradient estimation strategies
    """
    
    def compute_domain_gradients_improved(
        self,
        domains: Dict[str, any],
        obs_dim: int,
        action_dim: int,
        use_shared_policy: bool = True,
        use_multiple_inits: bool = True,
        n_policy_inits: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Improved gradient computation with multiple strategies
        
        Args:
            domains: Dictionary of domain_name -> environment
            obs_dim: Observation dimension
            action_dim: Action dimension
            use_shared_policy: Use same initial policy for all domains
            use_multiple_inits: Average over multiple policy initializations
            n_policy_inits: Number of different policy initializations
            
        Returns:
            Dictionary mapping domain names to gradient vectors
        """
        print(f"Computing gradients for {len(domains)} domains...")
        print(f"  Shared policy: {use_shared_policy}")
        print(f"  Multiple inits: {use_multiple_inits} (n={n_policy_inits})")
        
        if use_multiple_inits:
            # Average gradients over multiple policy initializations
            all_gradients = {name: [] for name in domains.keys()}
            
            for init_idx in range(n_policy_inits):
                print(f"\n--- Policy Initialization {init_idx + 1}/{n_policy_inits} ---")
                
                # Create policy for this initialization
                torch.manual_seed(self.config.random_seed + init_idx)
                np.random.seed(self.config.random_seed + init_idx)
                base_policy = SimplePolicy(obs_dim, action_dim, self.config.policy_hidden_dims)
                
                for domain_name, env in domains.items():
                    print(f"Processing domain: {domain_name}")
                    
                    # Use shared or separate policy
                    if use_shared_policy:
                        policy = copy.deepcopy(base_policy)
                    else:
                        policy = SimplePolicy(obs_dim, action_dim, self.config.policy_hidden_dims)
                    
                    # Compute gradients for different alpha values
                    domain_grad_vectors = []
                    
                    for alpha in self.config.alpha_values:
                        grad = self.compute_policy_gradient(
                            env, policy, alpha, self.config.n_gradient_samples
                        )
                        domain_grad_vectors.append(grad)
                    
                    # Concatenate and store
                    all_gradients[domain_name].append(np.concatenate(domain_grad_vectors))
            
            # Average gradients across initializations
            print("\nAveraging gradients across initializations...")
            for domain_name in domains.keys():
                self.domain_gradients[domain_name] = np.mean(
                    all_gradients[domain_name], axis=0
                )
        
        else:
            # Single initialization (original method)
            torch.manual_seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            base_policy = SimplePolicy(obs_dim, action_dim, self.config.policy_hidden_dims)
            
            for domain_name, env in domains.items():
                print(f"Processing domain: {domain_name}")
                
                if use_shared_policy:
                    policy = copy.deepcopy(base_policy)
                else:
                    policy = SimplePolicy(obs_dim, action_dim, self.config.policy_hidden_dims)
                
                domain_grad_vectors = []
                
                for alpha in self.config.alpha_values:
                    print(f"  Computing gradient for α={alpha:.2f}")
                    grad = self.compute_policy_gradient(
                        env, policy, alpha, self.config.n_gradient_samples
                    )
                    domain_grad_vectors.append(grad)
                
                self.domain_gradients[domain_name] = np.concatenate(domain_grad_vectors)
        
        return self.domain_gradients


def run_improved_experiment(
    strategy: str = "all_improvements",
    n_domains: int = 9,
    n_clusters: int = 3
):
    """
    Run experiment with different improvement strategies
    
    Args:
        strategy: One of:
            - "baseline": Original method
            - "more_samples": Increase gradient samples
            - "more_alphas": More trade-off values
            - "shared_policy": Use same initial policy
            - "multiple_inits": Average over multiple initializations
            - "all_improvements": Combine all improvements
    """
    print("=" * 80)
    print(f"IMPROVED CLUSTERING EXPERIMENT: {strategy.upper()}")
    print("=" * 80)
    
    # Create domains
    domains = create_synthetic_mo_domains(n_domains=n_domains, seed=42)
    domain_info = get_domain_info(domains)
    true_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    
    # Configure based on strategy
    if strategy == "baseline":
        config = ClusteringConfig(
            n_clusters=n_clusters,
            n_gradient_samples=50,
            alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0],
            random_seed=42,
            policy_hidden_dims=[32, 32]
        )
        use_shared = False
        use_multiple = False
        n_inits = 1
        
    elif strategy == "more_samples":
        config = ClusteringConfig(
            n_clusters=n_clusters,
            n_gradient_samples=200,  # 4x increase
            alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0],
            random_seed=42,
            policy_hidden_dims=[32, 32]
        )
        use_shared = False
        use_multiple = False
        n_inits = 1
        
    elif strategy == "more_alphas":
        config = ClusteringConfig(
            n_clusters=n_clusters,
            n_gradient_samples=50,
            alpha_values=np.linspace(0, 1, 11).tolist(),  # 11 values
            random_seed=42,
            policy_hidden_dims=[32, 32]
        )
        use_shared = False
        use_multiple = False
        n_inits = 1
        
    elif strategy == "shared_policy":
        config = ClusteringConfig(
            n_clusters=n_clusters,
            n_gradient_samples=50,
            alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0],
            random_seed=42,
            policy_hidden_dims=[32, 32]
        )
        use_shared = True
        use_multiple = False
        n_inits = 1
        
    elif strategy == "multiple_inits":
        config = ClusteringConfig(
            n_clusters=n_clusters,
            n_gradient_samples=50,
            alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0],
            random_seed=42,
            policy_hidden_dims=[32, 32]
        )
        use_shared = True
        use_multiple = True
        n_inits = 5
        
    elif strategy == "all_improvements":
        config = ClusteringConfig(
            n_clusters=n_clusters,
            n_gradient_samples=100,  # More samples
            alpha_values=np.linspace(0, 1, 11).tolist(),  # More alphas
            random_seed=42,
            policy_hidden_dims=[64, 64]  # Larger network
        )
        use_shared = True
        use_multiple = True
        n_inits = 3
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"\nConfiguration:")
    print(f"  Gradient samples: {config.n_gradient_samples}")
    print(f"  Alpha values: {len(config.alpha_values)}")
    print(f"  Policy hidden dims: {config.policy_hidden_dims}")
    print(f"  Shared policy: {use_shared}")
    print(f"  Multiple inits: {use_multiple} (n={n_inits})")
    
    # Run clustering
    clustering = ImprovedGradientClustering(config)
    clustering.compute_domain_gradients_improved(
        domains, 
        obs_dim=4, 
        action_dim=2,
        use_shared_policy=use_shared,
        use_multiple_inits=use_multiple,
        n_policy_inits=n_inits
    )
    
    clustering.build_similarity_matrix()
    predicted_labels = clustering.apply_spectral_clustering()
    
    # Evaluate
    metrics = clustering.evaluate_clustering(true_labels)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nStrategy: {strategy}")
    print(f"\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    print(f"\nTrue labels:      {true_labels}")
    print(f"Predicted labels: {predicted_labels}")
    
    # Cluster summary
    cluster_summary = clustering.get_cluster_summary(list(domains.keys()))
    print("\nCluster Summary:")
    for cluster_id, domain_names in cluster_summary.items():
        print(f"  Cluster {cluster_id}: {domain_names}")
    
    # Visualize
    import matplotlib.pyplot as plt
    clustering.visualize_results(
        list(domains.keys()), 
        save_path=f'results/improved_{strategy}.png'
    )
    
    return clustering, metrics


if __name__ == '__main__':
    import sys
    
    strategies = [
        "baseline",
        "more_samples",
        "more_alphas", 
        "shared_policy",
        "multiple_inits",
        "all_improvements"
    ]
    
    if len(sys.argv) > 1:
        strategy = sys.argv[1]
        if strategy not in strategies:
            print(f"Unknown strategy: {strategy}")
            print(f"Available: {strategies}")
            sys.exit(1)
    else:
        strategy = "all_improvements"
    
    clustering, metrics = run_improved_experiment(strategy)
