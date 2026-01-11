"""
Example script demonstrating gradient-based domain clustering on MORL environments

This script shows how to:
1. Load MO-Gymnasium environments or create synthetic domains
2. Compute policy gradients for different reward trade-offs
3. Build similarity matrix using cosine similarity
4. Apply spectral clustering
5. Evaluate and visualize results
"""

import numpy as np
import argparse
import os
from pathlib import Path

from gradient_domain_clustering import (
    GradientDomainClustering,
    ClusteringConfig,
    SimplePolicy
)
from mo_gymnasium_loader import (
    create_synthetic_mo_domains,
    load_mo_gymnasium_environments,
    get_default_mo_environments,
    get_mo_mujoco_environments,
    get_domain_info
)


def run_synthetic_experiment(
    n_domains: int = 9,
    n_clusters: int = 3,
    obs_dim: int = 4,
    action_dim: int = 2,
    n_gradient_samples: int = 50,
    output_dir: str = 'results'
):
    """
    Run clustering experiment on synthetic MO domains
    
    Args:
        n_domains: Number of synthetic domains
        n_clusters: Number of clusters to find
        obs_dim: Observation dimension
        action_dim: Action dimension
        n_gradient_samples: Number of samples for gradient estimation
        output_dir: Directory to save results
    """
    print("=" * 80)
    print("GRADIENT-BASED DOMAIN CLUSTERING: SYNTHETIC EXPERIMENT")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Create synthetic domains
    print(f"\n[Step 1] Creating {n_domains} synthetic MO domains...")
    domains = create_synthetic_mo_domains(
        n_domains=n_domains,
        obs_dim=obs_dim,
        action_dim=action_dim,
        seed=42
    )
    
    domain_info = get_domain_info(domains)
    print(f"Created domains: {list(domains.keys())}")
    
    # Print domain characteristics
    print("\nDomain characteristics:")
    for name, info in domain_info.items():
        if 'reward_weights' in info:
            weights = info['reward_weights']
            print(f"  {name}: reward_weights = [{weights[0]:.3f}, {weights[1]:.3f}]")
    
    # Ground truth labels (based on reward weight structure)
    true_labels = []
    for i in range(n_domains):
        if i < n_domains // 3:
            true_labels.append(0)  # Group 1: Favor objective 1
        elif i < 2 * n_domains // 3:
            true_labels.append(1)  # Group 2: Balanced
        else:
            true_labels.append(2)  # Group 3: Favor objective 2
    true_labels = np.array(true_labels)
    
    # Step 2: Configure clustering
    print(f"\n[Step 2] Configuring gradient-based clustering...")
    config = ClusteringConfig(
        n_clusters=n_clusters,
        n_gradient_samples=n_gradient_samples,
        alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0],
        random_seed=42,
        policy_hidden_dims=[32, 32]
    )
    print(f"  Number of clusters: {config.n_clusters}")
    print(f"  Gradient samples per domain: {config.n_gradient_samples}")
    print(f"  Alpha values: {config.alpha_values}")
    
    # Step 3: Initialize clustering
    clustering = GradientDomainClustering(config)
    
    # Step 4: Compute gradients
    print(f"\n[Step 3] Computing policy gradients...")
    print("This may take a few minutes...")
    clustering.compute_domain_gradients(domains, obs_dim, action_dim)
    print(f"Computed gradients for {len(clustering.domain_gradients)} domains")
    
    # Step 5: Build similarity matrix
    print(f"\n[Step 4] Building similarity matrix...")
    similarity_matrix = clustering.build_similarity_matrix()
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
    
    # Step 6: Apply spectral clustering
    print(f"\n[Step 5] Applying spectral clustering...")
    cluster_labels = clustering.apply_spectral_clustering()
    print(f"Cluster labels: {cluster_labels}")
    
    # Step 7: Evaluate clustering
    print(f"\n[Step 6] Evaluating clustering results...")
    metrics = clustering.evaluate_clustering(true_labels)
    
    print("\nClustering Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Step 8: Get cluster summary
    cluster_summary = clustering.get_cluster_summary(list(domains.keys()))
    print("\nCluster Summary:")
    for cluster_id, domain_names in cluster_summary.items():
        print(f"  Cluster {cluster_id}: {domain_names}")
    
    # Step 9: Visualize results
    print(f"\n[Step 7] Visualizing results...")
    save_path = os.path.join(output_dir, 'clustering_results_synthetic.png')
    clustering.visualize_results(list(domains.keys()), save_path=save_path)
    
    # Save results
    results_file = os.path.join(output_dir, 'clustering_results_synthetic.txt')
    with open(results_file, 'w') as f:
        f.write("GRADIENT-BASED DOMAIN CLUSTERING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of domains: {n_domains}\n")
        f.write(f"Number of clusters: {n_clusters}\n")
        f.write(f"Gradient samples: {n_gradient_samples}\n\n")
        
        f.write("Clustering Metrics:\n")
        for metric_name, value in metrics.items():
            f.write(f"  {metric_name}: {value:.4f}\n")
        
        f.write("\nCluster Summary:\n")
        for cluster_id, domain_names in cluster_summary.items():
            f.write(f"  Cluster {cluster_id}: {domain_names}\n")
        
        f.write("\nTrue Labels: " + str(true_labels) + "\n")
        f.write("Predicted Labels: " + str(cluster_labels) + "\n")
    
    print(f"\nResults saved to {results_file}")
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    
    return clustering, metrics


def run_mo_gymnasium_experiment(
    env_type: str = 'default',
    n_clusters: int = 3,
    n_gradient_samples: int = 20,
    output_dir: str = 'results'
):
    """
    Run clustering experiment on MO-Gymnasium environments
    
    Args:
        env_type: Type of environments ('default' or 'mujoco')
        n_clusters: Number of clusters to find
        n_gradient_samples: Number of samples for gradient estimation
        output_dir: Directory to save results
    """
    print("=" * 80)
    print("GRADIENT-BASED DOMAIN CLUSTERING: MO-GYMNASIUM EXPERIMENT")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load environments
    print(f"\n[Step 1] Loading MO-Gymnasium environments ({env_type})...")
    
    if env_type == 'default':
        env_configs = get_default_mo_environments()
    elif env_type == 'mujoco':
        env_configs = get_mo_mujoco_environments()
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
    
    try:
        domains = load_mo_gymnasium_environments(env_configs)
    except Exception as e:
        print(f"\nError loading MO-Gymnasium environments: {e}")
        print("Falling back to synthetic domains...")
        return run_synthetic_experiment(
            n_domains=6,
            n_clusters=n_clusters,
            n_gradient_samples=n_gradient_samples,
            output_dir=output_dir
        )
    
    if len(domains) == 0:
        print("No environments loaded. Falling back to synthetic domains...")
        return run_synthetic_experiment(
            n_domains=6,
            n_clusters=n_clusters,
            n_gradient_samples=n_gradient_samples,
            output_dir=output_dir
        )
    
    domain_info = get_domain_info(domains)
    print(f"Loaded {len(domains)} environments")
    
    # Get dimensions from first environment
    first_env = list(domains.values())[0]
    obs_dim = first_env.observation_space.shape[0]
    action_dim = first_env.action_space.shape[0]
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Step 2: Configure clustering
    print(f"\n[Step 2] Configuring gradient-based clustering...")
    config = ClusteringConfig(
        n_clusters=n_clusters,
        n_gradient_samples=n_gradient_samples,
        alpha_values=[0.0, 0.33, 0.67, 1.0],
        random_seed=42,
        policy_hidden_dims=[64, 64]
    )
    
    # Step 3: Initialize clustering
    clustering = GradientDomainClustering(config)
    
    # Step 4: Compute gradients
    print(f"\n[Step 3] Computing policy gradients...")
    print("This may take several minutes...")
    clustering.compute_domain_gradients(domains, obs_dim, action_dim)
    
    # Step 5: Build similarity matrix
    print(f"\n[Step 4] Building similarity matrix...")
    similarity_matrix = clustering.build_similarity_matrix()
    
    # Step 6: Apply spectral clustering
    print(f"\n[Step 5] Applying spectral clustering...")
    cluster_labels = clustering.apply_spectral_clustering()
    
    # Step 7: Evaluate clustering
    print(f"\n[Step 6] Evaluating clustering results...")
    metrics = clustering.evaluate_clustering()
    
    print("\nClustering Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Step 8: Get cluster summary
    cluster_summary = clustering.get_cluster_summary(list(domains.keys()))
    print("\nCluster Summary:")
    for cluster_id, domain_names in cluster_summary.items():
        print(f"  Cluster {cluster_id}: {domain_names}")
    
    # Step 9: Visualize results
    print(f"\n[Step 7] Visualizing results...")
    save_path = os.path.join(output_dir, f'clustering_results_{env_type}.png')
    clustering.visualize_results(list(domains.keys()), save_path=save_path)
    
    # Clean up
    for env in domains.values():
        env.close()
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    
    return clustering, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Gradient-based Domain Clustering for MORL'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='synthetic',
        choices=['synthetic', 'mo_gym_default', 'mo_gym_mujoco'],
        help='Type of experiment to run'
    )
    parser.add_argument(
        '--n_domains',
        type=int,
        default=9,
        help='Number of synthetic domains (for synthetic experiment)'
    )
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=3,
        help='Number of clusters'
    )
    parser.add_argument(
        '--n_gradient_samples',
        type=int,
        default=50,
        help='Number of samples for gradient estimation'
    )
    parser.add_argument(
        '--obs_dim',
        type=int,
        default=4,
        help='Observation dimension (for synthetic experiment)'
    )
    parser.add_argument(
        '--action_dim',
        type=int,
        default=2,
        help='Action dimension (for synthetic experiment)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    if args.experiment == 'synthetic':
        clustering, metrics = run_synthetic_experiment(
            n_domains=args.n_domains,
            n_clusters=args.n_clusters,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            n_gradient_samples=args.n_gradient_samples,
            output_dir=args.output_dir
        )
    elif args.experiment == 'mo_gym_default':
        clustering, metrics = run_mo_gymnasium_experiment(
            env_type='default',
            n_clusters=args.n_clusters,
            n_gradient_samples=args.n_gradient_samples,
            output_dir=args.output_dir
        )
    elif args.experiment == 'mo_gym_mujoco':
        clustering, metrics = run_mo_gymnasium_experiment(
            env_type='mujoco',
            n_clusters=args.n_clusters,
            n_gradient_samples=args.n_gradient_samples,
            output_dir=args.output_dir
        )
    
    return clustering, metrics


if __name__ == '__main__':
    main()
