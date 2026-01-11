"""
Wasserstein Gradient Domain Discovery (WGDD)

A novel unsupervised method for discovering strategy domains in multi-objective RL.

Key innovations over basic gradient subspace discovery:
1. Wasserstein distance between per-MDP gradient distributions (not just mean/std)
2. Conflict tensor analysis - pairwise objective conflicts per MDP
3. Automatic domain number selection via silhouette/gap statistic
4. Multi-scale gradient sampling (extremes + uniform + focused)
5. Bootstrap stability analysis
6. Pareto gradient alignment score

Method overview:
1. Train generalist on mixed MDP distribution
2. For each MDP, collect gradient distributions under multiple objectives
3. Compute pairwise Wasserstein distances between MDP gradient distributions
4. Build affinity matrix from distances
5. Spectral clustering with automatic k selection
6. Validate via bootstrap resampling

This creates a principled, unsupervised way to discover which MDPs induce
similar multi-objective trade-offs, without knowing the true domain labels.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_rand_score, 
    silhouette_score, silhouette_samples
)
from sklearn.manifold import TSNE

# Add project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.dataset_generator.gen_dataset import DatasetArgs


# Publication style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'font.size': 11,
    'axes.titleweight': 'bold'
})

PALETTE = ['#00FFFF', '#FF00FF', '#32CD32', '#FFA500', '#8A2BE2', '#008080', '#FF0000', '#0000FF']


@dataclass
class WGDDConfig:
    """Configuration for Wasserstein Gradient Domain Discovery."""
    # MDP sampling
    num_mdps: int = 40
    
    # Objective sampling (multi-scale)
    num_extreme_objectives: int = 2      # (1,0) and (0,1)
    num_uniform_objectives: int = 10     # Uniform in [0,1]
    num_focused_objectives: int = 8      # Concentrated at 0.25, 0.5, 0.75
    
    # Gradient collection
    num_replicates: int = 5              # More replicates for distribution
    num_steps: int = 256
    
    # Clustering
    max_clusters: int = 6
    clustering_method: str = "spectral"  # "spectral" or "kmeans"
    
    # Bootstrap
    num_bootstrap: int = 20
    bootstrap_ratio: float = 0.8
    
    # Output
    output_dir: str = "logs/wgdd"
    
    @property
    def total_objectives(self) -> int:
        return self.num_extreme_objectives + self.num_uniform_objectives + self.num_focused_objectives


def create_mdp_from_params(params: List[float], host_specs_file: str):
    """Create MDP environment from parameters."""
    edge_prob, min_tasks, max_tasks, min_length, max_length = params
    style = "long_cp" if edge_prob > 0.5 else "wide"
    
    args = DatasetArgs(
        host_count=10, vm_count=10, workflow_count=1, style=style,
        gnp_p=float(edge_prob), gnp_min_n=int(min_tasks), gnp_max_n=int(max_tasks),
        min_task_length=int(min_length), max_task_length=int(max_length),
    )
    os.environ['HOST_SPECS_PATH'] = host_specs_file
    
    env = GinAgentWrapper(CloudSchedulingGymEnvironment(
        dataset_args=args, collect_timelines=False, compute_metrics=True
    ))
    return env, args


def sample_objectives(config: WGDDConfig) -> List[Tuple[float, float]]:
    """
    Multi-scale objective sampling.
    
    Returns list of (alpha_makespan, alpha_energy) tuples covering:
    - Extremes: pure objectives
    - Uniform: broad exploration
    - Focused: key trade-off points
    """
    objectives = []
    
    # Extremes
    objectives.append((1.0, 0.0))  # Pure makespan
    objectives.append((0.0, 1.0))  # Pure energy
    
    # Uniform sampling
    rng = np.random.default_rng(42)
    for a in rng.uniform(0.05, 0.95, size=config.num_uniform_objectives):
        objectives.append((float(a), float(1 - a)))
    
    # Focused at key trade-off points
    focus_points = [0.25, 0.5, 0.75]
    for fp in focus_points:
        for delta in [-0.05, 0, 0.05]:
            a = np.clip(fp + delta, 0.01, 0.99)
            objectives.append((float(a), float(1 - a)))
    
    # Remove duplicates and limit
    seen = set()
    unique = []
    for obj in objectives:
        key = (round(obj[0], 3), round(obj[1], 3))
        if key not in seen:
            seen.add(key)
            unique.append(obj)
    
    return unique[:config.total_objectives]


def compute_gradient(
    agent, env, alpha_m: float, alpha_e: float,
    num_steps: int, device: torch.device, seed: int
) -> Optional[np.ndarray]:
    """Compute policy gradient for weighted objective."""
    obs_list, action_list, rewards = [], [], []
    
    obs, _ = env.reset(seed=int(seed))
    
    for _ in range(num_steps):
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(1, -1)).to(device)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_t)
        
        obs_list.append(obs_t)
        action_list.append(action)
        
        obs, reward, terminated, truncated, info = env.step(int(action.item()))
        
        if terminated or truncated:
            makespan = info.get('makespan', 0.0)
            energy = info.get('total_energy_active', 0.0) / 1e8
            episode_reward = -alpha_m * makespan - alpha_e * energy
            rewards.append(episode_reward)
            obs, _ = env.reset()
        else:
            rewards.append(0.0)
    
    if not rewards or not obs_list:
        return None
    
    # REINFORCE gradient
    agent.zero_grad()
    
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    
    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    loss = 0.0
    for obs_t, action_t, ret in zip(obs_list, action_list, returns):
        _, log_prob, _, _ = agent.get_action_and_value(obs_t, action_t)
        loss = loss - log_prob * ret
    
    loss = loss / len(obs_list)
    loss.backward()
    
    grad_parts = []
    for p in agent.actor.parameters():
        if p.grad is not None:
            grad_parts.append(p.grad.view(-1).detach().clone())
    
    agent.zero_grad()
    
    if not grad_parts:
        return None
    
    return torch.cat(grad_parts).cpu().numpy()


def collect_gradient_distributions(
    agent, mdp_configs: List[dict], objectives: List[Tuple[float, float]],
    host_specs_file: str, device: torch.device, config: WGDDConfig
) -> Dict[int, Dict[Tuple[float, float], np.ndarray]]:
    """
    Collect gradient distributions for all (MDP, objective) pairs.
    
    Returns: {mdp_idx: {objective: gradient_samples_array}}
    """
    distributions = {}
    
    pbar = tqdm(mdp_configs, desc="Collecting gradient distributions", ncols=100)
    
    for mdp_idx, cfg in enumerate(pbar):
        params = [cfg['edge_prob'], cfg['min_tasks'], cfg['max_tasks'], 
                  cfg['min_length'], cfg['max_length']]
        env, _ = create_mdp_from_params(params, host_specs_file)
        
        distributions[mdp_idx] = {}
        
        for obj in objectives:
            samples = []
            for r in range(config.num_replicates):
                seed = 12345 + mdp_idx * 10000 + hash(obj) % 1000 + r
                g = compute_gradient(agent, env, obj[0], obj[1], 
                                    config.num_steps, device, seed)
                if g is not None:
                    samples.append(g)
            
            if samples:
                distributions[mdp_idx][obj] = np.stack(samples, axis=0)
        
        env.close()
        pbar.set_postfix({'mdp': mdp_idx, 'objs': len(distributions[mdp_idx])})
    
    return distributions


def compute_wasserstein_distance_matrix(
    distributions: Dict[int, Dict[Tuple[float, float], np.ndarray]],
    objectives: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Compute pairwise Wasserstein distances between MDP gradient distributions.
    
    For each pair (MDP_i, MDP_j), we compute the average Wasserstein distance
    across all objectives and gradient dimensions (using 1D projections).
    """
    n_mdps = len(distributions)
    distance_matrix = np.zeros((n_mdps, n_mdps))
    
    print("\nComputing Wasserstein distance matrix...")
    
    # Use PCA to reduce dimensionality for stable Wasserstein computation
    # First, collect all gradients to fit PCA
    all_grads = []
    for mdp_idx in distributions:
        for obj in distributions[mdp_idx]:
            all_grads.extend(distributions[mdp_idx][obj])
    
    all_grads = np.array(all_grads)
    
    # Fit PCA
    n_components = min(10, all_grads.shape[0] // 2, all_grads.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(all_grads)
    print(f"  PCA: {n_components} components, explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Compute distances
    for i in tqdm(range(n_mdps), desc="  Distance computation", ncols=100):
        for j in range(i + 1, n_mdps):
            distances_ij = []
            
            for obj in objectives:
                if obj not in distributions[i] or obj not in distributions[j]:
                    continue
                
                # Project gradients
                grads_i = pca.transform(distributions[i][obj])
                grads_j = pca.transform(distributions[j][obj])
                
                # Compute 1D Wasserstein for each PC dimension
                for pc in range(n_components):
                    w_dist = wasserstein_distance(grads_i[:, pc], grads_j[:, pc])
                    distances_ij.append(w_dist)
            
            if distances_ij:
                distance_matrix[i, j] = np.mean(distances_ij)
                distance_matrix[j, i] = distance_matrix[i, j]
    
    return distance_matrix, pca


def compute_conflict_tensor(
    distributions: Dict[int, Dict[Tuple[float, float], np.ndarray]],
    objectives: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Compute conflict tensor: for each MDP, measure pairwise conflict between objectives.
    
    Returns: [n_mdps, n_objectives, n_objectives] tensor of cosine similarities
    """
    n_mdps = len(distributions)
    n_objs = len(objectives)
    conflict_tensor = np.zeros((n_mdps, n_objs, n_objs))
    
    print("\nComputing conflict tensor...")
    
    for mdp_idx in tqdm(range(n_mdps), desc="  Conflicts", ncols=100):
        for i, obj_i in enumerate(objectives):
            for j, obj_j in enumerate(objectives):
                if i >= j:
                    continue
                
                if obj_i not in distributions[mdp_idx] or obj_j not in distributions[mdp_idx]:
                    continue
                
                # Mean gradients
                g_i = distributions[mdp_idx][obj_i].mean(axis=0)
                g_j = distributions[mdp_idx][obj_j].mean(axis=0)
                
                # Cosine similarity
                norm_i = np.linalg.norm(g_i)
                norm_j = np.linalg.norm(g_j)
                
                if norm_i > 1e-8 and norm_j > 1e-8:
                    cos_sim = np.dot(g_i, g_j) / (norm_i * norm_j)
                    conflict_tensor[mdp_idx, i, j] = cos_sim
                    conflict_tensor[mdp_idx, j, i] = cos_sim
    
    return conflict_tensor


def compute_pareto_alignment_score(conflict_tensor: np.ndarray) -> np.ndarray:
    """
    Compute Pareto alignment score for each MDP.
    
    High score = objectives are aligned (no real trade-off)
    Low score = objectives conflict (true multi-objective problem)
    """
    # Average conflict between pure objectives (makespan vs energy)
    # Objectives 0 and 1 are the extremes
    pareto_scores = conflict_tensor[:, 0, 1]
    return pareto_scores


def select_optimal_k(
    affinity_matrix: np.ndarray, 
    max_k: int = 6,
    method: str = "silhouette"
) -> Tuple[int, Dict[int, float]]:
    """
    Automatically select optimal number of clusters.
    
    Returns: (optimal_k, scores_dict)
    """
    scores = {}
    
    print(f"\nSelecting optimal k (max={max_k})...")
    
    for k in range(2, max_k + 1):
        # Convert affinity to distance for clustering
        # Affinity = exp(-distance), so distance = -log(affinity + eps)
        distance_matrix = affinity_matrix.copy()
        
        clustering = SpectralClustering(
            n_clusters=k, affinity='precomputed',
            random_state=42, n_init=10
        )
        
        # Convert distance to similarity for spectral clustering
        sigma = np.median(distance_matrix[distance_matrix > 0])
        similarity = np.exp(-distance_matrix / (2 * sigma ** 2))
        np.fill_diagonal(similarity, 1.0)
        
        labels = clustering.fit_predict(similarity)
        
        if len(np.unique(labels)) > 1:
            score = silhouette_score(distance_matrix, labels, metric='precomputed')
        else:
            score = -1
        
        scores[k] = score
        print(f"  k={k}: silhouette={score:.3f}")
    
    optimal_k = max(scores, key=scores.get)
    print(f"  → Optimal k={optimal_k}")
    
    return optimal_k, scores


def bootstrap_stability_analysis(
    distributions: Dict[int, Dict[Tuple[float, float], np.ndarray]],
    objectives: List[Tuple[float, float]],
    n_clusters: int,
    config: WGDDConfig
) -> Dict[str, float]:
    """
    Assess clustering stability via bootstrap resampling.
    
    Returns stability metrics.
    """
    print(f"\nBootstrap stability analysis ({config.num_bootstrap} iterations)...")
    
    mdp_indices = list(distributions.keys())
    n_mdps = len(mdp_indices)
    
    # Co-occurrence matrix: how often pairs cluster together
    cooccurrence = np.zeros((n_mdps, n_mdps))
    
    for b in tqdm(range(config.num_bootstrap), desc="  Bootstrap", ncols=100):
        # Sample MDPs
        sample_size = int(n_mdps * config.bootstrap_ratio)
        sample_indices = np.random.choice(n_mdps, size=sample_size, replace=False)
        
        # Build distance matrix for sample
        sample_dists = {}
        for new_idx, orig_idx in enumerate(sample_indices):
            sample_dists[new_idx] = distributions[orig_idx]
        
        # Compute distance matrix
        dist_matrix, _ = compute_wasserstein_distance_matrix(sample_dists, objectives)
        
        # Cluster
        sigma = np.median(dist_matrix[dist_matrix > 0]) + 1e-8
        similarity = np.exp(-dist_matrix / (2 * sigma ** 2))
        np.fill_diagonal(similarity, 1.0)
        
        clustering = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed',
            random_state=b, n_init=5
        )
        labels = clustering.fit_predict(similarity)
        
        # Update co-occurrence
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                orig_i, orig_j = sample_indices[i], sample_indices[j]
                if labels[i] == labels[j]:
                    cooccurrence[orig_i, orig_j] += 1
                    cooccurrence[orig_j, orig_i] += 1
    
    # Normalize
    cooccurrence /= config.num_bootstrap
    
    # Stability score: average co-occurrence for pairs in same cluster
    stability = np.mean(cooccurrence[np.triu_indices(n_mdps, k=1)])
    
    return {
        'stability_score': float(stability),
        'cooccurrence_matrix': cooccurrence
    }


def wasserstein_domain_discovery(
    host_specs_file: str,
    model_path: str,
    config: WGDDConfig
) -> Dict:
    """
    Main Wasserstein Gradient Domain Discovery pipeline.
    """
    print("="*80)
    print("WASSERSTEIN GRADIENT DOMAIN DISCOVERY (WGDD)")
    print("="*80)
    print("\nA novel unsupervised method for multi-objective RL domain discovery")
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    os.environ['HOST_SPECS_PATH'] = host_specs_file
    
    # Sample MDPs (balanced wide + longcp)
    print(f"\n[1/7] Sampling {config.num_mdps} diverse MDPs...")
    mdp_configs = []
    true_labels = []
    
    for i in range(config.num_mdps // 2):
        # Wide
        edge_prob = 0.05 + (i / (config.num_mdps // 2)) * 0.35
        mdp_configs.append({
            'edge_prob': edge_prob,
            'min_tasks': random.randint(20, 40),
            'max_tasks': random.randint(40, 50),
            'min_length': random.randint(100, 2000),
            'max_length': random.randint(5000, 50000)
        })
        true_labels.append(0)
    
    for i in range(config.num_mdps // 2):
        # LongCP
        edge_prob = 0.60 + (i / (config.num_mdps // 2)) * 0.35
        mdp_configs.append({
            'edge_prob': edge_prob,
            'min_tasks': random.randint(5, 15),
            'max_tasks': random.randint(15, 20),
            'min_length': random.randint(50000, 100000),
            'max_length': random.randint(100000, 200000)
        })
        true_labels.append(1)
    
    true_labels = np.array(true_labels)
    print(f"  ✓ Wide: {np.sum(true_labels == 0)}, LongCP: {np.sum(true_labels == 1)}")
    
    # Sample objectives
    print(f"\n[2/7] Sampling multi-scale objectives...")
    objectives = sample_objectives(config)
    print(f"  ✓ {len(objectives)} objectives sampled")
    print(f"    Extremes: {objectives[:2]}")
    print(f"    Sample uniform: {objectives[2:5]}")
    
    # Load agent
    print(f"\n[3/7] Loading trained agent...")
    variant = AblationVariant(
        name="hetero", graph_type="hetero", gin_num_layers=2,
        use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
    )
    agent = AblationGinAgent(device=device, variant=variant, hidden_dim=128, embedding_dim=64)
    
    # Initialize
    dummy_env, _ = create_mdp_from_params([0.5, 20, 30, 1000, 5000], host_specs_file)
    dummy_obs, _ = dummy_env.reset(seed=42)
    dummy_obs_t = torch.from_numpy(np.asarray(dummy_obs, dtype=np.float32).reshape(1, -1)).to(device)
    with torch.no_grad():
        _ = agent.get_action_and_value(dummy_obs_t)
    dummy_env.close()
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    agent.load_state_dict(checkpoint)
    agent.eval()
    print(f"  ✓ Loaded from {model_path}")
    
    # Collect gradient distributions
    print(f"\n[4/7] Collecting gradient distributions...")
    distributions = collect_gradient_distributions(
        agent, mdp_configs, objectives, host_specs_file, device, config
    )
    
    # Compute Wasserstein distance matrix
    print(f"\n[5/7] Computing Wasserstein distances...")
    distance_matrix, pca = compute_wasserstein_distance_matrix(distributions, objectives)
    
    # Compute conflict tensor
    conflict_tensor = compute_conflict_tensor(distributions, objectives)
    pareto_scores = compute_pareto_alignment_score(conflict_tensor)
    
    # Select optimal k
    print(f"\n[6/7] Clustering...")
    optimal_k, k_scores = select_optimal_k(distance_matrix, config.max_clusters)
    
    # Final clustering
    sigma = np.median(distance_matrix[distance_matrix > 0]) + 1e-8
    similarity = np.exp(-distance_matrix / (2 * sigma ** 2))
    np.fill_diagonal(similarity, 1.0)
    
    clustering = SpectralClustering(
        n_clusters=optimal_k, affinity='precomputed',
        random_state=42, n_init=20
    )
    clusters = clustering.fit_predict(similarity)
    
    # Evaluate
    nmi = normalized_mutual_info_score(true_labels, clusters)
    ari = adjusted_rand_score(true_labels, clusters)
    sil = silhouette_score(distance_matrix, clusters, metric='precomputed')
    
    print(f"\n  Clustering results (k={optimal_k}):")
    print(f"    NMI: {nmi:.3f}")
    print(f"    ARI: {ari:.3f}")
    print(f"    Silhouette: {sil:.3f}")
    print(f"    Cluster sizes: {np.bincount(clusters)}")
    
    # Bootstrap stability
    print(f"\n[7/7] Bootstrap stability analysis...")
    stability = bootstrap_stability_analysis(distributions, objectives, optimal_k, config)
    print(f"  Stability score: {stability['stability_score']:.3f}")
    
    # Save results
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'mdp_configs': mdp_configs,
        'true_labels': true_labels.tolist(),
        'clusters': clusters.tolist(),
        'optimal_k': int(optimal_k),
        'k_scores': {str(k): float(v) for k, v in k_scores.items()},
        'nmi': float(nmi),
        'ari': float(ari),
        'silhouette': float(sil),
        'pareto_scores': pareto_scores.tolist(),
        'stability_score': stability['stability_score'],
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
        'objectives': [(float(a), float(b)) for a, b in objectives],
        'config': {
            'num_mdps': config.num_mdps,
            'total_objectives': len(objectives),
            'num_replicates': config.num_replicates,
            'num_steps': config.num_steps,
            'num_bootstrap': config.num_bootstrap
        }
    }
    
    with open(output_path / "wgdd_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save matrices
    np.save(output_path / "distance_matrix.npy", distance_matrix)
    np.save(output_path / "conflict_tensor.npy", conflict_tensor)
    np.save(output_path / "cooccurrence_matrix.npy", stability['cooccurrence_matrix'])
    
    # Visualizations
    create_wgdd_visualizations(
        distance_matrix, conflict_tensor, clusters, true_labels,
        mdp_configs, pareto_scores, k_scores, stability,
        pca, distributions, objectives, output_path
    )
    
    print("\n" + "="*80)
    print("WGDD COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_path}")
    
    return results


def create_wgdd_visualizations(
    distance_matrix, conflict_tensor, clusters, true_labels,
    mdp_configs, pareto_scores, k_scores, stability,
    pca, distributions, objectives, output_path
):
    """Create comprehensive visualizations."""
    
    edge_probs = np.array([cfg['edge_prob'] for cfg in mdp_configs])
    
    # Figure 1: Main results (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1a: Distance matrix heatmap
    ax = axes[0, 0]
    im = ax.imshow(distance_matrix, cmap='viridis')
    ax.set_title('Wasserstein Distance Matrix')
    ax.set_xlabel('MDP Index')
    ax.set_ylabel('MDP Index')
    plt.colorbar(im, ax=ax, label='Distance')
    
    # 1b: Clusters vs edge probability
    ax = axes[0, 1]
    for c in np.unique(clusters):
        mask = clusters == c
        ax.scatter(edge_probs[mask], np.zeros(mask.sum()) + c + np.random.randn(mask.sum())*0.05,
                   alpha=0.7, s=80, label=f'Cluster {c}')
    ax.set_xlabel('Edge Probability')
    ax.set_ylabel('Cluster')
    ax.set_title('Discovered Clusters vs DAG Structure')
    ax.legend()
    
    # 1c: Silhouette scores for different k
    ax = axes[1, 0]
    ks = sorted(k_scores.keys())
    scores = [k_scores[k] for k in ks]
    ax.bar([str(k) for k in ks], scores, color=PALETTE[5])
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Automatic k Selection')
    ax.axhline(y=max(scores), color='red', linestyle='--', alpha=0.5)
    
    # 1d: Pareto alignment scores
    ax = axes[1, 1]
    colors = [PALETTE[2] if l == 0 else PALETTE[6] for l in true_labels]
    ax.scatter(edge_probs, pareto_scores, c=colors, s=80, alpha=0.7)
    ax.set_xlabel('Edge Probability')
    ax.set_ylabel('Pareto Alignment (cos sim)')
    ax.set_title('Objective Alignment by MDP Structure')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path / "wgdd_main_results.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Conflict analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 2a: Mean conflict per MDP
    ax = axes[0]
    mean_conflict = np.mean(conflict_tensor, axis=(1, 2))
    ax.scatter(edge_probs, mean_conflict, c=colors, s=80, alpha=0.7)
    ax.set_xlabel('Edge Probability')
    ax.set_ylabel('Mean Gradient Conflict')
    ax.set_title('Gradient Conflict Profile')
    
    # 2b: Conflict heatmap for extremes
    ax = axes[1]
    # Average conflict between pure objectives across clusters
    cluster_conflict = []
    for c in np.unique(clusters):
        mask = clusters == c
        avg_conflict = conflict_tensor[mask, 0, 1].mean()
        cluster_conflict.append(avg_conflict)
    ax.bar([f'Cluster {c}' for c in np.unique(clusters)], cluster_conflict, color=PALETTE[:len(cluster_conflict)])
    ax.set_ylabel('Makespan-Energy Conflict')
    ax.set_title('Conflict by Cluster')
    
    # 2c: Bootstrap stability
    ax = axes[2]
    im = ax.imshow(stability['cooccurrence_matrix'], cmap='Blues')
    ax.set_title(f'Bootstrap Co-occurrence\n(Stability: {stability["stability_score"]:.3f})')
    ax.set_xlabel('MDP Index')
    ax.set_ylabel('MDP Index')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_path / "wgdd_conflict_analysis.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Embedding visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 3a: t-SNE of distance matrix
    ax = axes[0]
    tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=42, perplexity=min(10, len(distance_matrix)-1))
    embedding = tsne.fit_transform(distance_matrix)
    ax.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='tab10', s=80, alpha=0.8)
    ax.set_title('t-SNE of Wasserstein Distances (Predicted)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # 3b: Same but colored by true labels
    ax = axes[1]
    ax.scatter(embedding[:, 0], embedding[:, 1], c=true_labels, cmap='coolwarm', s=80, alpha=0.8)
    ax.set_title('t-SNE of Wasserstein Distances (True Labels)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig(output_path / "wgdd_embedding.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualizations to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Wasserstein Gradient Domain Discovery")
    parser.add_argument('--host-specs', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--num-mdps', type=int, default=40)
    parser.add_argument('--num-replicates', type=int, default=5)
    parser.add_argument('--num-bootstrap', type=int, default=20)
    parser.add_argument('--output-dir', type=str, default='logs/wgdd')
    
    args = parser.parse_args()
    
    config = WGDDConfig(
        num_mdps=args.num_mdps,
        num_replicates=args.num_replicates,
        num_bootstrap=args.num_bootstrap,
        output_dir=args.output_dir
    )
    
    results = wasserstein_domain_discovery(
        args.host_specs,
        args.model,
        config
    )
