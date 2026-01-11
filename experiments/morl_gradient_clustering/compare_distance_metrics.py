"""
Compare different distance metrics for gradient-based domain clustering.

Distance metrics compared:
1. Sliced Wasserstein Distance (OT-based, distributional)
2. Wasserstein-2 Distance (exact OT, distributional)
3. Cosine Similarity of Mean Gradients (point-based)
4. Euclidean Distance of Mean Gradients (point-based)
5. Maximum Mean Discrepancy (MMD) with RBF kernel
6. Energy Distance (distributional)
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mo_gymnasium as mo_gym
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from collections import defaultdict
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'


# =============================================================================
# DISTANCE METRICS
# =============================================================================

def sliced_wasserstein_distance(X: np.ndarray, Y: np.ndarray, n_projections: int = 100) -> float:
    """Sliced Wasserstein distance between two point clouds."""
    d = X.shape[1]
    
    # Normalize to unit vectors
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    
    # Random projections
    np.random.seed(42)  # Fixed seed for reproducibility
    projections = np.random.randn(n_projections, d)
    projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)
    
    distances = []
    for proj in projections:
        X_proj = X_norm @ proj
        Y_proj = Y_norm @ proj
        
        # 1D Wasserstein
        distances.append(wasserstein_distance(X_proj, Y_proj))
    
    return np.mean(distances)


def cosine_distance_of_means(X: np.ndarray, Y: np.ndarray) -> float:
    """Cosine distance between mean gradients."""
    mean_X = np.mean(X, axis=0)
    mean_Y = np.mean(Y, axis=0)
    
    norm_X = np.linalg.norm(mean_X)
    norm_Y = np.linalg.norm(mean_Y)
    
    if norm_X < 1e-8 or norm_Y < 1e-8:
        return 1.0
    
    cosine_sim = np.dot(mean_X, mean_Y) / (norm_X * norm_Y)
    return 1 - cosine_sim  # Convert to distance


def euclidean_distance_of_means(X: np.ndarray, Y: np.ndarray) -> float:
    """Euclidean distance between mean gradients."""
    mean_X = np.mean(X, axis=0)
    mean_Y = np.mean(Y, axis=0)
    return np.linalg.norm(mean_X - mean_Y)


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    """Maximum Mean Discrepancy with RBF kernel."""
    if gamma is None:
        # Use median heuristic
        combined = np.vstack([X, Y])
        pairwise_dists = cdist(combined, combined, 'sqeuclidean')
        gamma = 1.0 / np.median(pairwise_dists[pairwise_dists > 0])
    
    # Compute kernel matrices
    K_XX = np.exp(-gamma * cdist(X, X, 'sqeuclidean'))
    K_YY = np.exp(-gamma * cdist(Y, Y, 'sqeuclidean'))
    K_XY = np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))
    
    # MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
    mmd_sq = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return np.sqrt(max(0, mmd_sq))


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Energy distance between two distributions."""
    # Normalize first
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    
    # E[||X - Y||] - 0.5*E[||X - X'||] - 0.5*E[||Y - Y'||]
    D_XY = cdist(X_norm, Y_norm, 'euclidean').mean()
    D_XX = cdist(X_norm, X_norm, 'euclidean').mean()
    D_YY = cdist(Y_norm, Y_norm, 'euclidean').mean()
    
    return 2 * D_XY - D_XX - D_YY


def frechet_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Frechet distance (like FID but for gradients)."""
    # Compute means
    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(Y, axis=0)
    
    # Compute covariances (use diagonal for stability)
    var_X = np.var(X, axis=0)
    var_Y = np.var(Y, axis=0)
    
    # Frechet distance with diagonal covariance
    mean_diff = np.sum((mu_X - mu_Y) ** 2)
    cov_term = np.sum(var_X + var_Y - 2 * np.sqrt(var_X * var_Y + 1e-8))
    
    return np.sqrt(mean_diff + cov_term)


# =============================================================================
# ENVIRONMENT WRAPPERS
# =============================================================================

class RewardScaleWrapper(gym.Wrapper):
    def __init__(self, env, obj1_scale: float = 1.0, obj2_scale: float = 1.0):
        super().__init__(env)
        self.obj1_scale = obj1_scale
        self.obj2_scale = obj2_scale
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(reward, np.ndarray) and len(reward) >= 2:
            reward = np.array([reward[0] * self.obj1_scale, reward[1] * self.obj2_scale])
        return obs, reward, terminated, truncated, info


# =============================================================================
# POLICY AND GRADIENT COLLECTION
# =============================================================================

class SharedPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, obs):
        return torch.softmax(self.net(obs), dim=-1)
    
    def sample_action(self, obs):
        probs = self.forward(obs)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)


def get_env_dims(env):
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs_dim = sum(np.array(v).flatten().shape[0] for v in obs.values())
    else:
        obs_dim = np.array(obs).flatten().shape[0]
    n_actions = env.action_space.n
    return obs_dim, n_actions


def collect_trajectory_gradient(env, policy, alpha: float, max_steps: int = 200):
    obs, _ = env.reset()
    done = False
    log_probs = []
    rewards_obj1 = []
    rewards_obj2 = []
    step_count = 0
    
    while not done and step_count < max_steps:
        if isinstance(obs, dict):
            obs = np.concatenate([v.flatten() for v in obs.values()])
        obs = np.array(obs).flatten()
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action, log_prob = policy.sample_action(obs_tensor)
        log_probs.append(log_prob)
        
        obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        step_count += 1
        
        if isinstance(reward, np.ndarray):
            rewards_obj1.append(reward[0])
            rewards_obj2.append(reward[1] if len(reward) > 1 else 0)
        else:
            rewards_obj1.append(reward)
            rewards_obj2.append(0)
    
    if len(log_probs) == 0:
        return None, None
    
    total_r1 = sum(rewards_obj1)
    total_r2 = sum(rewards_obj2)
    scalarized_return = alpha * total_r1 + (1 - alpha) * total_r2
    
    policy_loss = -sum(log_probs) * scalarized_return
    policy.zero_grad()
    policy_loss.backward()
    
    grad_vector = []
    for param in policy.parameters():
        if param.grad is not None:
            grad_vector.append(param.grad.detach().cpu().numpy().flatten())
    
    if len(grad_vector) == 0:
        return None, None
    
    grad = np.concatenate(grad_vector)
    info = {'r1': total_r1, 'r2': total_r2, 'steps': step_count}
    
    return grad, info


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_distance_comparison():
    """Compare different distance metrics for domain clustering."""
    
    output_dir = 'results/distance_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("COMPARING DISTANCE METRICS FOR GRADIENT-BASED CLUSTERING")
    print("="*80)
    
    # Configuration
    base_env_id = 'deep-sea-treasure-v0'
    n_instances_per_domain = 8
    n_trajectories_per_instance = 30
    alpha = 0.5
    seed = 42
    
    # Domain configurations
    domain_configs = {
        'balanced': {'obj1_scale': 1.0, 'obj2_scale': 1.0},
        'treasure_heavy': {'obj1_scale': 5.0, 'obj2_scale': 0.2},
        'time_heavy': {'obj1_scale': 0.2, 'obj2_scale': 5.0},
    }
    
    # Distance metrics to compare
    distance_metrics = {
        'Sliced Wasserstein': sliced_wasserstein_distance,
        'Cosine (Mean)': cosine_distance_of_means,
        'Euclidean (Mean)': euclidean_distance_of_means,
        'MMD (RBF)': mmd_rbf,
        'Energy Distance': energy_distance,
        'Frechet': frechet_distance,
    }
    
    # Get environment dimensions
    base_env = mo_gym.make(base_env_id)
    obs_dim, n_actions = get_env_dims(base_env)
    base_env.close()
    
    print(f"\nBase environment: {base_env_id}")
    print(f"Instances per domain: {n_instances_per_domain}")
    print(f"Trajectories per instance: {n_trajectories_per_instance}")
    print(f"Distance metrics: {list(distance_metrics.keys())}")
    
    # Create shared policy
    torch.manual_seed(seed)
    shared_policy = SharedPolicy(obs_dim, n_actions)
    
    # Collect gradients
    print("\n" + "-"*80)
    print("COLLECTING GRADIENTS")
    print("-"*80)
    
    all_gradients = {}
    all_infos = {}
    true_labels = []
    domain_names = []
    domain_type_to_id = {name: i for i, name in enumerate(domain_configs.keys())}
    
    for domain_type, config in domain_configs.items():
        print(f"\nDomain: {domain_type}")
        
        for instance_idx in range(n_instances_per_domain):
            instance_name = f"{domain_type}_{instance_idx}"
            domain_names.append(instance_name)
            true_labels.append(domain_type_to_id[domain_type])
            
            # Create environment
            env = mo_gym.make(base_env_id)
            env = RewardScaleWrapper(env, config['obj1_scale'], config['obj2_scale'])
            
            gradients = []
            infos = []
            
            for traj_idx in range(n_trajectories_per_instance):
                np.random.seed((seed + abs(hash(instance_name)) + traj_idx) % (2**32 - 1))
                grad, info = collect_trajectory_gradient(env, shared_policy, alpha)
                
                if grad is not None:
                    gradients.append(grad)
                    infos.append(info)
            
            env.close()
            
            all_gradients[instance_name] = np.array(gradients)
            all_infos[instance_name] = infos
            
            r1_mean = np.mean([info['r1'] for info in infos])
            r2_mean = np.mean([info['r2'] for info in infos])
            print(f"  {instance_name}: {len(gradients)} grads, R1={r1_mean:.2f}, R2={r2_mean:.2f}")
    
    true_labels = np.array(true_labels)
    n_domains = len(domain_names)
    n_clusters = len(domain_configs)
    
    # Compute distance matrices for each metric
    print("\n" + "-"*80)
    print("COMPUTING DISTANCE MATRICES")
    print("-"*80)
    
    results = {}
    distance_matrices = {}
    
    for metric_name, metric_fn in distance_metrics.items():
        print(f"\n{metric_name}...")
        
        distance_matrix = np.zeros((n_domains, n_domains))
        n_pairs = n_domains * (n_domains - 1) // 2
        
        with tqdm(total=n_pairs, desc=f"  {metric_name}") as pbar:
            for i in range(n_domains):
                for j in range(i + 1, n_domains):
                    dist = metric_fn(all_gradients[domain_names[i]], 
                                    all_gradients[domain_names[j]])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
                    pbar.update(1)
        
        distance_matrices[metric_name] = distance_matrix
        
        # Clustering
        max_dist = distance_matrix.max()
        if max_dist > 0:
            similarity_matrix = 1 - distance_matrix / (max_dist + 1e-8)
        else:
            similarity_matrix = 1 - distance_matrix
        
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', 
                                        random_state=seed)
        predicted_labels = clustering.fit_predict(similarity_matrix)
        
        # Metrics
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        
        try:
            sil = silhouette_score(distance_matrix, predicted_labels, metric='precomputed')
        except:
            sil = 0.0
        
        # Purity
        purity = 0
        for cluster_id in np.unique(predicted_labels):
            cluster_mask = predicted_labels == cluster_id
            if cluster_mask.sum() > 0:
                true_in_cluster = true_labels[cluster_mask]
                most_common = np.bincount(true_in_cluster).max()
                purity += most_common
        purity = purity / len(true_labels) * 100
        
        # Separation ratio
        within_dists = []
        cross_dists = []
        for i in range(n_domains):
            for j in range(i + 1, n_domains):
                if true_labels[i] == true_labels[j]:
                    within_dists.append(distance_matrix[i, j])
                else:
                    cross_dists.append(distance_matrix[i, j])
        
        separation_ratio = np.mean(cross_dists) / (np.mean(within_dists) + 1e-8)
        
        results[metric_name] = {
            'ari': ari,
            'nmi': nmi,
            'silhouette': sil,
            'purity': purity,
            'separation_ratio': separation_ratio,
            'predicted_labels': predicted_labels,
            'within_dist': np.mean(within_dists),
            'cross_dist': np.mean(cross_dists),
        }
        
        print(f"    ARI: {ari:.4f}, NMI: {nmi:.4f}, Purity: {purity:.1f}%, Sep: {separation_ratio:.2f}")
    
    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    
    print("\n" + "-"*80)
    print("GENERATING VISUALIZATIONS")
    print("-"*80)
    
    # Figure 1: Distance matrices comparison
    n_metrics = len(distance_metrics)
    fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (metric_name, dist_matrix) in enumerate(distance_matrices.items()):
        ax = axes[idx]
        
        # Sort by true labels
        sort_idx = np.argsort(true_labels)
        sorted_dist = dist_matrix[np.ix_(sort_idx, sort_idx)]
        
        sns.heatmap(sorted_dist, cmap='viridis_r', ax=ax, cbar_kws={'label': 'Distance'})
        
        res = results[metric_name]
        ax.set_title(f'{metric_name}\nARI={res["ari"]:.3f}, Purity={res["purity"]:.1f}%', 
                    fontweight='bold', fontsize=10)
        
        # Add domain boundaries
        boundaries = np.cumsum([n_instances_per_domain] * n_clusters)[:-1]
        for b in boundaries:
            ax.axhline(y=b, color='white', linewidth=2)
            ax.axvline(x=b, color='white', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distance_matrices_comparison.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir}/distance_matrices_comparison.png")
    
    # Figure 2: Metrics comparison bar chart
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metric_names = list(results.keys())
    x = np.arange(len(metric_names))
    
    # ARI
    ax = axes[0, 0]
    ari_vals = [results[m]['ari'] for m in metric_names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metric_names)))
    bars = ax.bar(x, ari_vals, color=colors, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in metric_names], fontsize=9)
    ax.set_ylabel('ARI')
    ax.set_title('(a) Adjusted Rand Index', fontweight='bold')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, ari_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
               f'{val:.3f}', ha='center', fontsize=8)
    
    # NMI
    ax = axes[0, 1]
    nmi_vals = [results[m]['nmi'] for m in metric_names]
    bars = ax.bar(x, nmi_vals, color=colors, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in metric_names], fontsize=9)
    ax.set_ylabel('NMI')
    ax.set_title('(b) Normalized Mutual Information', fontweight='bold')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, nmi_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
               f'{val:.3f}', ha='center', fontsize=8)
    
    # Purity
    ax = axes[1, 0]
    purity_vals = [results[m]['purity'] for m in metric_names]
    bars = ax.bar(x, purity_vals, color=colors, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in metric_names], fontsize=9)
    ax.set_ylabel('Purity (%)')
    ax.set_title('(c) Cluster Purity', fontweight='bold')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, purity_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{val:.1f}%', ha='center', fontsize=8)
    
    # Separation Ratio
    ax = axes[1, 1]
    sep_vals = [results[m]['separation_ratio'] for m in metric_names]
    bars = ax.bar(x, sep_vals, color=colors, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in metric_names], fontsize=9)
    ax.set_ylabel('Separation Ratio')
    ax.set_title('(d) Cross/Within Distance Ratio', fontweight='bold')
    ax.axhline(y=1, color='red', linestyle='--', label='No separation')
    for bar, val in zip(bars, sep_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
               f'{val:.2f}', ha='center', fontsize=8)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir}/metrics_comparison.png")
    
    # Figure 3: Summary table
    fig3, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Distance Metric', 'ARI', 'NMI', 'Purity', 'Silhouette', 'Sep. Ratio', 'Rank']
    
    # Compute ranks based on ARI
    ari_ranks = np.argsort([results[m]['ari'] for m in metric_names])[::-1]
    rank_map = {metric_names[i]: rank + 1 for rank, i in enumerate(ari_ranks)}
    
    for metric_name in metric_names:
        res = results[metric_name]
        table_data.append([
            metric_name,
            f"{res['ari']:.4f}",
            f"{res['nmi']:.4f}",
            f"{res['purity']:.1f}%",
            f"{res['silhouette']:.4f}",
            f"{res['separation_ratio']:.2f}",
            f"#{rank_map[metric_name]}"
        ])
    
    # Sort by ARI
    table_data.sort(key=lambda x: float(x[1]), reverse=True)
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                    cellLoc='center', colColours=['lightblue']*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Highlight best row
    for j in range(len(headers)):
        table[(1, j)].set_facecolor('lightgreen')
    
    ax.set_title('Distance Metrics Comparison for Gradient-Based Domain Clustering\n(Sorted by ARI)', 
                fontweight='bold', fontsize=14, pad=20)
    
    plt.savefig(f'{output_dir}/summary_table.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir}/summary_table.png")
    
    # Save to paper figures
    paper_dir = 'paper/figures'
    os.makedirs(paper_dir, exist_ok=True)
    fig2.savefig(f'{paper_dir}/distance_metrics_comparison.png', dpi=300, 
                 bbox_inches='tight', facecolor='white')
    print(f"Saved: {paper_dir}/distance_metrics_comparison.png")
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL RESULTS (sorted by ARI)")
    print("="*80)
    print(f"\n{'Metric':<25} {'ARI':>10} {'NMI':>10} {'Purity':>10} {'Sep.Ratio':>12}")
    print("-" * 70)
    
    sorted_metrics = sorted(results.items(), key=lambda x: x[1]['ari'], reverse=True)
    for metric_name, res in sorted_metrics:
        print(f"{metric_name:<25} {res['ari']:>10.4f} {res['nmi']:>10.4f} "
              f"{res['purity']:>9.1f}% {res['separation_ratio']:>12.2f}")
    
    best_metric = sorted_metrics[0][0]
    print(f"\n*** BEST METRIC: {best_metric} ***")
    
    return results, distance_matrices


if __name__ == '__main__':
    results, distance_matrices = run_distance_comparison()
