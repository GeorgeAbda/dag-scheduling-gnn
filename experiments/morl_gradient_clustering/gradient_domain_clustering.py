"""
Gradient-based Domain Clustering for Multi-Objective Reinforcement Learning

This module implements a method for unsupervised domain clustering in MORL environments
by computing gradients of aggregate loss at randomly initialized policies and using
spectral clustering on the resulting similarity matrix.

Reference approach:
1. For each domain (MDP), compute gradient: g_i = ∇_θ(α*r1 + (1-α)*r2)
2. Build similarity matrix: S_ij = cos(g_i, g_j)
3. Apply spectral clustering to group domains
4. Evaluate clustering results
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


@dataclass
class ClusteringConfig:
    """Configuration for gradient-based clustering"""
    n_clusters: int = 3
    n_gradient_samples: int = 100
    alpha_values: List[float] = None
    random_seed: int = 42
    policy_hidden_dims: List[int] = None
    
    def __post_init__(self):
        if self.alpha_values is None:
            self.alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        if self.policy_hidden_dims is None:
            self.policy_hidden_dims = [64, 64]


class SimplePolicy(nn.Module):
    """Simple feedforward policy network for gradient computation"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs):
        return torch.tanh(self.network(obs))


class GradientDomainClustering:
    """
    Main class for gradient-based domain clustering in MORL
    """
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        
        self.domain_gradients = {}
        self.similarity_matrix = None
        self.cluster_labels = None
        
    def compute_policy_gradient(
        self,
        env,
        policy: nn.Module,
        alpha: float,
        n_samples: int = 100
    ) -> np.ndarray:
        """
        Compute gradient of aggregate loss: ∇_θ(α*r1 + (1-α)*r2)
        
        Args:
            env: MO-Gymnasium environment
            policy: Policy network
            alpha: Trade-off parameter between objectives
            n_samples: Number of trajectory samples
            
        Returns:
            Flattened gradient vector
        """
        policy.train()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        
        total_loss = 0.0
        gradients = []
        
        for _ in range(n_samples):
            obs, _ = env.reset()
            done = False
            episode_rewards = []
            
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = policy(obs_tensor)
                
                obs, reward, terminated, truncated, info = env.step(action.detach().numpy()[0])
                done = terminated or truncated
                
                # Reward is a vector [r1, r2, ...]
                episode_rewards.append(reward)
            
            # Compute aggregate reward
            episode_rewards = np.array(episode_rewards)
            if len(episode_rewards.shape) == 1:
                episode_rewards = episode_rewards.reshape(-1, 1)
            
            # Aggregate loss (negative reward for gradient ascent)
            r1 = episode_rewards[:, 0].sum()
            r2 = episode_rewards[:, 1].sum() if episode_rewards.shape[1] > 1 else 0
            aggregate_reward = alpha * r1 + (1 - alpha) * r2
            loss = -aggregate_reward
            
            # Compute gradient
            optimizer.zero_grad()
            loss_tensor = torch.tensor(loss, requires_grad=True)
            
            # Backpropagate through policy
            for param in policy.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            # Manual gradient computation
            obs, _ = env.reset()
            done = False
            log_probs = []
            rewards_collected = []
            
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = policy(obs_tensor)
                
                # Compute log probability (simplified for continuous actions)
                log_prob = -0.5 * torch.sum(action ** 2)
                log_probs.append(log_prob)
                
                obs, reward, terminated, truncated, info = env.step(action.detach().numpy()[0])
                done = terminated or truncated
                rewards_collected.append(reward)
            
            # Policy gradient
            rewards_collected = np.array(rewards_collected)
            if len(rewards_collected.shape) == 1:
                rewards_collected = rewards_collected.reshape(-1, 1)
            
            r1 = rewards_collected[:, 0].sum()
            r2 = rewards_collected[:, 1].sum() if rewards_collected.shape[1] > 1 else 0
            aggregate_return = alpha * r1 + (1 - alpha) * r2
            
            policy_loss = -sum(log_probs) * aggregate_return
            policy_loss.backward()
            
            # Collect gradients
            grad_vector = []
            for param in policy.parameters():
                if param.grad is not None:
                    grad_vector.append(param.grad.detach().cpu().numpy().flatten())
            
            if len(grad_vector) > 0:
                gradients.append(np.concatenate(grad_vector))
        
        # Average gradients
        if len(gradients) > 0:
            avg_gradient = np.mean(gradients, axis=0)
        else:
            # Fallback: use random gradient
            avg_gradient = np.random.randn(sum(p.numel() for p in policy.parameters()))
        
        return avg_gradient
    
    def compute_domain_gradients(
        self,
        domains: Dict[str, any],
        obs_dim: int,
        action_dim: int
    ) -> Dict[str, np.ndarray]:
        """
        Compute gradients for all domains across different alpha values
        
        Args:
            domains: Dictionary of domain_name -> environment
            obs_dim: Observation dimension
            action_dim: Action dimension
            
        Returns:
            Dictionary mapping domain names to concatenated gradient vectors
        """
        print(f"Computing gradients for {len(domains)} domains...")
        
        for domain_name, env in domains.items():
            print(f"Processing domain: {domain_name}")
            
            # Initialize random policy
            policy = SimplePolicy(obs_dim, action_dim, self.config.policy_hidden_dims)
            
            # Compute gradients for different alpha values
            domain_grad_vectors = []
            
            for alpha in self.config.alpha_values:
                print(f"  Computing gradient for α={alpha:.2f}")
                grad = self.compute_policy_gradient(
                    env, policy, alpha, self.config.n_gradient_samples
                )
                domain_grad_vectors.append(grad)
            
            # Concatenate gradients across different alpha values
            self.domain_gradients[domain_name] = np.concatenate(domain_grad_vectors)
        
        return self.domain_gradients
    
    def build_similarity_matrix(self) -> np.ndarray:
        """
        Build cosine similarity matrix: S_ij = cos(g_i, g_j)
        
        Returns:
            Similarity matrix of shape (n_domains, n_domains)
        """
        domain_names = list(self.domain_gradients.keys())
        n_domains = len(domain_names)
        
        similarity_matrix = np.zeros((n_domains, n_domains))
        
        for i, name_i in enumerate(domain_names):
            for j, name_j in enumerate(domain_names):
                grad_i = self.domain_gradients[name_i]
                grad_j = self.domain_gradients[name_j]
                
                # Cosine similarity
                similarity = np.dot(grad_i, grad_j) / (
                    np.linalg.norm(grad_i) * np.linalg.norm(grad_j) + 1e-8
                )
                similarity_matrix[i, j] = similarity
        
        self.similarity_matrix = similarity_matrix
        return similarity_matrix
    
    def apply_spectral_clustering(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Apply spectral clustering to the similarity matrix
        
        Args:
            n_clusters: Number of clusters (uses config value if None)
            
        Returns:
            Cluster labels for each domain
        """
        if self.similarity_matrix is None:
            raise ValueError("Must build similarity matrix first")
        
        if n_clusters is None:
            n_clusters = self.config.n_clusters
        
        # Convert similarity to affinity (ensure non-negative)
        affinity_matrix = (self.similarity_matrix + 1) / 2
        
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=self.config.random_seed
        )
        
        self.cluster_labels = clustering.fit_predict(affinity_matrix)
        return self.cluster_labels
    
    def evaluate_clustering(
        self,
        true_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate clustering results
        
        Args:
            true_labels: Ground truth labels (if available)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.cluster_labels is None:
            raise ValueError("Must perform clustering first")
        
        metrics = {}
        
        # Silhouette score (intrinsic quality)
        if len(np.unique(self.cluster_labels)) > 1:
            # Convert similarity to distance
            distance_matrix = 1 - (self.similarity_matrix + 1) / 2
            # Ensure diagonal is exactly zero
            np.fill_diagonal(distance_matrix, 0)
            
            metrics['silhouette_score'] = silhouette_score(
                distance_matrix,
                self.cluster_labels,
                metric='precomputed'
            )
        
        # External validation (if true labels provided)
        if true_labels is not None:
            metrics['adjusted_rand_index'] = adjusted_rand_score(true_labels, self.cluster_labels)
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(
                true_labels, self.cluster_labels
            )
        
        return metrics
    
    def visualize_results(
        self,
        domain_names: List[str],
        save_path: Optional[str] = None
    ):
        """
        Visualize clustering results
        
        Args:
            domain_names: List of domain names
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot similarity matrix
        sns.heatmap(
            self.similarity_matrix,
            xticklabels=domain_names,
            yticklabels=domain_names,
            cmap='coolwarm',
            center=0,
            ax=axes[0],
            cbar_kws={'label': 'Cosine Similarity'}
        )
        axes[0].set_title('Domain Similarity Matrix', fontsize=14, fontweight='bold')
        
        # Plot cluster assignments
        cluster_matrix = np.zeros((len(domain_names), self.config.n_clusters))
        for i, label in enumerate(self.cluster_labels):
            cluster_matrix[i, label] = 1
        
        sns.heatmap(
            cluster_matrix,
            xticklabels=[f'Cluster {i}' for i in range(self.config.n_clusters)],
            yticklabels=domain_names,
            cmap='Blues',
            ax=axes[1],
            cbar=False
        )
        axes[1].set_title('Cluster Assignments', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def get_cluster_summary(self, domain_names: List[str]) -> Dict[int, List[str]]:
        """
        Get summary of which domains belong to which clusters
        
        Args:
            domain_names: List of domain names
            
        Returns:
            Dictionary mapping cluster IDs to lists of domain names
        """
        cluster_summary = defaultdict(list)
        
        for domain_name, cluster_id in zip(domain_names, self.cluster_labels):
            cluster_summary[cluster_id].append(domain_name)
        
        return dict(cluster_summary)


def create_synthetic_morl_domains(n_domains: int = 6) -> Dict[str, Dict]:
    """
    Create synthetic MORL domains with different reward structures
    
    Args:
        n_domains: Number of domains to create
        
    Returns:
        Dictionary of domain configurations
    """
    domains = {}
    
    # Create domains with different reward trade-offs
    for i in range(n_domains):
        domain_config = {
            'name': f'domain_{i}',
            'reward_weights': np.random.dirichlet([1, 1]),  # Random weights for 2 objectives
            'difficulty': np.random.choice(['easy', 'medium', 'hard']),
            'true_cluster': i // 2  # Group domains in pairs
        }
        domains[f'domain_{i}'] = domain_config
    
    return domains
