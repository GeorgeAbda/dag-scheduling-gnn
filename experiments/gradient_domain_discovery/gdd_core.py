"""
Gradient Domain Discovery (GDD) - Core Implementation

This module implements the theoretical framework from:
"Gradient-Based Domain Discovery: A Theoretical Framework for 
Unsupervised Domain Identification in Multi-Source Learning"

Key components:
1. Gradient collection and storage
2. Similarity matrix computation (cosine, conflict rate, subspace)
3. Domain clustering via spectral methods
4. Validation metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.linalg import svd
import warnings


@dataclass
class GradientSample:
    """A single gradient sample from a data source."""
    source_id: int
    gradient: np.ndarray  # Flattened gradient vector
    step: int  # Training step when collected
    loss: float  # Loss value at this point


@dataclass
class DomainDiscoveryResult:
    """Results from domain discovery."""
    num_sources: int
    num_domains: int
    domain_assignments: np.ndarray  # Shape: (num_sources,)
    similarity_matrix: np.ndarray  # Shape: (num_sources, num_sources)
    conflict_matrix: np.ndarray  # Shape: (num_sources, num_sources)
    silhouette_score: float
    cluster_sizes: List[int]
    
    def summary(self) -> str:
        lines = [
            f"Domain Discovery Results",
            f"  Sources: {self.num_sources}",
            f"  Discovered domains: {self.num_domains}",
            f"  Silhouette score: {self.silhouette_score:.4f}",
            f"  Cluster sizes: {self.cluster_sizes}",
            f"  Assignments: {self.domain_assignments.tolist()}",
        ]
        return "\n".join(lines)


class GradientCollector:
    """
    Collects and stores gradients from multiple data sources.
    
    Usage:
        collector = GradientCollector(num_sources=4)
        
        # During training:
        for source_id, batch in enumerate(batches):
            loss = compute_loss(model, batch)
            loss.backward()
            collector.collect(source_id, model, step, loss.item())
            optimizer.step()
            optimizer.zero_grad()
        
        # After collection:
        gradients = collector.get_gradients()
    """
    
    def __init__(
        self,
        num_sources: int,
        max_samples_per_source: int = 100,
        gradient_dim: Optional[int] = None,
        use_random_projection: bool = False,
        projection_dim: int = 1024,
        layers_to_track: Optional[List[str]] = None,
    ):
        """
        Args:
            num_sources: Number of data sources
            max_samples_per_source: Maximum gradient samples to store per source
            gradient_dim: If set, truncate/pad gradients to this dimension
            use_random_projection: If True, project gradients to lower dimension
            projection_dim: Dimension for random projection
            layers_to_track: If set, only track gradients from these layers
        """
        self.num_sources = num_sources
        self.max_samples = max_samples_per_source
        self.gradient_dim = gradient_dim
        self.use_random_projection = use_random_projection
        self.projection_dim = projection_dim
        self.layers_to_track = layers_to_track
        
        # Storage
        self.samples: Dict[int, List[GradientSample]] = {
            i: [] for i in range(num_sources)
        }
        
        # Random projection matrix (initialized lazily)
        self._projection_matrix: Optional[np.ndarray] = None
        self._full_dim: Optional[int] = None
    
    def _flatten_gradients(
        self, 
        model: nn.Module,
    ) -> np.ndarray:
        """Extract and flatten gradients from model parameters."""
        grads = []
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            # Filter by layer name if specified
            if self.layers_to_track is not None:
                if not any(layer in name for layer in self.layers_to_track):
                    continue
            
            grads.append(param.grad.detach().cpu().numpy().flatten())
        
        if not grads:
            raise ValueError("No gradients found. Did you call backward()?")
        
        flat_grad = np.concatenate(grads)
        
        # Initialize projection matrix if needed
        if self.use_random_projection:
            if self._projection_matrix is None:
                self._full_dim = len(flat_grad)
                # Random Gaussian projection (preserves cosine similarity)
                self._projection_matrix = np.random.randn(
                    self.projection_dim, self._full_dim
                ) / np.sqrt(self.projection_dim)
            
            flat_grad = self._projection_matrix @ flat_grad
        
        return flat_grad
    
    def collect(
        self,
        source_id: int,
        model: nn.Module,
        step: int,
        loss: float,
    ) -> None:
        """
        Collect gradient from model for a given source.
        
        Args:
            source_id: ID of the data source (0 to num_sources-1)
            model: The model with gradients computed
            step: Current training step
            loss: Loss value
        """
        if source_id < 0 or source_id >= self.num_sources:
            raise ValueError(f"Invalid source_id: {source_id}")
        
        gradient = self._flatten_gradients(model)
        
        sample = GradientSample(
            source_id=source_id,
            gradient=gradient,
            step=step,
            loss=loss,
        )
        
        # Add to storage (with capacity limit)
        if len(self.samples[source_id]) >= self.max_samples:
            # Remove oldest sample
            self.samples[source_id].pop(0)
        
        self.samples[source_id].append(sample)
    
    def get_gradients(self, source_id: int) -> np.ndarray:
        """Get all gradient samples for a source as a matrix."""
        samples = self.samples[source_id]
        if not samples:
            raise ValueError(f"No samples for source {source_id}")
        return np.stack([s.gradient for s in samples])
    
    def get_mean_gradient(self, source_id: int) -> np.ndarray:
        """Get mean gradient for a source."""
        return self.get_gradients(source_id).mean(axis=0)
    
    def num_samples(self, source_id: int) -> int:
        """Number of samples collected for a source."""
        return len(self.samples[source_id])
    
    def total_samples(self) -> int:
        """Total samples across all sources."""
        return sum(len(s) for s in self.samples.values())
    
    def clear(self) -> None:
        """Clear all collected samples."""
        for source_id in self.samples:
            self.samples[source_id] = []


class GradientDomainDiscovery:
    """
    Main class for gradient-based domain discovery.
    
    Implements the GDD algorithm from the theoretical framework.
    """
    
    def __init__(
        self,
        similarity_metric: str = "cosine",  # "cosine", "conflict", "subspace"
        clustering_method: str = "spectral",  # "spectral", "kmeans"
        num_domains: Optional[int] = None,  # If None, auto-select
        max_domains: int = 10,
    ):
        """
        Args:
            similarity_metric: How to compute source similarity
            clustering_method: Clustering algorithm to use
            num_domains: Number of domains (None for auto-selection)
            max_domains: Maximum domains to consider for auto-selection
        """
        self.similarity_metric = similarity_metric
        self.clustering_method = clustering_method
        self.num_domains = num_domains
        self.max_domains = max_domains
    
    def compute_similarity_matrix(
        self,
        collector: GradientCollector,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pairwise similarity and conflict matrices.
        
        Returns:
            similarity_matrix: Shape (K, K), values in [-1, 1]
            conflict_matrix: Shape (K, K), values in [0, 1]
        """
        K = collector.num_sources
        similarity = np.zeros((K, K))
        conflict = np.zeros((K, K))
        
        for i in range(K):
            for j in range(i, K):
                sim, conf = self._compute_pairwise_metrics(
                    collector.get_gradients(i),
                    collector.get_gradients(j),
                )
                similarity[i, j] = similarity[j, i] = sim
                conflict[i, j] = conflict[j, i] = conf
        
        return similarity, conflict
    
    def _compute_pairwise_metrics(
        self,
        grads_i: np.ndarray,  # Shape: (N_i, d)
        grads_j: np.ndarray,  # Shape: (N_j, d)
    ) -> Tuple[float, float]:
        """Compute similarity and conflict rate between two sources."""
        
        if self.similarity_metric == "cosine":
            # Mean cosine similarity
            similarities = []
            conflicts = 0
            total = 0
            
            for g_i in grads_i:
                for g_j in grads_j:
                    norm_i = np.linalg.norm(g_i)
                    norm_j = np.linalg.norm(g_j)
                    
                    if norm_i < 1e-10 or norm_j < 1e-10:
                        continue
                    
                    cos_sim = np.dot(g_i, g_j) / (norm_i * norm_j)
                    similarities.append(cos_sim)
                    
                    if cos_sim < 0:
                        conflicts += 1
                    total += 1
            
            mean_sim = np.mean(similarities) if similarities else 0.0
            conflict_rate = conflicts / total if total > 0 else 0.0
            
            return mean_sim, conflict_rate
        
        elif self.similarity_metric == "subspace":
            # Subspace similarity via principal angles
            return self._subspace_similarity(grads_i, grads_j)
        
        else:
            raise ValueError(f"Unknown metric: {self.similarity_metric}")
    
    def _subspace_similarity(
        self,
        grads_i: np.ndarray,
        grads_j: np.ndarray,
        rank: int = 5,
    ) -> Tuple[float, float]:
        """Compute subspace similarity via principal angles."""
        # SVD to get principal directions
        U_i, _, _ = svd(grads_i.T, full_matrices=False)
        U_j, _, _ = svd(grads_j.T, full_matrices=False)
        
        # Keep top-k directions
        k = min(rank, U_i.shape[1], U_j.shape[1])
        U_i = U_i[:, :k]
        U_j = U_j[:, :k]
        
        # Principal angles via SVD of U_i^T @ U_j
        _, sigmas, _ = svd(U_i.T @ U_j)
        
        # Similarity = mean of cosines of principal angles
        cos_angles = np.clip(sigmas, -1, 1)
        similarity = np.mean(cos_angles)
        
        # Conflict = fraction of angles > 90 degrees (cos < 0)
        conflict = np.mean(cos_angles < 0)
        
        return similarity, conflict
    
    def discover_domains(
        self,
        collector: GradientCollector,
        true_labels: Optional[np.ndarray] = None,
    ) -> DomainDiscoveryResult:
        """
        Discover domains from collected gradients.
        
        Args:
            collector: GradientCollector with samples
            true_labels: Ground truth domain labels (for evaluation only)
        
        Returns:
            DomainDiscoveryResult with discovered domains
        """
        # Compute similarity matrix
        similarity, conflict = self.compute_similarity_matrix(collector)
        
        # Determine number of domains
        if self.num_domains is not None:
            n_clusters = self.num_domains
        else:
            n_clusters = self._auto_select_domains(similarity)
        
        # Cluster sources
        if self.clustering_method == "spectral":
            # Convert similarity to affinity (shift to positive)
            affinity = (similarity + 1) / 2  # Map [-1, 1] to [0, 1]
            
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                random_state=42,
            )
            labels = clustering.fit_predict(affinity)
        
        elif self.clustering_method == "kmeans":
            # Use mean gradients as features
            features = np.stack([
                collector.get_mean_gradient(i) 
                for i in range(collector.num_sources)
            ])
            
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clustering.fit_predict(features)
        
        else:
            raise ValueError(f"Unknown clustering: {self.clustering_method}")
        
        # Compute silhouette score
        if n_clusters > 1 and n_clusters < collector.num_sources:
            # Convert similarity to distance for silhouette
            # Distance = 1 - similarity (maps [-1,1] to [2,0])
            distance = 1 - similarity
            np.fill_diagonal(distance, 0)  # Diagonal must be 0
            sil_score = silhouette_score(distance, labels, metric="precomputed")
        else:
            sil_score = 0.0
        
        # Cluster sizes
        cluster_sizes = [np.sum(labels == c) for c in range(n_clusters)]
        
        result = DomainDiscoveryResult(
            num_sources=collector.num_sources,
            num_domains=n_clusters,
            domain_assignments=labels,
            similarity_matrix=similarity,
            conflict_matrix=conflict,
            silhouette_score=sil_score,
            cluster_sizes=cluster_sizes,
        )
        
        return result
    
    def _auto_select_domains(
        self,
        similarity: np.ndarray,
    ) -> int:
        """Automatically select number of domains using silhouette score."""
        K = similarity.shape[0]
        
        if K <= 2:
            return K
        
        best_score = -1
        best_k = 2
        
        affinity = (similarity + 1) / 2
        
        # Convert similarity to distance for silhouette
        distance = 1 - similarity
        np.fill_diagonal(distance, 0)
        
        for k in range(2, min(K, self.max_domains + 1)):
            try:
                clustering = SpectralClustering(
                    n_clusters=k,
                    affinity="precomputed",
                    random_state=42,
                )
                labels = clustering.fit_predict(affinity)
                
                score = silhouette_score(distance, labels, metric="precomputed")
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue
        
        return best_k


def compute_transfer_gap(
    model: nn.Module,
    source_loader: torch.utils.data.DataLoader,
    target_loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: torch.device,
) -> float:
    """
    Compute transfer gap: performance drop when applying source-trained model to target.
    
    This validates that discovered domains predict negative transfer.
    """
    model.eval()
    
    source_loss = 0.0
    target_loss = 0.0
    n_source = 0
    n_target = 0
    
    with torch.no_grad():
        for batch in source_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            pred = model(x)
            source_loss += loss_fn(pred, y).item() * len(x)
            n_source += len(x)
        
        for batch in target_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            pred = model(x)
            target_loss += loss_fn(pred, y).item() * len(x)
            n_target += len(x)
    
    source_loss /= n_source
    target_loss /= n_target
    
    # Transfer gap = how much worse on target vs source
    return target_loss - source_loss


def validate_domain_discovery(
    result: DomainDiscoveryResult,
    true_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Validate discovered domains against ground truth.
    
    Args:
        result: Discovery result
        true_labels: Ground truth domain labels
    
    Returns:
        Dictionary with validation metrics
    """
    # Adjusted Rand Index (clustering quality)
    ari = adjusted_rand_score(true_labels, result.domain_assignments)
    
    # Check if conflict correlates with different domains
    K = result.num_sources
    same_domain_sim = []
    diff_domain_sim = []
    same_domain_conf = []
    diff_domain_conf = []
    
    for i in range(K):
        for j in range(i + 1, K):
            if true_labels[i] == true_labels[j]:
                same_domain_sim.append(result.similarity_matrix[i, j])
                same_domain_conf.append(result.conflict_matrix[i, j])
            else:
                diff_domain_sim.append(result.similarity_matrix[i, j])
                diff_domain_conf.append(result.conflict_matrix[i, j])
    
    return {
        "adjusted_rand_index": ari,
        "same_domain_similarity": np.mean(same_domain_sim) if same_domain_sim else 0.0,
        "diff_domain_similarity": np.mean(diff_domain_sim) if diff_domain_sim else 0.0,
        "same_domain_conflict": np.mean(same_domain_conf) if same_domain_conf else 0.0,
        "diff_domain_conflict": np.mean(diff_domain_conf) if diff_domain_conf else 0.0,
        "silhouette_score": result.silhouette_score,
    }
