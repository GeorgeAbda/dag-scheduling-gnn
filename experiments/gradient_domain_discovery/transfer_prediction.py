"""
Transfer Prediction using Gradient Divergence

Predicts transfer success between domains BEFORE training by analyzing
gradient geometry. This enables:
1. Early detection of negative transfer
2. Source selection for transfer learning
3. Domain grouping for multi-task learning

Theory:
- High gradient similarity → positive transfer expected
- High gradient conflict → negative transfer expected
- Transfer gap bounded by gradient divergence (Theorem 3)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class TransferPrediction:
    """Prediction of transfer success between source and target."""
    source_id: int
    target_id: int
    gradient_similarity: float
    conflict_rate: float
    predicted_transfer_gap: float
    transfer_recommendation: str  # "positive", "negative", "neutral"
    confidence: float


@dataclass
class TransferMatrix:
    """Full transfer prediction matrix."""
    num_sources: int
    source_names: List[str]
    similarity_matrix: np.ndarray
    conflict_matrix: np.ndarray
    predicted_gap_matrix: np.ndarray
    recommendations: np.ndarray  # 1=positive, 0=neutral, -1=negative
    
    def get_best_sources(self, target_id: int, top_k: int = 3) -> List[int]:
        """Get best source domains for transfer to target."""
        similarities = self.similarity_matrix[:, target_id].copy()
        similarities[target_id] = -np.inf  # Exclude self
        return np.argsort(similarities)[-top_k:][::-1].tolist()
    
    def get_transfer_groups(self, threshold: float = 0.1) -> List[List[int]]:
        """Group sources that can transfer well to each other."""
        n = self.num_sources
        # Use similarity > threshold as edge criterion
        adjacency = self.similarity_matrix > threshold
        
        # Simple connected components
        visited = [False] * n
        groups = []
        
        for i in range(n):
            if not visited[i]:
                group = []
                stack = [i]
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        group.append(node)
                        for j in range(n):
                            if adjacency[node, j] and not visited[j]:
                                stack.append(j)
                groups.append(sorted(group))
        
        return groups
    
    def summary(self) -> str:
        lines = [
            "Transfer Prediction Summary",
            f"  Sources: {self.num_sources}",
            f"  Mean similarity: {np.mean(self.similarity_matrix[np.triu_indices(self.num_sources, k=1)]):.4f}",
            f"  Mean conflict rate: {np.mean(self.conflict_matrix[np.triu_indices(self.num_sources, k=1)]):.4f}",
            f"  Positive transfer pairs: {np.sum(self.recommendations == 1) // 2}",
            f"  Negative transfer pairs: {np.sum(self.recommendations == -1) // 2}",
        ]
        return "\n".join(lines)


class TransferPredictor:
    """
    Predicts transfer success using gradient geometry.
    
    Based on Theorem 3 (Transfer Gap Bound):
    The transfer gap is bounded by gradient divergence.
    """
    
    def __init__(
        self,
        positive_threshold: float = 0.1,
        negative_threshold: float = -0.05,
        conflict_threshold: float = 0.6,
    ):
        """
        Args:
            positive_threshold: Similarity above this predicts positive transfer
            negative_threshold: Similarity below this predicts negative transfer
            conflict_threshold: Conflict rate above this predicts negative transfer
        """
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.conflict_threshold = conflict_threshold
    
    def predict_transfer(
        self,
        similarity_matrix: np.ndarray,
        conflict_matrix: np.ndarray,
        source_names: Optional[List[str]] = None,
    ) -> TransferMatrix:
        """
        Predict transfer success for all source pairs.
        
        Args:
            similarity_matrix: Pairwise gradient cosine similarities
            conflict_matrix: Pairwise gradient conflict rates
            source_names: Optional names for sources
        
        Returns:
            TransferMatrix with predictions
        """
        n = similarity_matrix.shape[0]
        
        if source_names is None:
            source_names = [f"Source_{i}" for i in range(n)]
        
        # Predict transfer gap from similarity
        # Theory: gap ∝ -similarity (negative correlation)
        # We use a simple linear model: gap = α - β * similarity
        predicted_gap = 0.1 - 0.3 * similarity_matrix
        
        # Determine recommendations
        recommendations = np.zeros((n, n), dtype=int)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    recommendations[i, j] = 1  # Self-transfer is always good
                else:
                    sim = similarity_matrix[i, j]
                    conf = conflict_matrix[i, j]
                    
                    if sim > self.positive_threshold and conf < self.conflict_threshold:
                        recommendations[i, j] = 1  # Positive transfer
                    elif sim < self.negative_threshold or conf > self.conflict_threshold:
                        recommendations[i, j] = -1  # Negative transfer
                    else:
                        recommendations[i, j] = 0  # Neutral
        
        return TransferMatrix(
            num_sources=n,
            source_names=source_names,
            similarity_matrix=similarity_matrix,
            conflict_matrix=conflict_matrix,
            predicted_gap_matrix=predicted_gap,
            recommendations=recommendations,
        )
    
    def predict_single(
        self,
        source_id: int,
        target_id: int,
        similarity: float,
        conflict_rate: float,
    ) -> TransferPrediction:
        """Predict transfer for a single source-target pair."""
        
        predicted_gap = 0.1 - 0.3 * similarity
        
        if similarity > self.positive_threshold and conflict_rate < self.conflict_threshold:
            recommendation = "positive"
            confidence = min(similarity / 0.3, 1.0)
        elif similarity < self.negative_threshold or conflict_rate > self.conflict_threshold:
            recommendation = "negative"
            confidence = min(abs(similarity) / 0.2 + (conflict_rate - 0.5) / 0.5, 1.0)
        else:
            recommendation = "neutral"
            confidence = 0.5
        
        return TransferPrediction(
            source_id=source_id,
            target_id=target_id,
            gradient_similarity=similarity,
            conflict_rate=conflict_rate,
            predicted_transfer_gap=predicted_gap,
            transfer_recommendation=recommendation,
            confidence=confidence,
        )


def validate_transfer_predictions(
    predictions: TransferMatrix,
    actual_transfer_gaps: np.ndarray,
) -> Dict[str, float]:
    """
    Validate transfer predictions against actual transfer performance.
    
    Args:
        predictions: TransferMatrix from predictor
        actual_transfer_gaps: Matrix of actual transfer gaps (target_loss - source_loss)
    
    Returns:
        Validation metrics
    """
    n = predictions.num_sources
    
    # Flatten upper triangle (excluding diagonal)
    pred_gaps = []
    actual_gaps = []
    pred_recs = []
    
    for i in range(n):
        for j in range(n):
            if i != j:
                pred_gaps.append(predictions.predicted_gap_matrix[i, j])
                actual_gaps.append(actual_transfer_gaps[i, j])
                pred_recs.append(predictions.recommendations[i, j])
    
    pred_gaps = np.array(pred_gaps)
    actual_gaps = np.array(actual_gaps)
    pred_recs = np.array(pred_recs)
    
    # Correlation between predicted and actual gaps
    correlation = np.corrcoef(pred_gaps, actual_gaps)[0, 1]
    
    # Classification accuracy
    actual_positive = actual_gaps < 0.05  # Small gap = positive transfer
    actual_negative = actual_gaps > 0.15  # Large gap = negative transfer
    
    pred_positive = pred_recs == 1
    pred_negative = pred_recs == -1
    
    # Precision and recall for negative transfer detection
    true_neg = np.sum(pred_negative & actual_negative)
    false_neg = np.sum(pred_negative & ~actual_negative)
    false_pos_neg = np.sum(~pred_negative & actual_negative)
    
    precision_neg = true_neg / (true_neg + false_neg + 1e-10)
    recall_neg = true_neg / (true_neg + false_pos_neg + 1e-10)
    
    return {
        "gap_correlation": correlation,
        "negative_transfer_precision": precision_neg,
        "negative_transfer_recall": recall_neg,
        "mean_absolute_error": np.mean(np.abs(pred_gaps - actual_gaps)),
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_transfer_predictions(
    predictions: TransferMatrix,
    output_dir: str = "./figures",
):
    """Create visualizations for transfer predictions."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    n = predictions.num_sources
    names = [s[:8] for s in predictions.source_names]
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # (a) Transfer Recommendation Matrix
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list('rec', ['#CC3311', '#FFFFFF', '#009988'], N=256)
    im = ax.imshow(predictions.recommendations, cmap=cmap, vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Target')
    ax.set_ylabel('Source')
    ax.set_title('(a) Transfer Recommendations', fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#009988', label='Positive'),
        Patch(facecolor='#FFFFFF', edgecolor='black', label='Neutral'),
        Patch(facecolor='#CC3311', label='Negative'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7)
    
    # (b) Predicted Transfer Gap
    ax = axes[1]
    cmap2 = LinearSegmentedColormap.from_list('gap', ['#009988', '#FFFFFF', '#CC3311'], N=256)
    im2 = ax.imshow(predictions.predicted_gap_matrix, cmap=cmap2, 
                    vmin=-0.1, vmax=0.2, aspect='equal')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Target')
    ax.set_ylabel('Source')
    ax.set_title('(b) Predicted Transfer Gap', fontweight='bold')
    plt.colorbar(im2, ax=ax, shrink=0.8, label='Gap')
    
    # (c) Transfer Groups
    ax = axes[2]
    groups = predictions.get_transfer_groups(threshold=0.05)
    
    colors = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#AA3377']
    
    # Create group membership visualization
    group_matrix = np.zeros((n, n))
    for g_idx, group in enumerate(groups):
        for i in group:
            for j in group:
                group_matrix[i, j] = g_idx + 1
    
    im3 = ax.imshow(group_matrix, cmap='tab10', aspect='equal')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title('(c) Discovered Transfer Groups', fontweight='bold')
    
    # Add group labels
    group_text = "\n".join([f"G{i+1}: {[predictions.source_names[j][:6] for j in g]}" 
                           for i, g in enumerate(groups)])
    ax.text(1.02, 0.5, group_text, transform=ax.transAxes, fontsize=7,
           verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'transfer_predictions.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'transfer_predictions.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Transfer prediction plots saved to: {output_dir}")


# =============================================================================
# Demo
# =============================================================================

def demo_transfer_prediction():
    """Demonstrate transfer prediction on Meta-World MT10 data."""
    
    print("="*70)
    print("Transfer Prediction Demo: Meta-World MT10")
    print("="*70)
    
    # Use similarity data from experiments
    tasks = ['reach', 'push', 'pick-place', 'door-open', 'drawer-open', 
             'drawer-close', 'button-press', 'peg-insert', 'window-open', 'window-close']
    
    similarity = np.array([
        [0.13, -0.07, 0.02, 0.02, 0.04, -0.07, 0.02, -0.01, 0.09, 0.06],
        [-0.07, 0.11, -0.02, -0.02, -0.03, 0.09, -0.02, -0.03, -0.12, -0.09],
        [0.02, -0.02, 0.04, -0.00, 0.01, -0.02, 0.01, -0.00, 0.03, 0.02],
        [0.02, -0.02, -0.00, 0.04, 0.02, -0.02, -0.01, -0.00, 0.02, 0.03],
        [0.04, -0.03, 0.01, 0.02, 0.06, -0.04, -0.00, -0.00, 0.05, 0.04],
        [-0.07, 0.09, -0.02, -0.02, -0.04, 0.22, -0.01, -0.04, -0.22, -0.16],
        [0.02, -0.02, 0.01, -0.01, -0.00, -0.01, 0.06, 0.00, 0.03, 0.03],
        [-0.01, -0.03, -0.00, -0.00, -0.00, -0.04, 0.00, 0.05, 0.05, 0.04],
        [0.09, -0.12, 0.03, 0.02, 0.05, -0.22, 0.03, 0.05, 0.28, 0.19],
        [0.06, -0.09, 0.02, 0.03, 0.04, -0.16, 0.03, 0.04, 0.19, 0.18],
    ])
    
    # Approximate conflict rates
    conflict = np.where(similarity < 0, 0.5 - similarity, 0.5 - similarity * 0.5)
    conflict = np.clip(conflict, 0, 1)
    np.fill_diagonal(conflict, 0)
    
    # Predict transfer
    predictor = TransferPredictor(
        positive_threshold=0.08,
        negative_threshold=-0.05,
        conflict_threshold=0.55,
    )
    
    predictions = predictor.predict_transfer(similarity, conflict, tasks)
    
    print(f"\n{predictions.summary()}")
    
    # Show transfer groups
    groups = predictions.get_transfer_groups(threshold=0.05)
    print(f"\nDiscovered Transfer Groups:")
    for i, group in enumerate(groups):
        group_names = [tasks[j] for j in group]
        print(f"  Group {i+1}: {group_names}")
    
    # Show best sources for each target
    print(f"\nBest Transfer Sources:")
    for target_id, target_name in enumerate(tasks):
        best_sources = predictions.get_best_sources(target_id, top_k=2)
        best_names = [tasks[s] for s in best_sources]
        print(f"  {target_name:15} <- {best_names}")
    
    # Show negative transfer warnings
    print(f"\nNegative Transfer Warnings:")
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            if i != j and predictions.recommendations[i, j] == -1:
                print(f"  ⚠ {tasks[i]:12} -> {tasks[j]:12}: "
                      f"sim={similarity[i,j]:.2f}, conflict={conflict[i,j]:.2f}")
    
    # Generate plots
    plot_transfer_predictions(predictions)
    
    return predictions


if __name__ == "__main__":
    demo_transfer_prediction()
