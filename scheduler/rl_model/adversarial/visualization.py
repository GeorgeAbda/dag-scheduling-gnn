"""
Visualization tools for adversarial attack analysis.

This module provides comprehensive visualization tools for analyzing
adversarial attacks on RL-based cloud scheduling agents.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import os
from dataclasses import dataclass
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from scheduler.rl_model.agents.gin_agent.mapper import GinAgentObsTensor
from scheduler.rl_model.adversarial.attacks import AttackEnsemble, BaseAttack


@dataclass
class VisualizationConfig:
    """Configuration for visualization tools."""
    # Output settings
    save_plots: bool = True
    plot_dir: str = "adversarial_plots"
    plot_format: str = "png"  # png, pdf, svg
    dpi: int = 300
    
    # Visualization parameters
    figsize: Tuple[int, int] = (12, 8)
    color_palette: str = "husl"
    style: str = "whitegrid"
    
    # Feature analysis
    use_pca: bool = True
    use_tsne: bool = True
    pca_components: int = 2
    tsne_components: int = 2
    tsne_perplexity: float = 30.0


class AdversarialVisualizer:
    """
    Comprehensive visualization tools for adversarial attack analysis.
    """
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette(config.color_palette)
        
        # Create output directory
        if config.save_plots:
            os.makedirs(config.plot_dir, exist_ok=True)
    
    def visualize_attack_impact(self, clean_obs: GinAgentObsTensor, 
                               attack_results: Dict[str, GinAgentObsTensor],
                               agent: torch.nn.Module, save_name: str = "attack_impact") -> None:
        """
        Visualize the impact of different attacks on the observation space.
        
        Args:
            clean_obs: Clean observation
            attack_results: Dictionary of attack results
            agent: The RL agent
            save_name: Name for saving the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Extract features for visualization
        clean_features = self._extract_features(clean_obs)
        
        # Plot 1: Task length distribution
        self._plot_feature_distribution(axes[0], clean_obs.task_length, 
                                      [attack_results[attack].task_length for attack in attack_results.keys()],
                                      "Task Length", attack_results.keys())
        
        # Plot 2: VM speed distribution
        self._plot_feature_distribution(axes[1], clean_obs.vm_speed,
                                      [attack_results[attack].vm_speed for attack in attack_results.keys()],
                                      "VM Speed", attack_results.keys())
        
        # Plot 3: Memory requirements
        self._plot_feature_distribution(axes[2], clean_obs.task_memory_req_mb,
                                      [attack_results[attack].task_memory_req_mb for attack in attack_results.keys()],
                                      "Task Memory (MB)", attack_results.keys())
        
        # Plot 4: Action probability changes
        self._plot_action_probability_changes(axes[3], clean_obs, attack_results, agent)
        
        # Plot 5: Feature correlation heatmap
        self._plot_feature_correlation(axes[4], clean_obs, attack_results)
        
        # Plot 6: Attack effectiveness comparison
        self._plot_attack_effectiveness(axes[5], clean_obs, attack_results, agent)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(os.path.join(self.config.plot_dir, f"{save_name}.{self.config.plot_format}"),
                       dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def visualize_graph_attacks(self, clean_obs: GinAgentObsTensor,
                               attack_results: Dict[str, GinAgentObsTensor],
                               save_name: str = "graph_attacks") -> None:
        """
        Visualize graph structure attacks.
        
        Args:
            clean_obs: Clean observation
            attack_results: Dictionary of attack results
            save_name: Name for saving the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Original graph
        self._plot_task_vm_graph(axes[0, 0], clean_obs, "Original Graph")
        
        # Plot 2: Edge removal attack
        if "EdgeRemoval" in attack_results:
            self._plot_task_vm_graph(axes[0, 1], attack_results["EdgeRemoval"], "Edge Removal Attack")
        
        # Plot 3: Edge addition attack
        if "EdgeAddition" in attack_results:
            self._plot_task_vm_graph(axes[1, 0], attack_results["EdgeAddition"], "Edge Addition Attack")
        
        # Plot 4: Graph statistics comparison
        self._plot_graph_statistics(axes[1, 1], clean_obs, attack_results)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(os.path.join(self.config.plot_dir, f"{save_name}.{self.config.plot_format}"),
                       dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def visualize_feature_perturbations(self, clean_obs: GinAgentObsTensor,
                                       attack_results: Dict[str, GinAgentObsTensor],
                                       save_name: str = "feature_perturbations") -> None:
        """
        Visualize feature perturbations in 2D space.
        
        Args:
            clean_obs: Clean observation
            attack_results: Dictionary of attack results
            save_name: Name for saving the plot
        """
        # Extract and combine features
        clean_features = self._extract_features(clean_obs)
        attack_features = {attack: self._extract_features(obs) for attack, obs in attack_results.items()}
        
        # Apply dimensionality reduction
        if self.config.use_pca:
            reduced_features = self._apply_pca(clean_features, attack_features)
        elif self.config.use_tsne:
            reduced_features = self._apply_tsne(clean_features, attack_features)
        else:
            # Use first two features
            reduced_features = {
                'clean': clean_features[:, :2],
                **{attack: features[:, :2] for attack, features in attack_features.items()}
            }
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Plot clean data
        ax.scatter(reduced_features['clean'][:, 0], reduced_features['clean'][:, 1],
                  c='blue', s=50, alpha=0.7, label='Clean', marker='o')
        
        # Plot attacked data
        colors = plt.cm.tab10(np.linspace(0, 1, len(attack_features)))
        for i, (attack_name, features) in enumerate(reduced_features.items()):
            if attack_name == 'clean':
                continue
            ax.scatter(features[:, 0], features[:, 1],
                      c=[colors[i]], s=50, alpha=0.7, label=attack_name, marker='^')
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title('Feature Space Visualization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if self.config.save_plots:
            plt.savefig(os.path.join(self.config.plot_dir, f"{save_name}.{self.config.plot_format}"),
                       dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def visualize_attack_robustness(self, robustness_data: Dict[str, Any],
                                   save_name: str = "attack_robustness") -> None:
        """
        Visualize robustness metrics across different attacks and strengths.
        
        Args:
            robustness_data: Dictionary containing robustness evaluation results
            save_name: Name for saving the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Performance degradation vs attack strength
        self._plot_performance_degradation(axes[0, 0], robustness_data)
        
        # Plot 2: Robustness score comparison
        self._plot_robustness_scores(axes[0, 1], robustness_data)
        
        # Plot 3: Attack success rate
        self._plot_attack_success_rate(axes[1, 0], robustness_data)
        
        # Plot 4: Defense effectiveness
        self._plot_defense_effectiveness(axes[1, 1], robustness_data)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(os.path.join(self.config.plot_dir, f"{save_name}.{self.config.plot_format}"),
                       dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, evaluation_results: Dict[str, Any],
                                   save_name: str = "adversarial_dashboard") -> None:
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            evaluation_results: Dictionary containing evaluation results
            save_name: Name for saving the dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Degradation', 'Robustness Scores',
                          'Attack Effectiveness', 'Defense Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Performance degradation
        if 'attack_strengths' in evaluation_results and 'makespan_degradation' in evaluation_results:
            fig.add_trace(
                go.Scatter(x=evaluation_results['attack_strengths'],
                          y=evaluation_results['makespan_degradation'],
                          mode='lines+markers',
                          name='Makespan Degradation',
                          line=dict(color='blue')),
                row=1, col=1
            )
            
            if 'energy_degradation' in evaluation_results:
                fig.add_trace(
                    go.Scatter(x=evaluation_results['attack_strengths'],
                              y=evaluation_results['energy_degradation'],
                              mode='lines+markers',
                              name='Energy Degradation',
                              line=dict(color='red')),
                    row=1, col=1
                )
        
        # Plot 2: Robustness scores
        if 'robustness_scores' in evaluation_results:
            fig.add_trace(
                go.Bar(x=list(evaluation_results['robustness_scores'].keys()),
                      y=list(evaluation_results['robustness_scores'].values()),
                      name='Robustness Score'),
                row=1, col=2
            )
        
        # Plot 3: Attack effectiveness heatmap
        if 'attack_effectiveness' in evaluation_results:
            effectiveness_matrix = evaluation_results['attack_effectiveness']
            fig.add_trace(
                go.Heatmap(z=effectiveness_matrix['values'],
                          x=effectiveness_matrix['x_labels'],
                          y=effectiveness_matrix['y_labels'],
                          colorscale='Reds'),
                row=2, col=1
            )
        
        # Plot 4: Defense comparison
        if 'defense_comparison' in evaluation_results:
            defense_data = evaluation_results['defense_comparison']
            fig.add_trace(
                go.Bar(x=defense_data['methods'],
                      y=defense_data['effectiveness'],
                      name='Defense Effectiveness'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Adversarial Attack Analysis Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save as HTML
        if self.config.save_plots:
            fig.write_html(os.path.join(self.config.plot_dir, f"{save_name}.html"))
        
        # Show the dashboard
        fig.show()
    
    def _extract_features(self, obs: GinAgentObsTensor) -> np.ndarray:
        """Extract features from observation for visualization."""
        # Combine task and VM features
        task_features = torch.stack([
            obs.task_state_scheduled,
            obs.task_state_ready,
            obs.task_length,
            obs.task_completion_time,
            obs.task_memory_req_mb / 1000.0,
            obs.task_cpu_req_cores / 10.0,
        ], dim=-1)
        
        vm_features = torch.stack([
            obs.vm_completion_time,
            1 / (obs.vm_speed + 1e-8),
            obs.vm_energy_rate,
            obs.vm_memory_mb / 1000.0,
            obs.vm_available_memory_mb / 1000.0,
            obs.vm_used_memory_fraction,
            obs.vm_active_tasks_count / 10.0,
            obs.vm_cpu_cores / 10.0,
            obs.vm_available_cpu_cores / 10.0,
            obs.vm_used_cpu_fraction_cores,
        ], dim=-1)
        
        # Pad to same size
        max_len = max(task_features.shape[0], vm_features.shape[0])
        if task_features.shape[0] < max_len:
            task_features = torch.cat([
                task_features,
                torch.zeros(max_len - task_features.shape[0], task_features.shape[1])
            ])
        if vm_features.shape[0] < max_len:
            vm_features = torch.cat([
                vm_features,
                torch.zeros(max_len - vm_features.shape[0], vm_features.shape[1])
            ])
        
        # Concatenate features
        combined_features = torch.cat([task_features, vm_features], dim=-1)
        return combined_features.numpy()
    
    def _plot_feature_distribution(self, ax, clean_feature, attack_features, title, attack_names):
        """Plot feature distribution comparison."""
        # Plot clean distribution
        ax.hist(clean_feature.numpy(), bins=20, alpha=0.7, label='Clean', density=True)
        
        # Plot attack distributions
        colors = plt.cm.tab10(np.linspace(0, 1, len(attack_features)))
        for i, (attack_feature, attack_name) in enumerate(zip(attack_features, attack_names)):
            ax.hist(attack_feature.numpy(), bins=20, alpha=0.5, 
                   label=attack_name, density=True, color=colors[i])
        
        ax.set_title(title)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_action_probability_changes(self, ax, clean_obs, attack_results, agent):
        """Plot changes in action probabilities."""
        with torch.no_grad():
            clean_scores = agent(clean_obs)
            clean_probs = torch.softmax(clean_scores, dim=-1)
            
            attack_names = []
            prob_changes = []
            
            for attack_name, attack_obs in attack_results.items():
                attack_scores = agent(attack_obs)
                attack_probs = torch.softmax(attack_scores, dim=-1)
                
                # Compute probability change
                prob_change = torch.abs(attack_probs - clean_probs).mean().item()
                attack_names.append(attack_name)
                prob_changes.append(prob_change)
        
        # Create bar plot
        bars = ax.bar(attack_names, prob_changes, alpha=0.7)
        ax.set_title('Action Probability Changes')
        ax.set_ylabel('Average Probability Change')
        ax.set_xlabel('Attack Type')
        ax.tick_params(axis='x', rotation=45)
        
        # Color bars by magnitude
        colors = plt.cm.Reds(np.array(prob_changes) / max(prob_changes))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    def _plot_feature_correlation(self, ax, clean_obs, attack_results):
        """Plot feature correlation heatmap."""
        # Extract features
        clean_features = self._extract_features(clean_obs)
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(clean_features.T)
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title('Feature Correlation Matrix')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation')
    
    def _plot_attack_effectiveness(self, ax, clean_obs, attack_results, agent):
        """Plot attack effectiveness metrics."""
        with torch.no_grad():
            clean_scores = agent(clean_obs)
            clean_entropy = -torch.sum(torch.softmax(clean_scores, dim=-1) * 
                                     torch.log(torch.softmax(clean_scores, dim=-1) + 1e-8), dim=-1).item()
            
            attack_names = []
            entropy_changes = []
            
            for attack_name, attack_obs in attack_results.items():
                attack_scores = agent(attack_obs)
                attack_entropy = -torch.sum(torch.softmax(attack_scores, dim=-1) * 
                                          torch.log(torch.softmax(attack_scores, dim=-1) + 1e-8), dim=-1).item()
                
                entropy_change = attack_entropy - clean_entropy
                attack_names.append(attack_name)
                entropy_changes.append(entropy_change)
        
        # Create bar plot
        bars = ax.bar(attack_names, entropy_changes, alpha=0.7)
        ax.set_title('Attack Effectiveness (Entropy Change)')
        ax.set_ylabel('Entropy Change')
        ax.set_xlabel('Attack Type')
        ax.tick_params(axis='x', rotation=45)
        
        # Color bars by effectiveness
        colors = plt.cm.Reds(np.array(entropy_changes) / max(entropy_changes) if max(entropy_changes) > 0 else [0] * len(entropy_changes))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    def _plot_task_vm_graph(self, ax, obs, title):
        """Plot task-VM compatibility graph."""
        # Create graph
        G = nx.DiGraph()
        
        # Add task nodes
        num_tasks = obs.task_state_scheduled.shape[0]
        for i in range(num_tasks):
            G.add_node(f"T{i}", node_type="task")
        
        # Add VM nodes
        num_vms = obs.vm_completion_time.shape[0]
        for i in range(num_vms):
            G.add_node(f"V{i}", node_type="vm")
        
        # Add edges
        for i in range(obs.compatibilities.shape[1]):
            task_id = obs.compatibilities[0, i].item()
            vm_id = obs.compatibilities[1, i].item()
            G.add_edge(f"T{task_id}", f"V{vm_id}")
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        task_nodes = [n for n in G.nodes() if n.startswith('T')]
        vm_nodes = [n for n in G.nodes() if n.startswith('V')]
        
        nx.draw_networkx_nodes(G, pos, nodelist=task_nodes, node_color='lightblue', 
                              node_size=300, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=vm_nodes, node_color='lightcoral', 
                              node_size=300, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.set_title(title)
        ax.axis('off')
    
    def _plot_graph_statistics(self, ax, clean_obs, attack_results):
        """Plot graph statistics comparison."""
        stats = {
            'Clean': self._compute_graph_stats(clean_obs),
        }
        
        for attack_name, attack_obs in attack_results.items():
            stats[attack_name] = self._compute_graph_stats(attack_obs)
        
        # Create comparison plot
        metrics = ['num_edges', 'avg_degree', 'density']
        x = np.arange(len(metrics))
        width = 0.8 / len(stats)
        
        for i, (name, stat_dict) in enumerate(stats.items()):
            values = [stat_dict[metric] for metric in metrics]
            ax.bar(x + i * width, values, width, label=name, alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Graph Statistics Comparison')
        ax.set_xticks(x + width * (len(stats) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _compute_graph_stats(self, obs):
        """Compute graph statistics."""
        num_tasks = obs.task_state_scheduled.shape[0]
        num_vms = obs.vm_completion_time.shape[0]
        num_edges = obs.compatibilities.shape[1]
        
        # Create graph for analysis
        G = nx.DiGraph()
        for i in range(obs.compatibilities.shape[1]):
            task_id = obs.compatibilities[0, i].item()
            vm_id = obs.compatibilities[1, i].item()
            G.add_edge(f"T{task_id}", f"V{vm_id}")
        
        return {
            'num_edges': num_edges,
            'avg_degree': 2 * num_edges / (num_tasks + num_vms) if (num_tasks + num_vms) > 0 else 0,
            'density': nx.density(G) if G.number_of_nodes() > 0 else 0
        }
    
    def _apply_pca(self, clean_features, attack_features):
        """Apply PCA dimensionality reduction."""
        # Combine all features
        all_features = np.vstack([clean_features] + list(attack_features.values()))
        
        # Fit PCA
        pca = PCA(n_components=self.config.pca_components)
        pca.fit(all_features)
        
        # Transform features
        reduced_features = {'clean': pca.transform(clean_features)}
        for attack_name, features in attack_features.items():
            reduced_features[attack_name] = pca.transform(features)
        
        return reduced_features
    
    def _apply_tsne(self, clean_features, attack_features):
        """Apply t-SNE dimensionality reduction."""
        # Combine all features
        all_features = np.vstack([clean_features] + list(attack_features.values()))
        
        # Fit t-SNE
        tsne = TSNE(n_components=self.config.tsne_components, 
                   perplexity=self.config.tsne_perplexity, random_state=42)
        tsne_result = tsne.fit_transform(all_features)
        
        # Split results
        clean_len = len(clean_features)
        reduced_features = {'clean': tsne_result[:clean_len]}
        
        start_idx = clean_len
        for attack_name, features in attack_features.items():
            end_idx = start_idx + len(features)
            reduced_features[attack_name] = tsne_result[start_idx:end_idx]
            start_idx = end_idx
        
        return reduced_features
    
    def _plot_performance_degradation(self, ax, robustness_data):
        """Plot performance degradation."""
        if 'attack_strengths' in robustness_data and 'makespan_degradation' in robustness_data:
            ax.plot(robustness_data['attack_strengths'], 
                   robustness_data['makespan_degradation'], 
                   'o-', label='Makespan', linewidth=2)
        
        if 'attack_strengths' in robustness_data and 'energy_degradation' in robustness_data:
            ax.plot(robustness_data['attack_strengths'], 
                   robustness_data['energy_degradation'], 
                   's-', label='Energy', linewidth=2)
        
        ax.set_xlabel('Attack Strength')
        ax.set_ylabel('Performance Degradation')
        ax.set_title('Performance Degradation vs Attack Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_robustness_scores(self, ax, robustness_data):
        """Plot robustness scores."""
        if 'robustness_scores' in robustness_data:
            methods = list(robustness_data['robustness_scores'].keys())
            scores = list(robustness_data['robustness_scores'].values())
            
            bars = ax.bar(methods, scores, alpha=0.7)
            ax.set_ylabel('Robustness Score')
            ax.set_title('Robustness Score Comparison')
            ax.tick_params(axis='x', rotation=45)
            
            # Color bars by score
            colors = plt.cm.Greens(np.array(scores) / max(scores))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
    
    def _plot_attack_success_rate(self, ax, robustness_data):
        """Plot attack success rate."""
        if 'attack_success_rates' in robustness_data:
            attacks = list(robustness_data['attack_success_rates'].keys())
            rates = list(robustness_data['attack_success_rates'].values())
            
            bars = ax.bar(attacks, rates, alpha=0.7)
            ax.set_ylabel('Success Rate')
            ax.set_title('Attack Success Rate')
            ax.tick_params(axis='x', rotation=45)
            
            # Color bars by success rate
            colors = plt.cm.Reds(np.array(rates) / max(rates))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
    
    def _plot_defense_effectiveness(self, ax, robustness_data):
        """Plot defense effectiveness."""
        if 'defense_effectiveness' in robustness_data:
            defenses = list(robustness_data['defense_effectiveness'].keys())
            effectiveness = list(robustness_data['defense_effectiveness'].values())
            
            bars = ax.bar(defenses, effectiveness, alpha=0.7)
            ax.set_ylabel('Effectiveness')
            ax.set_title('Defense Effectiveness')
            ax.tick_params(axis='x', rotation=45)
            
            # Color bars by effectiveness
            colors = plt.cm.Blues(np.array(effectiveness) / max(effectiveness))
            for bar, color in zip(bars, colors):
                bar.set_color(color)


def create_attack_analysis_report(clean_obs: GinAgentObsTensor, 
                                 attack_results: Dict[str, GinAgentObsTensor],
                                 agent: torch.nn.Module,
                                 output_dir: str = "adversarial_analysis") -> str:
    """
    Create a comprehensive attack analysis report.
    
    Args:
        clean_obs: Clean observation
        attack_results: Dictionary of attack results
        agent: The RL agent
        output_dir: Directory to save the report
    
    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizer
    config = VisualizationConfig(plot_dir=output_dir)
    visualizer = AdversarialVisualizer(config)
    
    # Generate visualizations
    visualizer.visualize_attack_impact(clean_obs, attack_results, agent, "attack_impact")
    visualizer.visualize_graph_attacks(clean_obs, attack_results, "graph_attacks")
    visualizer.visualize_feature_perturbations(clean_obs, attack_results, "feature_perturbations")
    
    # Create summary report
    report_path = os.path.join(output_dir, "attack_analysis_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Adversarial Attack Analysis Report\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Analyzed {len(attack_results)} different attack types:\n")
        for attack_name in attack_results.keys():
            f.write(f"- {attack_name}\n")
        
        f.write("\n## Visualizations\n\n")
        f.write("The following visualizations have been generated:\n")
        f.write("- `attack_impact.png`: Overall impact of attacks\n")
        f.write("- `graph_attacks.png`: Graph structure attack analysis\n")
        f.write("- `feature_perturbations.png`: Feature space visualization\n")
        
        f.write("\n## Analysis\n\n")
        f.write("### Attack Effectiveness\n")
        
        # Compute attack effectiveness
        with torch.no_grad():
            clean_scores = agent(clean_obs)
            clean_entropy = -torch.sum(torch.softmax(clean_scores, dim=-1) * 
                                     torch.log(torch.softmax(clean_scores, dim=-1) + 1e-8), dim=-1).item()
            
            for attack_name, attack_obs in attack_results.items():
                attack_scores = agent(attack_obs)
                attack_entropy = -torch.sum(torch.softmax(attack_scores, dim=-1) * 
                                          torch.log(torch.softmax(attack_scores, dim=-1) + 1e-8), dim=-1).item()
                
                entropy_change = attack_entropy - clean_entropy
                f.write(f"- **{attack_name}**: Entropy change = {entropy_change:.4f}\n")
    
    print(f"Attack analysis report generated: {report_path}")
    return report_path


if __name__ == "__main__":
    # Example usage
    print("Adversarial visualization tools loaded successfully!")
    print("Use create_attack_analysis_report() to generate comprehensive analysis reports.")

