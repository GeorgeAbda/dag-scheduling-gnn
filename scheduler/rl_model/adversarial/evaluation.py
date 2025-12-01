"""
Evaluation framework for adversarial robustness testing.

This module provides comprehensive evaluation tools for testing the robustness
of RL-based cloud scheduling agents against various adversarial attacks.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import os
import json
import time
import sys
from collections import defaultdict

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
project_root = os.path.dirname(grandparent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scheduler.rl_model.agents.gin_agent.agent import GinAgent
from scheduler.rl_model.agents.gin_agent.mapper import GinAgentMapper
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.core.env.observation import EnvObservation
from scheduler.rl_model.adversarial.attacks import (
    AttackConfig, AttackEnsemble, BaseAttack, 
    EdgeRemovalAttack, EdgeAdditionAttack, TaskLengthAttack, 
    VMSpeedAttack, MemoryAttack, RedTeamAttack
)
from scheduler.rl_model.adversarial.defenses import (
    DefenseConfig, AdversarialTrainingDefense, EnsembleDefense, 
    InputValidationDefense, RobustAgent
)


@dataclass
class EvaluationConfig:
    """Configuration for robustness evaluation."""
    # Evaluation parameters
    num_test_episodes: int = 100
    num_attack_variants: int = 5  # Number of different attack strengths to test
    
    # Attack parameters
    attack_strengths: List[float] = None  # Different attack strengths to test
    
    # Evaluation metrics
    compute_energy_metrics: bool = True
    compute_makespan_metrics: bool = True
    compute_robustness_metrics: bool = True
    
    # Output settings
    save_results: bool = True
    results_dir: str = "adversarial_results"
    save_plots: bool = True
    plot_dir: str = "adversarial_plots"
    
    def __post_init__(self):
        if self.attack_strengths is None:
            self.attack_strengths = [0.0, 0.05, 0.1, 0.15, 0.2]  # 0.0 = no attack


class RobustnessEvaluator:
    """
    Comprehensive robustness evaluator for RL-based cloud scheduling agents.
    """
    
    def __init__(self, config: EvaluationConfig, device: torch.device):
        self.config = config
        self.device = device
        self.results = defaultdict(list)
        
        # Create output directories
        if config.save_results:
            os.makedirs(config.results_dir, exist_ok=True)
        if config.save_plots:
            os.makedirs(config.plot_dir, exist_ok=True)
    
    def evaluate_agent_robustness(self, agent: GinAgent, dataset_generator, 
                                 attack_config: AttackConfig, defense_config: DefenseConfig = None) -> Dict[str, Any]:
        """
        Evaluate the robustness of an agent against various attacks.
        
        Args:
            agent: The RL agent to evaluate
            dataset_generator: Function to generate test datasets
            attack_config: Configuration for attacks
            defense_config: Configuration for defenses (optional)
        
        Returns:
            Dictionary containing evaluation results
        """
        print("Starting robustness evaluation...")
        
        # Initialize attack ensemble
        attack_ensemble = AttackEnsemble(attack_config, agent, self.device)
        
        # Initialize defense if provided
        robust_agent = None
        if defense_config:
            robust_agent = RobustAgent(agent, defense_config, attack_config, self.device)
        
        # Test different attack strengths
        for strength in self.config.attack_strengths:
            print(f"Testing attack strength: {strength}")
            
            # Update attack configuration
            current_attack_config = AttackConfig(
                edge_removal_ratio=strength,
                edge_addition_ratio=strength,
                task_length_noise_std=strength,
                vm_speed_noise_std=strength,
                memory_noise_std=strength,
                epsilon=strength
            )
            
            # Run evaluation for this strength
            strength_results = self._evaluate_attack_strength(
                agent, robust_agent, dataset_generator, current_attack_config
            )
            
            # Store results
            for metric, value in strength_results.items():
                self.results[metric].append(value)
        
        # Generate summary statistics
        summary = self._generate_summary_statistics()
        
        # Save results
        if self.config.save_results:
            self._save_results(summary)
        
        # Generate plots
        if self.config.save_plots:
            self._generate_plots(summary)
        
        return summary
    
    def _evaluate_attack_strength(self, agent: GinAgent, robust_agent: Optional[RobustAgent],
                                 dataset_generator, attack_config: AttackConfig) -> Dict[str, float]:
        """Evaluate agent performance under a specific attack strength."""
        
        # Initialize attack ensemble with current config
        attack_ensemble = AttackEnsemble(attack_config, agent, self.device)
        
        # Metrics to track
        clean_metrics = defaultdict(list)
        attack_metrics = defaultdict(list)
        
        # Run test episodes
        for episode in range(self.config.num_test_episodes):
            # Generate test dataset
            dataset = dataset_generator(seed=episode)
            
            # Test clean performance
            clean_makespan, clean_energy, clean_metrics_ep = self._run_episode(
                agent, dataset, episode, use_attacks=False
            )
            
            # Test under attacks
            attack_makespan, attack_energy, attack_metrics_ep = self._run_episode(
                agent, dataset, episode, use_attacks=True, attack_ensemble=attack_ensemble
            )
            
            # Test robust agent if available
            if robust_agent:
                robust_makespan, robust_energy, robust_metrics_ep = self._run_episode(
                    robust_agent, dataset, episode, use_attacks=True, attack_ensemble=attack_ensemble
                )
            
            # Store metrics
            clean_metrics['makespan'].append(clean_makespan)
            clean_metrics['energy'].append(clean_energy)
            attack_metrics['makespan'].append(attack_makespan)
            attack_metrics['energy'].append(attack_energy)
            
            if robust_agent:
                attack_metrics['robust_makespan'].append(robust_makespan)
                attack_metrics['robust_energy'].append(robust_energy)
        
        # Compute average metrics
        results = {
            'clean_makespan': np.mean(clean_metrics['makespan']),
            'clean_energy': np.mean(clean_metrics['energy']),
            'attack_makespan': np.mean(attack_metrics['makespan']),
            'attack_energy': np.mean(attack_metrics['energy']),
            'makespan_degradation': (np.mean(attack_metrics['makespan']) - np.mean(clean_metrics['makespan'])) / np.mean(clean_metrics['makespan']),
            'energy_degradation': (np.mean(attack_metrics['energy']) - np.mean(clean_metrics['energy'])) / np.mean(clean_metrics['energy']),
        }
        
        if robust_agent:
            results.update({
                'robust_makespan': np.mean(attack_metrics['robust_makespan']),
                'robust_energy': np.mean(attack_metrics['robust_energy']),
                'robust_makespan_improvement': (np.mean(attack_metrics['makespan']) - np.mean(attack_metrics['robust_makespan'])) / np.mean(attack_metrics['makespan']),
                'robust_energy_improvement': (np.mean(attack_metrics['energy']) - np.mean(attack_metrics['robust_energy'])) / np.mean(attack_metrics['energy']),
            })
        
        return results
    
    def _run_episode(self, agent: Union[GinAgent, RobustAgent], dataset, seed: int, 
                    use_attacks: bool = False, attack_ensemble: AttackEnsemble = None) -> Tuple[float, float, Dict]:
        """Run a single episode and return metrics."""
        
        # Create environment
        env = CloudSchedulingGymEnvironment(dataset=dataset, compute_metrics=True)
        env = GinAgentWrapper(env)
        
        # Reset environment
        obs, _ = env.reset(seed=seed)
        final_info = None
        
        # Run episode
        while True:
            # Apply attacks if requested (convert to/from GinAgentObsTensor correctly)
            if use_attacks and attack_ensemble:
                mapper = GinAgentMapper(obs.shape[0])
                # Unmap flat observation to structured tensor
                gin_obs = mapper.unmap(torch.from_numpy(obs.astype(np.float32)))
                # Apply random attack from ensemble
                attack_results = attack_ensemble.evaluate_all_attacks(gin_obs, agent)
                attack_names = list(attack_results.keys())
                if attack_names:
                    selected_attack = np.random.choice(attack_names)
                    gin_obs = attack_results[selected_attack]
                # Map back to flat numpy array for the environment/agent
                obs = mapper.map(
                    gin_obs.task_state_scheduled.numpy(),
                    gin_obs.task_state_ready.numpy(),
                    gin_obs.task_length.numpy(),
                    gin_obs.task_completion_time.numpy(),
                    gin_obs.task_memory_req_mb.numpy(),
                    gin_obs.task_cpu_req_cores.numpy(),
                    gin_obs.vm_speed.numpy(),
                    gin_obs.vm_energy_rate.numpy(),
                    gin_obs.vm_completion_time.numpy(),
                    gin_obs.vm_memory_mb.numpy(),
                    gin_obs.vm_available_memory_mb.numpy(),
                    gin_obs.vm_used_memory_fraction.numpy(),
                    gin_obs.vm_active_tasks_count.numpy(),
                    gin_obs.vm_next_release_time.numpy(),
                    gin_obs.vm_cpu_cores.numpy(),
                    gin_obs.vm_available_cpu_cores.numpy(),
                    gin_obs.vm_used_cpu_fraction_cores.numpy(),
                    gin_obs.vm_next_core_release_time.numpy(),
                    gin_obs.task_dependencies.numpy(),
                    gin_obs.compatibilities.numpy(),
                )

            # Convert observation to tensor for the agent
            obs_tensor = torch.from_numpy(obs.astype(np.float32).reshape(1, -1))

            # Get action
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            
            # Take step
            obs, _, terminated, truncated, info = env.step(int(action.item()))
            
            if terminated or truncated:
                final_info = info
                break
        
        # Extract metrics
        makespan = env.prev_obs.makespan() if env.prev_obs else 0.0
        energy = env.prev_obs.energy_consumption() if env.prev_obs else 0.0
        
        # Additional metrics from final_info
        metrics = {}
        if isinstance(final_info, dict):
            metrics.update({
                'total_energy': final_info.get('total_energy', energy),
                'total_energy_active': final_info.get('total_energy_active', 0.0),
                'total_energy_idle': final_info.get('total_energy_idle', 0.0),
                'bottleneck_steps': final_info.get('bottleneck_steps', 0),
                'decision_steps': final_info.get('decision_steps', 0),
            })
        
        env.close()
        return makespan, energy, metrics
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics from evaluation results."""
        summary = {}
        
        for metric, values in self.results.items():
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        # Compute robustness metrics
        if 'makespan_degradation' in summary and 'energy_degradation' in summary:
            summary['robustness_score'] = 1.0 - (summary['makespan_degradation']['mean'] + summary['energy_degradation']['mean']) / 2.0
        
        return summary
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save evaluation results to files."""
        timestamp = int(time.time())
        
        # Save summary as JSON
        summary_path = os.path.join(self.config.results_dir, f"robustness_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_summary = {}
            for key, value in summary.items():
                if isinstance(value, dict) and 'values' in value:
                    json_summary[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in value.items()}
                else:
                    json_summary[key] = value.tolist() if isinstance(value, np.ndarray) else value
            json.dump(json_summary, f, indent=2)
        
        # Save detailed results as CSV
        csv_path = os.path.join(self.config.results_dir, f"robustness_detailed_{timestamp}.csv")
        df_data = []
        for strength_idx, strength in enumerate(self.config.attack_strengths):
            row = {'attack_strength': strength}
            for metric, values in self.results.items():
                if values and len(values) > strength_idx:
                    row[metric] = values[strength_idx]
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to {summary_path} and {csv_path}")
    
    def _generate_plots(self, summary: Dict[str, Any]):
        """Generate visualization plots for the evaluation results."""
        timestamp = int(time.time())
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Plot 1: Performance degradation vs attack strength
        self._plot_performance_degradation(timestamp)
        
        # Plot 2: Robustness comparison
        if 'robust_makespan' in summary:
            self._plot_robustness_comparison(timestamp)
        
        # Plot 3: Attack effectiveness
        self._plot_attack_effectiveness(timestamp)
        
        print(f"Plots saved to {self.config.plot_dir}")
    
    def _plot_performance_degradation(self, timestamp: int):
        """Plot performance degradation vs attack strength."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Makespan degradation
        if 'makespan_degradation' in self.results and self.results['makespan_degradation']:
            ax1.plot(self.config.attack_strengths, self.results['makespan_degradation'], 
                    'o-', linewidth=2, markersize=8, label='Makespan')
            ax1.set_xlabel('Attack Strength')
            ax1.set_ylabel('Performance Degradation')
            ax1.set_title('Makespan Degradation vs Attack Strength')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # Energy degradation
        if 'energy_degradation' in self.results and self.results['energy_degradation']:
            ax2.plot(self.config.attack_strengths, self.results['energy_degradation'], 
                    'o-', linewidth=2, markersize=8, label='Energy', color='red')
            ax2.set_xlabel('Attack Strength')
            ax2.set_ylabel('Performance Degradation')
            ax2.set_title('Energy Degradation vs Attack Strength')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plot_dir, f"performance_degradation_{timestamp}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_robustness_comparison(self, timestamp: int):
        """Plot robustness comparison between clean, attacked, and robust agents."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Makespan comparison
        if 'clean_makespan' in self.results and 'attack_makespan' in self.results:
            x = np.arange(len(self.config.attack_strengths))
            width = 0.25
            
            ax1.bar(x - width, self.results['clean_makespan'], width, label='Clean', alpha=0.8)
            ax1.bar(x, self.results['attack_makespan'], width, label='Attacked', alpha=0.8)
            
            if 'robust_makespan' in self.results:
                ax1.bar(x + width, self.results['robust_makespan'], width, label='Robust', alpha=0.8)
            
            ax1.set_xlabel('Attack Strength')
            ax1.set_ylabel('Makespan')
            ax1.set_title('Makespan Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(self.config.attack_strengths)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Energy comparison
        if 'clean_energy' in self.results and 'attack_energy' in self.results:
            x = np.arange(len(self.config.attack_strengths))
            width = 0.25
            
            ax2.bar(x - width, self.results['clean_energy'], width, label='Clean', alpha=0.8)
            ax2.bar(x, self.results['attack_energy'], width, label='Attacked', alpha=0.8)
            
            if 'robust_energy' in self.results:
                ax2.bar(x + width, self.results['robust_energy'], width, label='Robust', alpha=0.8)
            
            ax2.set_xlabel('Attack Strength')
            ax2.set_ylabel('Energy Consumption')
            ax2.set_title('Energy Consumption Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(self.config.attack_strengths)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plot_dir, f"robustness_comparison_{timestamp}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attack_effectiveness(self, timestamp: int):
        """Plot attack effectiveness metrics."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create a heatmap of attack effectiveness
        if 'makespan_degradation' in self.results and 'energy_degradation' in self.results:
            degradation_matrix = np.array([
                self.results['makespan_degradation'],
                self.results['energy_degradation']
            ])
            
            im = ax.imshow(degradation_matrix, cmap='Reds', aspect='auto')
            ax.set_xticks(range(len(self.config.attack_strengths)))
            ax.set_xticklabels(self.config.attack_strengths)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Makespan', 'Energy'])
            ax.set_xlabel('Attack Strength')
            ax.set_title('Attack Effectiveness Heatmap')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Performance Degradation')
            
            # Add text annotations
            for i in range(2):
                for j in range(len(self.config.attack_strengths)):
                    text = ax.text(j, i, f'{degradation_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plot_dir, f"attack_effectiveness_{timestamp}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def run_robustness_evaluation(model_path: str, output_dir: str = "adversarial_results", 
                            num_episodes: int = 50, device: str = "cpu") -> Dict[str, Any]:
    """
    Convenience function to run a complete robustness evaluation.
    
    Args:
        model_path: Path to the trained model
        output_dir: Directory to save results
        num_episodes: Number of test episodes
        device: Device to use for evaluation
    
    Returns:
        Dictionary containing evaluation results
    """
    # Load the trained agent
    device = torch.device(device)
    agent = GinAgent(device)
    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.eval()
    
    # Create evaluation configuration
    eval_config = EvaluationConfig(
        num_test_episodes=num_episodes,
        results_dir=output_dir,
        plot_dir=os.path.join(output_dir, "plots")
    )
    
    # Create attack configuration
    attack_config = AttackConfig()
    
    # Create defense configuration
    defense_config = DefenseConfig()
    
    # Create evaluator
    evaluator = RobustnessEvaluator(eval_config, device)
    
    # Create dataset generator (you'll need to implement this based on your dataset)
    def dataset_generator(seed):
        # This should return a dataset compatible with your environment
        # You'll need to implement this based on your specific dataset generation
        from scheduler.dataset_generator.gen_dataset import generate_dataset
        return generate_dataset(
            seed=seed,
            host_count=4,
            vm_count=10,
            workflow_count=5,
            gnp_min_n=10,
            gnp_max_n=15,
            max_memory_gb=8,
            min_cpu_speed_mips=1000,
            max_cpu_speed_mips=3000,
            min_task_length=1000,
            max_task_length=50000,
            task_arrival="static",
            dag_method="gnp",
            task_length_dist="uniform",
            arrival_rate=1.0,
            vm_rng_seed=0
        )
    
    # Run evaluation
    results = evaluator.evaluate_agent_robustness(
        agent, dataset_generator, attack_config, defense_config
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        results = run_robustness_evaluation(model_path)
        print("Robustness evaluation completed!")
        print(f"Results: {results}")
    else:
        print("Usage: python evaluation.py <model_path>")

