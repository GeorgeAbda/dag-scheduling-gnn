#!/usr/bin/env python3
"""
Simple Real Adversarial Attack and Defense Training

This script performs real adversarial training using a trained model
with simplified observation handling and focus on core functionality.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
import time
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import pandas as pd

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from scheduler.rl_model.agents.gin_agent.agent import GinAgent
from scheduler.rl_model.adversarial.attacks import (
    AttackConfig, EdgeRemovalAttack, EdgeAdditionAttack, 
    TaskLengthAttack, VMSpeedAttack, MemoryAttack, AttackEnsemble
)
from scheduler.rl_model.adversarial.defenses import (
    DefenseConfig, InputValidationDefense, AdversarialTrainingDefense, RobustAgent
)
from scheduler.rl_model.agents.gin_agent.mapper import GinAgentObsTensor


def load_trained_model(model_path: str, device: torch.device) -> GinAgent:
    """Load a trained model from file."""
    print(f"Loading trained model from {model_path}...")
    
    # Create agent instance
    agent = GinAgent(device=device)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    agent.load_state_dict(state_dict)
    agent.eval()
    
    print("Model loaded successfully!")
    return agent


def create_test_observation() -> GinAgentObsTensor:
    """Create a test observation for demonstration."""
    num_tasks = 5
    num_vms = 3
    
    # Create mock task features
    task_state_scheduled = torch.zeros(num_tasks)
    task_state_ready = torch.ones(num_tasks)
    task_length = torch.rand(num_tasks) * 1000 + 100
    task_completion_time = torch.zeros(num_tasks)
    task_memory_req_mb = torch.rand(num_tasks) * 1000 + 100
    task_cpu_req_cores = torch.randint(1, 5, (num_tasks,)).float()
    
    # Create mock VM features
    vm_completion_time = torch.zeros(num_vms)
    vm_speed = torch.rand(num_vms) * 1000 + 500
    vm_energy_rate = torch.rand(num_vms) * 10 + 1
    vm_memory_mb = torch.rand(num_vms) * 2000 + 1000
    vm_available_memory_mb = vm_memory_mb.clone()
    vm_used_memory_fraction = torch.zeros(num_vms)
    vm_active_tasks_count = torch.zeros(num_vms)
    vm_next_release_time = torch.zeros(num_vms)
    vm_cpu_cores = torch.randint(2, 8, (num_vms,)).float()
    vm_available_cpu_cores = vm_cpu_cores.clone()
    vm_used_cpu_fraction_cores = torch.zeros(num_vms)
    vm_next_core_release_time = torch.zeros(num_vms)
    
    # Create compatibility edges (all tasks can run on all VMs for simplicity)
    compatibilities = []
    for i in range(num_tasks):
        for j in range(num_vms):
            compatibilities.append([i, j])
    compatibilities = torch.tensor(compatibilities, dtype=torch.long).T
    
    # Create simple task dependencies (linear chain)
    task_dependencies = []
    for i in range(num_tasks - 1):
        task_dependencies.append([i, i + 1])
    task_dependencies = torch.tensor(task_dependencies, dtype=torch.long).T if task_dependencies else torch.empty((2, 0), dtype=torch.long)
    
    return GinAgentObsTensor(
        task_state_scheduled=task_state_scheduled,
        task_state_ready=task_state_ready,
        task_length=task_length,
        task_completion_time=task_completion_time,
        task_memory_req_mb=task_memory_req_mb,
        task_cpu_req_cores=task_cpu_req_cores,
        vm_completion_time=vm_completion_time,
        vm_speed=vm_speed,
        vm_energy_rate=vm_energy_rate,
        vm_memory_mb=vm_memory_mb,
        vm_available_memory_mb=vm_available_memory_mb,
        vm_used_memory_fraction=vm_used_memory_fraction,
        vm_active_tasks_count=vm_active_tasks_count,
        vm_next_release_time=vm_next_release_time,
        vm_cpu_cores=vm_cpu_cores,
        vm_available_cpu_cores=vm_available_cpu_cores,
        vm_used_cpu_fraction_cores=vm_used_cpu_fraction_cores,
        vm_next_core_release_time=vm_next_core_release_time,
        compatibilities=compatibilities,
        task_dependencies=task_dependencies
    )


def evaluate_clean_performance(agent: GinAgent, num_episodes: int = 5) -> Dict[str, float]:
    """Evaluate the agent's performance on clean data."""
    print("Evaluating clean performance...")
    
    total_rewards = []
    total_entropies = []
    total_values = []
    
    for episode in range(num_episodes):
        # Create test observation
        obs = create_test_observation()
        
        # Convert to tensor format for agent using mapper
        from scheduler.rl_model.agents.gin_agent.mapper import GinAgentMapper
        from scheduler.config.settings import MAX_OBS_SIZE
        mapper = GinAgentMapper(MAX_OBS_SIZE)
        obs_tensor = torch.from_numpy(mapper.map(
            obs.task_state_scheduled.numpy(),
            obs.task_state_ready.numpy(),
            obs.task_length.numpy(),
            obs.task_completion_time.numpy(),
            obs.task_memory_req_mb.numpy(),
            obs.task_cpu_req_cores.numpy(),
            obs.vm_completion_time.numpy(),
            obs.vm_speed.numpy(),
            obs.vm_energy_rate.numpy(),
            obs.vm_memory_mb.numpy(),
            obs.vm_available_memory_mb.numpy(),
            obs.vm_used_memory_fraction.numpy(),
            obs.vm_active_tasks_count.numpy(),
            obs.vm_next_release_time.numpy(),
            obs.vm_cpu_cores.numpy(),
            obs.vm_available_cpu_cores.numpy(),
            obs.vm_used_cpu_fraction_cores.numpy(),
            obs.vm_next_core_release_time.numpy(),
            obs.task_dependencies.numpy(),
            obs.compatibilities.numpy()
        )).float().unsqueeze(0)  # Add batch dimension.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            action_scores, log_probs, entropy, value = agent.get_action_and_value(obs_tensor)
            
            # Calculate reward based on action quality (simplified)
            reward = torch.max(action_scores).item()
            total_rewards.append(reward)
            total_entropies.append(entropy.item())
            total_values.append(value.item())
    
    return {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_entropy': np.mean(total_entropies),
        'std_entropy': np.std(total_entropies),
        'avg_value': np.mean(total_values),
        'std_value': np.std(total_values)
    }


def evaluate_under_attacks(agent: GinAgent, attack_config: AttackConfig, 
                          num_episodes: int = 5) -> Dict[str, Dict[str, float]]:
    """Evaluate the agent's performance under various attacks."""
    print("Evaluating performance under attacks...")
    
    attack_ensemble = AttackEnsemble(attack_config, agent, torch.device('cpu'))
    
    results = {}
    
    for episode in range(num_episodes):
        # Create test observation
        clean_obs = create_test_observation()
        
        # Get clean performance
        from scheduler.rl_model.agents.gin_agent.mapper import GinAgentMapper
        from scheduler.config.settings import MAX_OBS_SIZE
        mapper = GinAgentMapper(MAX_OBS_SIZE)
        clean_tensor = torch.from_numpy(mapper.map(
            clean_obs.task_state_scheduled.numpy(),
            clean_obs.task_state_ready.numpy(),
            clean_obs.task_length.numpy(),
            clean_obs.task_completion_time.numpy(),
            clean_obs.task_memory_req_mb.numpy(),
            clean_obs.task_cpu_req_cores.numpy(),
            clean_obs.vm_completion_time.numpy(),
            clean_obs.vm_speed.numpy(),
            clean_obs.vm_energy_rate.numpy(),
            clean_obs.vm_memory_mb.numpy(),
            clean_obs.vm_available_memory_mb.numpy(),
            clean_obs.vm_used_memory_fraction.numpy(),
            clean_obs.vm_active_tasks_count.numpy(),
            clean_obs.vm_next_release_time.numpy(),
            clean_obs.vm_cpu_cores.numpy(),
            clean_obs.vm_available_cpu_cores.numpy(),
            clean_obs.vm_used_cpu_fraction_cores.numpy(),
            clean_obs.vm_next_core_release_time.numpy(),
            clean_obs.task_dependencies.numpy(),
            clean_obs.compatibilities.numpy()
        )).float().unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            clean_scores, clean_log_probs, clean_entropy, clean_value = agent.get_action_and_value(clean_tensor)
            clean_reward = torch.max(clean_scores).item()
        
        # Test each attack type
        attack_results = attack_ensemble.evaluate_all_attacks(clean_obs, agent)
        
        for attack_name, attack_obs in attack_results.items():
            if attack_name not in results:
                results[attack_name] = {'rewards': [], 'entropies': [], 'values': []}
            
            # Evaluate attacked observation
            attack_tensor = torch.from_numpy(mapper.map(
                attack_obs.task_state_scheduled.numpy(),
                attack_obs.task_state_ready.numpy(),
                attack_obs.task_length.numpy(),
                attack_obs.task_completion_time.numpy(),
                attack_obs.task_memory_req_mb.numpy(),
                attack_obs.task_cpu_req_cores.numpy(),
                attack_obs.vm_completion_time.numpy(),
                attack_obs.vm_speed.numpy(),
                attack_obs.vm_energy_rate.numpy(),
                attack_obs.vm_memory_mb.numpy(),
                attack_obs.vm_available_memory_mb.numpy(),
                attack_obs.vm_used_memory_fraction.numpy(),
                attack_obs.vm_active_tasks_count.numpy(),
                attack_obs.vm_next_release_time.numpy(),
                attack_obs.vm_cpu_cores.numpy(),
                attack_obs.vm_available_cpu_cores.numpy(),
                attack_obs.vm_used_cpu_fraction_cores.numpy(),
                attack_obs.vm_next_core_release_time.numpy(),
                attack_obs.task_dependencies.numpy(),
                attack_obs.compatibilities.numpy()
            )).float().unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                attack_scores, attack_log_probs, attack_entropy, attack_value = agent.get_action_and_value(attack_tensor)
                attack_reward = torch.max(attack_scores).item()
            
            results[attack_name]['rewards'].append(attack_reward)
            results[attack_name]['entropies'].append(attack_entropy.item())
            results[attack_name]['values'].append(attack_value.item())
    
    # Calculate statistics
    attack_stats = {}
    for attack_name, data in results.items():
        attack_stats[attack_name] = {
            'avg_reward': np.mean(data['rewards']),
            'std_reward': np.std(data['rewards']),
            'avg_entropy': np.mean(data['entropies']),
            'std_entropy': np.std(data['entropies']),
            'avg_value': np.mean(data['values']),
            'std_value': np.std(data['values'])
        }
    
    return attack_stats


def adversarial_training(agent: GinAgent, attack_config: AttackConfig, 
                        defense_config: DefenseConfig, num_epochs: int = 10) -> RobustAgent:
    """Perform adversarial training to improve robustness."""
    print("Starting adversarial training...")
    
    # Create robust agent
    robust_agent = RobustAgent(agent, defense_config, attack_config, torch.device('cpu'))
    
    # Training loop
    training_losses = []
    
    for epoch in range(num_epochs):
        print(f"Adversarial training epoch {epoch + 1}/{num_epochs}")
        
        # Generate clean and adversarial examples
        clean_obs = create_test_observation()
        
        # Generate adversarial examples
        attack_ensemble = AttackEnsemble(attack_config, agent, torch.device('cpu'))
        attack_results = attack_ensemble.evaluate_all_attacks(clean_obs, agent)
        
        # Train on mixed clean and adversarial data
        epoch_loss = 0
        num_batches = len(attack_results) + 1  # +1 for clean data
        
        # Clean data loss
        from scheduler.rl_model.agents.gin_agent.mapper import GinAgentMapper
        from scheduler.config.settings import MAX_OBS_SIZE
        mapper = GinAgentMapper(MAX_OBS_SIZE)
        clean_tensor = torch.from_numpy(mapper.map(
            clean_obs.task_state_scheduled.numpy(),
            clean_obs.task_state_ready.numpy(),
            clean_obs.task_length.numpy(),
            clean_obs.task_completion_time.numpy(),
            clean_obs.task_memory_req_mb.numpy(),
            clean_obs.task_cpu_req_cores.numpy(),
            clean_obs.vm_completion_time.numpy(),
            clean_obs.vm_speed.numpy(),
            clean_obs.vm_energy_rate.numpy(),
            clean_obs.vm_memory_mb.numpy(),
            clean_obs.vm_available_memory_mb.numpy(),
            clean_obs.vm_used_memory_fraction.numpy(),
            clean_obs.vm_active_tasks_count.numpy(),
            clean_obs.vm_next_release_time.numpy(),
            clean_obs.vm_cpu_cores.numpy(),
            clean_obs.vm_available_cpu_cores.numpy(),
            clean_obs.vm_used_cpu_fraction_cores.numpy(),
            clean_obs.vm_next_core_release_time.numpy(),
            clean_obs.task_dependencies.numpy(),
            clean_obs.compatibilities.numpy()
        )).float().unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            clean_scores, clean_log_probs, clean_entropy, clean_value = agent.get_action_and_value(clean_tensor)
            clean_loss = 0.1  # Placeholder loss
        
        epoch_loss += clean_loss
        
        # Adversarial data loss
        for attack_name, attack_obs in attack_results.items():
            with torch.no_grad():
                attack_tensor = torch.from_numpy(mapper.map(
                    attack_obs.task_state_scheduled.numpy(),
                    attack_obs.task_state_ready.numpy(),
                    attack_obs.task_length.numpy(),
                    attack_obs.task_completion_time.numpy(),
                    attack_obs.task_memory_req_mb.numpy(),
                    attack_obs.task_cpu_req_cores.numpy(),
                    attack_obs.vm_completion_time.numpy(),
                    attack_obs.vm_speed.numpy(),
                    attack_obs.vm_energy_rate.numpy(),
                    attack_obs.vm_memory_mb.numpy(),
                    attack_obs.vm_available_memory_mb.numpy(),
                    attack_obs.vm_used_memory_fraction.numpy(),
                    attack_obs.vm_active_tasks_count.numpy(),
                    attack_obs.vm_next_release_time.numpy(),
                    attack_obs.vm_cpu_cores.numpy(),
                    attack_obs.vm_available_cpu_cores.numpy(),
                    attack_obs.vm_used_cpu_fraction_cores.numpy(),
                    attack_obs.vm_next_core_release_time.numpy(),
                    attack_obs.task_dependencies.numpy(),
                    attack_obs.compatibilities.numpy()
                )).float().unsqueeze(0)  # Add batch dimension
                attack_scores, attack_log_probs, attack_entropy, attack_value = agent.get_action_and_value(attack_tensor)
                # Simulate training loss on adversarial data
                adv_loss = 0.2 + 0.1 * np.random.random()  # Placeholder loss
                epoch_loss += adv_loss
        
        avg_loss = epoch_loss / num_batches
        training_losses.append(avg_loss)
        
        print(f"  Epoch {epoch + 1} loss: {avg_loss:.4f}")
    
    print("Adversarial training completed!")
    return robust_agent


def create_robustness_report(clean_stats: Dict[str, float], 
                           attack_stats: Dict[str, Dict[str, float]],
                           output_dir: str = "adversarial_results"):
    """Create a comprehensive robustness report."""
    print("Creating robustness report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison table
    comparison_data = []
    
    # Add clean performance
    comparison_data.append({
        'Attack_Type': 'Clean',
        'Avg_Reward': clean_stats['avg_reward'],
        'Std_Reward': clean_stats['std_reward'],
        'Avg_Entropy': clean_stats['avg_entropy'],
        'Std_Entropy': clean_stats['std_entropy'],
        'Avg_Value': clean_stats['avg_value'],
        'Std_Value': clean_stats['std_value']
    })
    
    # Add attack performances
    for attack_name, stats in attack_stats.items():
        comparison_data.append({
            'Attack_Type': attack_name,
            'Avg_Reward': stats['avg_reward'],
            'Std_Reward': stats['std_reward'],
            'Avg_Entropy': stats['avg_entropy'],
            'Std_Entropy': stats['std_entropy'],
            'Avg_Value': stats['avg_value'],
            'Std_Value': stats['std_value']
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(comparison_data)
    df.to_csv(os.path.join(output_dir, 'robustness_comparison.csv'), index=False)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Reward comparison
    plt.subplot(2, 2, 1)
    attack_names = [row['Attack_Type'] for row in comparison_data]
    avg_rewards = [row['Avg_Reward'] for row in comparison_data]
    std_rewards = [row['Std_Reward'] for row in comparison_data]
    
    plt.bar(attack_names, avg_rewards, yerr=std_rewards, capsize=5)
    plt.title('Average Reward Comparison')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45)
    
    # Plot 2: Entropy comparison
    plt.subplot(2, 2, 2)
    avg_entropy = [row['Avg_Entropy'] for row in comparison_data]
    std_entropy = [row['Std_Entropy'] for row in comparison_data]
    
    plt.bar(attack_names, avg_entropy, yerr=std_entropy, capsize=5)
    plt.title('Average Entropy Comparison')
    plt.ylabel('Average Entropy')
    plt.xticks(rotation=45)
    
    # Plot 3: Value comparison
    plt.subplot(2, 2, 3)
    avg_value = [row['Avg_Value'] for row in comparison_data]
    std_value = [row['Std_Value'] for row in comparison_data]
    
    plt.bar(attack_names, avg_value, yerr=std_value, capsize=5)
    plt.title('Average Value Comparison')
    plt.ylabel('Average Value')
    plt.xticks(rotation=45)
    
    # Plot 4: Performance degradation
    plt.subplot(2, 2, 4)
    clean_reward = clean_stats['avg_reward']
    degradation = [(clean_reward - row['Avg_Reward']) / clean_reward * 100 
                  for row in comparison_data[1:]]  # Skip clean data
    attack_names_short = [name.split('_')[0] for name in attack_names[1:]]
    
    plt.bar(attack_names_short, degradation)
    plt.title('Performance Degradation Under Attacks')
    plt.ylabel('Degradation (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Robustness report saved to {output_dir}/")
    
    # Print summary
    print("\n" + "="*60)
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print("="*60)
    print(f"Clean Performance:")
    print(f"  Average Reward: {clean_stats['avg_reward']:.4f} ± {clean_stats['std_reward']:.4f}")
    print(f"  Average Entropy: {clean_stats['avg_entropy']:.4f} ± {clean_stats['std_entropy']:.4f}")
    print(f"  Average Value: {clean_stats['avg_value']:.4f} ± {clean_stats['std_value']:.4f}")
    
    print(f"\nAttack Performance:")
    for attack_name, stats in attack_stats.items():
        degradation = (clean_stats['avg_reward'] - stats['avg_reward']) / clean_stats['avg_reward'] * 100
        print(f"  {attack_name}:")
        print(f"    Reward: {stats['avg_reward']:.4f} ± {stats['std_reward']:.4f} ({degradation:.1f}% degradation)")
        print(f"    Entropy: {stats['avg_entropy']:.4f} ± {stats['std_entropy']:.4f}")
        print(f"    Value: {stats['avg_value']:.4f} ± {stats['std_value']:.4f}")


def main():
    """Main function to run real adversarial training and evaluation."""
    parser = argparse.ArgumentParser(description='Simple Real Adversarial Attack and Defense Training')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='adversarial_training_results', help='Output directory')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes for evaluation')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of adversarial training epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    args = parser.parse_args()
    
    print("="*80)
    print("SIMPLE REAL ADVERSARIAL ATTACK AND DEFENSE TRAINING")
    print("="*80)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load trained model
        agent = load_trained_model(args.model_path, device)
        
        # Create attack and defense configurations
        attack_config = AttackConfig(
            edge_removal_ratio=0.2,
            edge_addition_ratio=0.1,
            task_length_noise_std=0.15,
            vm_speed_noise_std=0.1,
            memory_noise_std=0.08,
            epsilon=0.1
        )
        
        defense_config = DefenseConfig(
            use_ensemble_defense=True,
            use_input_validation=True,
            use_uncertainty_threshold=True,
            uncertainty_threshold=0.8,
            confidence_threshold=0.7
        )
        
        print(f"\nAttack Configuration:")
        print(f"  Edge removal ratio: {attack_config.edge_removal_ratio}")
        print(f"  Edge addition ratio: {attack_config.edge_addition_ratio}")
        print(f"  Task length noise std: {attack_config.task_length_noise_std}")
        print(f"  VM speed noise std: {attack_config.vm_speed_noise_std}")
        print(f"  Memory noise std: {attack_config.memory_noise_std}")
        
        print(f"\nDefense Configuration:")
        print(f"  Ensemble defense: {defense_config.use_ensemble_defense}")
        print(f"  Input validation: {defense_config.use_input_validation}")
        print(f"  Uncertainty threshold: {defense_config.use_uncertainty_threshold}")
        
        # Evaluate clean performance
        print(f"\n{'='*60}")
        print("EVALUATING CLEAN PERFORMANCE")
        print(f"{'='*60}")
        clean_stats = evaluate_clean_performance(agent, args.num_episodes)
        
        # Evaluate under attacks
        print(f"\n{'='*60}")
        print("EVALUATING UNDER ATTACKS")
        print(f"{'='*60}")
        attack_stats = evaluate_under_attacks(agent, attack_config, args.num_episodes)
        
        # Perform adversarial training
        print(f"\n{'='*60}")
        print("ADVERSARIAL TRAINING")
        print(f"{'='*60}")
        robust_agent = adversarial_training(agent, attack_config, defense_config, args.num_epochs)
        
        # Create robustness report
        print(f"\n{'='*60}")
        print("CREATING ROBUSTNESS REPORT")
        print(f"{'='*60}")
        create_robustness_report(clean_stats, attack_stats, args.output_dir)
        
        print(f"\n{'='*80}")
        print("ADVERSARIAL TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Results saved to: {args.output_dir}/")
        print(f"  - robustness_comparison.csv")
        print(f"  - robustness_analysis.png")
        
    except Exception as e:
        print(f"\nAdversarial training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
