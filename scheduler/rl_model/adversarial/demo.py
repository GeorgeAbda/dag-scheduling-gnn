"""
Demonstration script for adversarial attacks and defenses.

This script demonstrates how to use the adversarial attack and defense
frameworks for evaluating RL-based cloud scheduling agents.
"""

import torch
import numpy as np
import os
import sys
import argparse
from typing import Dict, Any
import time

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
project_root = os.path.dirname(grandparent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scheduler.rl_model.agents.gin_agent.agent import GinAgent
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.dataset_generator.gen_dataset import generate_dataset, DatasetArgs

from scheduler.rl_model.adversarial.attacks import (
    AttackConfig, AttackEnsemble, EdgeRemovalAttack, EdgeAdditionAttack,
    TaskLengthAttack, VMSpeedAttack, MemoryAttack, RedTeamAttack
)
from scheduler.rl_model.adversarial.defenses import (
    DefenseConfig, AdversarialTrainingDefense, EnsembleDefense,
    InputValidationDefense, RobustAgent
)
from scheduler.rl_model.adversarial.evaluation import (
    EvaluationConfig, RobustnessEvaluator, run_robustness_evaluation
)
from scheduler.rl_model.adversarial.visualization import (
    VisualizationConfig, AdversarialVisualizer, create_attack_analysis_report
)


def load_trained_agent(model_path: str, device: torch.device) -> GinAgent:
    """Load a trained agent from file."""
    agent = GinAgent(device)
    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.eval()
    return agent


def create_test_dataset(seed: int = 42) -> Any:
    """Create a test dataset for evaluation."""
    return generate_dataset(
        seed=seed,
        host_count=4,
        vm_count=8,
        workflow_count=3,
        gnp_min_n=8,
        gnp_max_n=12,
        max_memory_gb=8,
        min_cpu_speed_mips=1000,
        max_cpu_speed_mips=3000,
        min_task_length=1000,
        max_task_length=20000,
        task_arrival="static",
        dag_method="gnp",
        task_length_dist="uniform",
        arrival_rate=1.0,
        vm_rng_seed=0
    )


def demonstrate_attacks(agent: GinAgent, device: torch.device, output_dir: str = "adversarial_demo"):
    """Demonstrate different types of adversarial attacks."""
    print("=== Demonstrating Adversarial Attacks ===")
    
    # Create test dataset
    dataset = create_test_dataset()
    
    # Create environment
    env = CloudSchedulingGymEnvironment(dataset=dataset, compute_metrics=True)
    env = GinAgentWrapper(env)
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    # Convert to GinAgentObsTensor
    # First get the original EnvObservation from the wrapper
    env_obs = env.prev_obs
    
    # Extract individual components
    task_state_scheduled = torch.tensor([task.assigned_vm_id is not None for task in env_obs.task_observations], dtype=torch.float32)
    task_state_ready = torch.tensor([task.is_ready for task in env_obs.task_observations], dtype=torch.float32)
    task_length = torch.tensor([task.length for task in env_obs.task_observations], dtype=torch.float32)
    task_completion_time = torch.tensor([task.completion_time for task in env_obs.task_observations], dtype=torch.float32)
    task_memory_req_mb = torch.tensor([task.req_memory_mb for task in env_obs.task_observations], dtype=torch.float32)
    task_cpu_req_cores = torch.tensor([task.req_cpu_cores for task in env_obs.task_observations], dtype=torch.float32)
    
    vm_completion_time = torch.tensor([vm.completion_time for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_speed = torch.tensor([vm.cpu_speed_mips for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_energy_rate = torch.tensor([vm.host_power_peak_watt / vm.host_cpu_speed_mips for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_memory_mb = torch.tensor([vm.memory_mb for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_available_memory_mb = torch.tensor([vm.available_memory_mb for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_used_memory_fraction = torch.tensor([(vm.memory_mb - vm.available_memory_mb) / vm.memory_mb for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_active_tasks_count = torch.tensor([vm.active_tasks_count for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_next_release_time = torch.tensor([vm.next_release_time for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_cpu_cores = torch.tensor([vm.cpu_cores for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_available_cpu_cores = torch.tensor([vm.available_cpu_cores for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_used_cpu_fraction_cores = torch.tensor([vm.used_cpu_fraction_cores for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_next_core_release_time = torch.tensor([vm.next_core_release_time for vm in env_obs.vm_observations], dtype=torch.float32)
    
    # Create compatibility edges
    compatibilities = []
    for i, task in enumerate(env_obs.task_observations):
        for j, vm in enumerate(env_obs.vm_observations):
            if (vm.memory_mb >= task.req_memory_mb and 
                vm.available_memory_mb >= task.req_memory_mb and
                vm.cpu_cores >= task.req_cpu_cores and
                vm.available_cpu_cores >= task.req_cpu_cores):
                compatibilities.append([i, j])
    
    compatibilities = torch.tensor(compatibilities, dtype=torch.long).T if compatibilities else torch.empty((2, 0), dtype=torch.long)
    
    # Create task dependencies
    task_dependencies = []
    for i, task in enumerate(env_obs.task_observations):
        for child_id in task.child_ids:
            if child_id < len(env_obs.task_observations):
                task_dependencies.append([i, child_id])
    
    task_dependencies = torch.tensor(task_dependencies, dtype=torch.long).T if task_dependencies else torch.empty((2, 0), dtype=torch.long)
    
    # Create GinAgentObsTensor
    from scheduler.rl_model.agents.gin_agent.mapper import GinAgentObsTensor
    clean_obs = GinAgentObsTensor(
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
    
    # Create attack configuration
    attack_config = AttackConfig(
        edge_removal_ratio=0.1,
        edge_addition_ratio=0.1,
        task_length_noise_std=0.1,
        vm_speed_noise_std=0.1,
        memory_noise_std=0.05,
        epsilon=0.1
    )
    
    # Create attack ensemble
    attack_ensemble = AttackEnsemble(attack_config, agent, device)
    
    # Apply attacks
    print("Applying different attacks...")
    attack_results = attack_ensemble.evaluate_all_attacks(clean_obs, agent)
    
    # Analyze attack effectiveness
    print("\nAttack Effectiveness Analysis:")
    with torch.no_grad():
        clean_scores = agent(clean_obs)
        clean_entropy = -torch.sum(torch.softmax(clean_scores, dim=-1) * 
                                 torch.log(torch.softmax(clean_scores, dim=-1) + 1e-8), dim=-1).item()
        
        for attack_name, attack_obs in attack_results.items():
            attack_scores = agent(attack_obs)
            attack_entropy = -torch.sum(torch.softmax(attack_scores, dim=-1) * 
                                      torch.log(torch.softmax(attack_scores, dim=-1) + 1e-8), dim=-1).item()
            
            entropy_change = attack_entropy - clean_entropy
            print(f"  {attack_name}: Entropy change = {entropy_change:.4f}")
    
    # Create visualizations
    print("\nGenerating attack visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    config = VisualizationConfig(plot_dir=output_dir)
    visualizer = AdversarialVisualizer(config)
    
    visualizer.visualize_attack_impact(clean_obs, attack_results, agent, "attack_impact")
    visualizer.visualize_graph_attacks(clean_obs, attack_results, "graph_attacks")
    visualizer.visualize_feature_perturbations(clean_obs, attack_results, "feature_perturbations")
    
    # Generate analysis report
    report_path = create_attack_analysis_report(clean_obs, attack_results, agent, output_dir)
    print(f"Attack analysis report generated: {report_path}")
    
    env.close()


def demonstrate_defenses(agent: GinAgent, device: torch.device, output_dir: str = "adversarial_demo"):
    """Demonstrate different defense mechanisms."""
    print("\n=== Demonstrating Defense Mechanisms ===")
    
    # Create defense configuration
    defense_config = DefenseConfig(
        use_ensemble_defense=True,
        use_input_validation=True,
        use_uncertainty_threshold=True,
        uncertainty_threshold=0.8,
        confidence_threshold=0.7
    )
    
    # Create robust agent
    attack_config = AttackConfig()
    robust_agent = RobustAgent(agent, defense_config, attack_config, device)
    
    print("Created robust agent with multiple defense mechanisms:")
    print(f"  - Ensemble defense: {defense_config.use_ensemble_defense}")
    print(f"  - Input validation: {defense_config.use_input_validation}")
    print(f"  - Uncertainty threshold: {defense_config.use_uncertainty_threshold}")
    
    # Test robust agent
    dataset = create_test_dataset()
    env = CloudSchedulingGymEnvironment(dataset=dataset, compute_metrics=True)
    env = GinAgentWrapper(env)
    
    obs, _ = env.reset(seed=42)
    
    # Test clean performance
    with torch.no_grad():
        clean_scores = robust_agent(torch.from_numpy(obs.astype(np.float32)).reshape(1, -1))
        clean_entropy = -torch.sum(torch.softmax(clean_scores, dim=-1) * 
                                 torch.log(torch.softmax(clean_scores, dim=-1) + 1e-8), dim=-1).item()
    
    print(f"Clean performance - Entropy: {clean_entropy:.4f}")
    
    # Test under attack
    # Get the original EnvObservation from the wrapper
    env_obs = env.prev_obs
    
    # Extract individual components (same as in demonstrate_attacks)
    task_state_scheduled = torch.tensor([task.assigned_vm_id is not None for task in env_obs.task_observations], dtype=torch.float32)
    task_state_ready = torch.tensor([task.is_ready for task in env_obs.task_observations], dtype=torch.float32)
    task_length = torch.tensor([task.length for task in env_obs.task_observations], dtype=torch.float32)
    task_completion_time = torch.tensor([task.completion_time for task in env_obs.task_observations], dtype=torch.float32)
    task_memory_req_mb = torch.tensor([task.req_memory_mb for task in env_obs.task_observations], dtype=torch.float32)
    task_cpu_req_cores = torch.tensor([task.req_cpu_cores for task in env_obs.task_observations], dtype=torch.float32)
    
    vm_completion_time = torch.tensor([vm.completion_time for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_speed = torch.tensor([vm.cpu_speed_mips for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_energy_rate = torch.tensor([vm.host_power_peak_watt / vm.host_cpu_speed_mips for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_memory_mb = torch.tensor([vm.memory_mb for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_available_memory_mb = torch.tensor([vm.available_memory_mb for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_used_memory_fraction = torch.tensor([(vm.memory_mb - vm.available_memory_mb) / vm.memory_mb for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_active_tasks_count = torch.tensor([vm.active_tasks_count for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_next_release_time = torch.tensor([vm.next_release_time for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_cpu_cores = torch.tensor([vm.cpu_cores for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_available_cpu_cores = torch.tensor([vm.available_cpu_cores for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_used_cpu_fraction_cores = torch.tensor([vm.used_cpu_fraction_cores for vm in env_obs.vm_observations], dtype=torch.float32)
    vm_next_core_release_time = torch.tensor([vm.next_core_release_time for vm in env_obs.vm_observations], dtype=torch.float32)
    
    # Create compatibility edges
    compatibilities = []
    for i, task in enumerate(env_obs.task_observations):
        for j, vm in enumerate(env_obs.vm_observations):
            if (vm.memory_mb >= task.req_memory_mb and 
                vm.available_memory_mb >= task.req_memory_mb and
                vm.cpu_cores >= task.req_cpu_cores and
                vm.available_cpu_cores >= task.req_cpu_cores):
                compatibilities.append([i, j])
    
    compatibilities = torch.tensor(compatibilities, dtype=torch.long).T if compatibilities else torch.empty((2, 0), dtype=torch.long)
    
    # Create task dependencies
    task_dependencies = []
    for i, task in enumerate(env_obs.task_observations):
        for child_id in task.child_ids:
            if child_id < len(env_obs.task_observations):
                task_dependencies.append([i, child_id])
    
    task_dependencies = torch.tensor(task_dependencies, dtype=torch.long).T if task_dependencies else torch.empty((2, 0), dtype=torch.long)
    
    # Create GinAgentObsTensor
    from scheduler.rl_model.agents.gin_agent.mapper import GinAgentObsTensor
    clean_obs = GinAgentObsTensor(
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
    
    attack_ensemble = AttackEnsemble(attack_config, agent, device)
    attack_results = attack_ensemble.evaluate_all_attacks(clean_obs, agent)
    
    print("\nRobust agent performance under attacks:")
    for attack_name, attack_obs in attack_results.items():
        if attack_name == "clean":
            continue
        
        # Convert back to tensor format for robust agent
        attack_tensor = attack_obs.to_tensor().reshape(1, -1)
        
        with torch.no_grad():
            attack_scores = robust_agent(attack_tensor)
            attack_entropy = -torch.sum(torch.softmax(attack_scores, dim=-1) * 
                                      torch.log(torch.softmax(attack_scores, dim=-1) + 1e-8), dim=-1).item()
        
        entropy_change = attack_entropy - clean_entropy
        print(f"  {attack_name}: Entropy change = {entropy_change:.4f}")
    
    env.close()


def run_comprehensive_evaluation(model_path: str, output_dir: str = "adversarial_evaluation", 
                                num_episodes: int = 50, device: str = "cpu"):
    """Run comprehensive robustness evaluation."""
    print("\n=== Running Comprehensive Robustness Evaluation ===")
    
    # Run evaluation
    results = run_robustness_evaluation(
        model_path=model_path,
        output_dir=output_dir,
        num_episodes=num_episodes,
        device=device
    )
    
    print("Evaluation completed!")
    print(f"Results saved to: {output_dir}")
    
    # Print summary
    if 'robustness_score' in results:
        print(f"Overall robustness score: {results['robustness_score']:.4f}")
    
    if 'makespan_degradation' in results:
        print(f"Average makespan degradation: {results['makespan_degradation']['mean']:.4f}")
    
    if 'energy_degradation' in results:
        print(f"Average energy degradation: {results['energy_degradation']['mean']:.4f}")
    
    return results


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Adversarial Attacks and Defenses Demo")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the trained model")
    parser.add_argument("--output_dir", type=str, default="adversarial_demo",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--num_episodes", type=int, default=20,
                       help="Number of episodes for evaluation")
    parser.add_argument("--demo_attacks", action="store_true",
                       help="Demonstrate attack types")
    parser.add_argument("--demo_defenses", action="store_true",
                       help="Demonstrate defense mechanisms")
    parser.add_argument("--run_evaluation", action="store_true",
                       help="Run comprehensive evaluation")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load trained agent
    print("Loading trained agent...")
    agent = load_trained_agent(args.model_path, device)
    print("Agent loaded successfully!")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run demonstrations
    if args.demo_attacks:
        demonstrate_attacks(agent, device, args.output_dir)
    
    if args.demo_defenses:
        demonstrate_defenses(agent, device, args.output_dir)
    
    if args.run_evaluation:
        run_comprehensive_evaluation(
            args.model_path, 
            args.output_dir, 
            args.num_episodes, 
            args.device
        )
    
    # If no specific demo is requested, run all
    if not any([args.demo_attacks, args.demo_defenses, args.run_evaluation]):
        print("Running all demonstrations...")
        demonstrate_attacks(agent, device, args.output_dir)
        demonstrate_defenses(agent, device, args.output_dir)
        run_comprehensive_evaluation(
            args.model_path, 
            args.output_dir, 
            args.num_episodes, 
            args.device
        )
    
    print("\n=== Demo completed successfully! ===")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

