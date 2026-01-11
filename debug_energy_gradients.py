"""
Debug why energy gradients are zero.
"""

import os
import torch
import numpy as np

from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.dataset_generator.gen_dataset import DatasetArgs


def test_energy_collection():
    """Test if environment returns energy metrics."""
    
    # Create simple MDP
    args = DatasetArgs(
        host_count=10,
        vm_count=10,
        workflow_count=1,
        style="wide",
        gnp_p=0.1,
        gnp_min_n=20,
        gnp_max_n=30,
        min_task_length=1000,
        max_task_length=5000,
    )
    
    os.environ['HOST_SPECS_PATH'] = "data/host_specs_NAL.json"
    
    env = GinAgentWrapper(CloudSchedulingGymEnvironment(
        dataset_args=args, collect_timelines=False, compute_metrics=True
    ))
    
    # Initialize agent
    variant = AblationVariant(
        name="hetero", graph_type="hetero", gin_num_layers=2,
        use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
    )
    agent = AblationGinAgent(device=torch.device("cpu"), variant=variant, hidden_dim=128, embedding_dim=64)
    
    # Run one episode
    obs, _ = env.reset(seed=42)
    
    episode_count = 0
    step_count = 0
    
    for _ in range(500):
        obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(1, -1))
        
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        
        obs, reward, terminated, truncated, info = env.step(int(action.item()))
        step_count += 1
        
        if terminated or truncated:
            episode_count += 1
            print(f"\nEpisode {episode_count} completed (step {step_count}):")
            print(f"  Info keys: {list(info.keys())}")
            print(f"  Makespan: {info.get('makespan', 'NOT FOUND')}")
            print(f"  Active energy: {info.get('active_energy', 'NOT FOUND')}")
            print(f"  Total energy: {info.get('total_energy', 'NOT FOUND')}")
            print(f"  Reward: {reward}")
            
            if episode_count >= 3:
                break
            
            obs, _ = env.reset()
    
    env.close()


if __name__ == "__main__":
    test_energy_collection()
