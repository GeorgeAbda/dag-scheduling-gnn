"""
Gradient Domain Discovery on Meta-World MT10

Meta-World is a standard benchmark for multi-task RL with 50 robotic manipulation tasks.
MT10 uses 10 tasks that are known to have different optimal policies.

This is a well-established benchmark from:
"Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning"
(Yu et al., 2020)

The 10 tasks in MT10:
1. reach-v2: Reach a target position
2. push-v2: Push an object to a target
3. pick-place-v2: Pick up and place an object
4. door-open-v2: Open a door
5. drawer-open-v2: Open a drawer
6. drawer-close-v2: Close a drawer
7. button-press-topdown-v2: Press a button from above
8. peg-insert-side-v2: Insert a peg from the side
9. window-open-v2: Open a window
10. window-close-v2: Close a window

These tasks have known groupings based on required skills:
- Reaching tasks: reach
- Pushing/placing: push, pick-place
- Door/window manipulation: door-open, window-open, window-close
- Drawer manipulation: drawer-open, drawer-close
- Precision tasks: button-press, peg-insert
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from gdd_core import (
    GradientCollector,
    GradientDomainDiscovery,
    validate_domain_discovery,
)

# Check for metaworld
try:
    import metaworld
    HAS_METAWORLD = True
except ImportError:
    HAS_METAWORLD = False
    print("ERROR: metaworld not installed.")
    print("Install with: pip install metaworld mujoco")
    sys.exit(1)


# MT10 task groupings based on skill similarity
# Updated for v3 task names
MT10_TASK_GROUPS = {
    "reaching": ["reach-v2", "reach-v3"],
    "pushing": ["push-v2", "pick-place-v2", "push-v3", "pick-place-v3"],
    "doors_windows": ["door-open-v2", "window-open-v2", "window-close-v2",
                      "door-open-v3", "window-open-v3", "window-close-v3"],
    "drawers": ["drawer-open-v2", "drawer-close-v2", 
                "drawer-open-v3", "drawer-close-v3"],
    "precision": ["button-press-topdown-v2", "peg-insert-side-v2",
                  "button-press-topdown-v3", "peg-insert-side-v3"],
}

# Create domain labels based on groupings
def get_domain_labels(task_names: List[str]) -> np.ndarray:
    """Get domain labels for tasks based on skill groupings."""
    labels = []
    group_to_id = {g: i for i, g in enumerate(MT10_TASK_GROUPS.keys())}
    
    for task in task_names:
        found = False
        for group, tasks in MT10_TASK_GROUPS.items():
            if task in tasks:
                labels.append(group_to_id[group])
                found = True
                break
        if not found:
            labels.append(-1)
    
    return np.array(labels)


class ValueNetwork(nn.Module):
    """Value function network for Meta-World."""
    
    def __init__(self, obs_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs).squeeze(-1)


class PolicyNetwork(nn.Module):
    """Simple policy network for Meta-World (continuous actions)."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        
        self.backbone = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mean = torch.tanh(self.mean(features))  # Actions in [-1, 1]
        std = self.log_std.exp()
        return mean, std
    
    def get_action(self, obs: torch.Tensor) -> np.ndarray:
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.squeeze(0).detach().cpu().numpy().clip(-1, 1)


def collect_trajectories(
    env,
    policy: PolicyNetwork,
    num_episodes: int,
    max_steps: int = 150,
    gamma: float = 0.99,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect trajectories and compute returns."""
    all_obs = []
    all_returns = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        
        episode_obs = []
        episode_rewards = []
        
        for step in range(max_steps):
            episode_obs.append(obs.copy())
            
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = policy.get_action(obs_t)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_rewards.append(reward)
            
            if done:
                break
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        all_obs.extend(episode_obs)
        all_returns.extend(returns)
    
    return (
        torch.FloatTensor(np.array(all_obs)),
        torch.FloatTensor(all_returns),
    )


def run_metaworld_domain_discovery(
    num_epochs: int = 20,
    episodes_per_task: int = 10,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Run domain discovery on Meta-World MT10.
    """
    print(f"\n{'='*70}")
    print("Gradient Domain Discovery on Meta-World MT10")
    print(f"{'='*70}")
    
    # Create MT10 benchmark
    print("\nLoading Meta-World MT10...")
    mt10 = metaworld.MT10(seed=seed)
    
    # Get task names and create environments
    task_names = list(mt10.train_classes.keys())
    print(f"Tasks: {task_names}")
    
    # Create environments
    envs = {}
    for i, (name, env_cls) in enumerate(mt10.train_classes.items()):
        env = env_cls()
        task = [t for t in mt10.train_tasks if t.env_name == name][0]
        env.set_task(task)
        envs[i] = (name, env)
    
    num_tasks = len(envs)
    
    # Get domain labels
    true_labels = get_domain_labels(task_names)
    print(f"\nTrue domain labels (by skill group): {true_labels}")
    print(f"Number of domains: {len(set(true_labels))}")
    
    # Get observation and action dimensions
    first_env = envs[0][1]
    obs_dim = first_env.observation_space.shape[0]
    act_dim = first_env.action_space.shape[0]
    
    print(f"\nObs dim: {obs_dim}, Act dim: {act_dim}")
    
    # Create networks
    value_net = ValueNetwork(obs_dim).to(device)
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    
    optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    
    # Gradient collector
    collector = GradientCollector(
        num_sources=num_tasks,
        max_samples_per_source=num_epochs,
        use_random_projection=True,
        projection_dim=512,
    )
    
    print(f"\nTraining for {num_epochs} epochs, {episodes_per_task} episodes/task...")
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_rewards = {}
        
        for task_id, (task_name, env) in envs.items():
            # Collect trajectories
            obs, returns = collect_trajectories(
                env, policy, episodes_per_task, device=device
            )
            obs = obs.to(device)
            returns = returns.to(device)
            
            # Normalize returns
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Value function loss
            value_net.zero_grad()
            values = value_net(obs)
            loss = nn.MSELoss()(values, returns)
            loss.backward()
            
            # Collect gradient
            collector.collect(task_id, value_net, epoch, loss.item())
            
            # Update
            optimizer.step()
            
            epoch_rewards[task_name] = returns.mean().item()
        
        if (epoch + 1) % max(1, num_epochs // 4) == 0:
            mean_return = np.mean(list(epoch_rewards.values()))
            print(f"  Epoch {epoch+1}: Mean return = {mean_return:.4f}")
    
    print(f"\nCollected {collector.total_samples()} gradient samples")
    
    # Run domain discovery
    print("\nRunning Gradient Domain Discovery...")
    
    gdd = GradientDomainDiscovery(
        similarity_metric="cosine",
        clustering_method="spectral",
        num_domains=None,
    )
    
    result = gdd.discover_domains(collector)
    
    print(f"\n{result.summary()}")
    
    # Validate
    validation = validate_domain_discovery(result, true_labels)
    
    print(f"\nValidation Metrics:")
    for key, value in validation.items():
        print(f"  {key}: {value:.4f}")
    
    # Print similarity matrix with task names
    print(f"\nGradient Similarity Matrix:")
    short_names = [n.replace("-v2", "")[:8] for n in task_names]
    header = "         " + " ".join([f"{n:>8}" for n in short_names])
    print(header)
    for i in range(num_tasks):
        row = " ".join([f"{result.similarity_matrix[i,j]:8.2f}" for j in range(num_tasks)])
        print(f"{short_names[i]:>8} {row}")
    
    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    within_sim = validation["same_domain_similarity"]
    cross_sim = validation["diff_domain_similarity"]
    gap = within_sim - cross_sim
    
    print(f"\nWithin-domain similarity: {within_sim:.4f}")
    print(f"Cross-domain similarity:  {cross_sim:.4f}")
    print(f"Separability gap:         {gap:.4f}")
    
    if gap > 0.2:
        print("✓ Strong domain separation detected!")
    elif gap > 0.1:
        print("~ Moderate domain separation detected.")
    elif gap > 0:
        print("~ Weak domain separation detected.")
    else:
        print("✗ No domain separation (cross > within).")
    
    within_conf = validation["same_domain_conflict"]
    cross_conf = validation["diff_domain_conflict"]
    
    print(f"\nWithin-domain conflict rate: {within_conf:.4f}")
    print(f"Cross-domain conflict rate:  {cross_conf:.4f}")
    
    if cross_conf > within_conf + 0.1:
        print("✓ Cross-domain gradients conflict more than within-domain!")
    
    ari = validation["adjusted_rand_index"]
    print(f"\nAdjusted Rand Index: {ari:.4f}")
    
    if ari > 0.5:
        print("✓ Good domain recovery!")
    elif ari > 0.2:
        print("~ Partial domain recovery.")
    else:
        print("✗ Poor domain recovery.")
    
    # Print discovered clusters
    print(f"\nDiscovered Clusters:")
    for cluster_id in range(result.num_domains):
        tasks_in_cluster = [task_names[i] for i in range(num_tasks) 
                          if result.domain_assignments[i] == cluster_id]
        print(f"  Cluster {cluster_id}: {tasks_in_cluster}")
    
    return {
        "benchmark": "metaworld_mt10",
        "task_names": task_names,
        "num_tasks": num_tasks,
        "true_domains": len(set(true_labels)),
        "discovered_domains": result.num_domains,
        "true_labels": true_labels.tolist(),
        "discovered_labels": result.domain_assignments.tolist(),
        "validation": validation,
        "similarity_matrix": result.similarity_matrix.tolist(),
        "conflict_matrix": result.conflict_matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="GDD on Meta-World MT10")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--episodes_per_task", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    results = run_metaworld_domain_discovery(
        num_epochs=args.num_epochs,
        episodes_per_task=args.episodes_per_task,
        seed=args.seed,
        device=device,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "metaworld_mt10_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
