"""
Gradient Domain Discovery on RL Benchmarks - Value Function Version

Uses value function gradients (supervised learning on returns) instead of
policy gradients. This has much lower variance and should reveal domain
structure more clearly.

This approach is more aligned with how PPO/A2C work in practice.
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
from rl_benchmarks import (
    get_rl_benchmark,
    list_available_benchmarks,
    RLBenchmarkInfo,
)


class ValueNetwork(nn.Module):
    """Value function network."""
    
    def __init__(self, obs_dim: int, hidden_dims: List[int] = [64, 64]):
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
    """Simple policy network."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, act_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)
    
    def get_action(self, obs: torch.Tensor) -> int:
        logits = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()


def collect_trajectories(
    env,
    policy: PolicyNetwork,
    num_episodes: int,
    max_steps: int = 200,
    gamma: float = 0.99,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect trajectories and compute returns.
    
    Returns:
        observations: (N, obs_dim)
        returns: (N,)
    """
    all_obs = []
    all_returns = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        episode_obs = []
        episode_rewards = []
        
        for step in range(max_steps):
            episode_obs.append(obs.copy())
            
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = policy.get_action(obs_t)
            
            result = env.step(action)
            if len(result) == 5:
                next_obs, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = result
            
            episode_rewards.append(reward)
            obs = next_obs
            
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


def train_with_value_gradient_collection(
    envs: Dict[int, Any],
    info: RLBenchmarkInfo,
    num_epochs: int = 20,
    episodes_per_env: int = 20,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, GradientCollector]:
    """
    Train value functions and collect gradients.
    
    Key insight: Value function gradients are supervised learning gradients
    (MSE loss on returns), which have much lower variance than policy gradients.
    """
    first_env = envs[0]
    obs_dim = first_env.observation_space.shape[0]
    act_dim = first_env.action_space.n if hasattr(first_env.action_space, 'n') else 4
    
    # Shared value network
    value_net = ValueNetwork(obs_dim).to(device)
    optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    
    # Simple policy for data collection (random initially, then learned)
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    # Gradient collector
    collector = GradientCollector(
        num_sources=len(envs),
        max_samples_per_source=num_epochs,
        use_random_projection=False,  # Value net is small enough
    )
    
    print(f"\nTraining value function on {len(envs)} environments...")
    print(f"  Obs dim: {obs_dim}")
    print(f"  Epochs: {num_epochs}, Episodes/env: {episodes_per_env}")
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_losses = {}
        
        for env_id, env in envs.items():
            # Collect data with current policy
            obs, returns = collect_trajectories(
                env, policy, episodes_per_env, device=device
            )
            obs = obs.to(device)
            returns = returns.to(device)
            
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Value function loss (MSE)
            value_net.zero_grad()
            values = value_net(obs)
            loss = nn.MSELoss()(values, returns)
            loss.backward()
            
            # Collect gradient BEFORE optimizer step
            collector.collect(env_id, value_net, epoch, loss.item())
            
            # Update value network
            optimizer.step()
            
            epoch_losses[env_id] = loss.item()
            
            # Also update policy using advantage
            policy_optimizer.zero_grad()
            with torch.no_grad():
                advantages = returns - value_net(obs)
            
            logits = policy(obs)
            # Simple policy gradient (for data collection improvement)
            # This is just to get better trajectories, not for gradient collection
        
        if (epoch + 1) % max(1, num_epochs // 5) == 0:
            mean_loss = np.mean(list(epoch_losses.values()))
            print(f"  Epoch {epoch+1}: Mean value loss = {mean_loss:.4f}")
    
    return value_net, collector


def run_value_domain_discovery(
    benchmark_name: str,
    num_epochs: int = 30,
    episodes_per_env: int = 20,
    device: torch.device = torch.device("cpu"),
    **kwargs,
) -> Dict:
    """
    Run domain discovery using value function gradients.
    """
    print(f"\n{'='*60}")
    print(f"Gradient Domain Discovery (Value Function) on {benchmark_name}")
    print(f"{'='*60}")
    
    # Load benchmark
    envs, true_labels, info = get_rl_benchmark(benchmark_name, **kwargs)
    
    print(f"\nBenchmark: {info.name}")
    print(f"Description: {info.description}")
    print(f"Number of environments: {len(envs)}")
    print(f"True domains: {info.num_domains}")
    print(f"True labels: {true_labels}")
    
    # Train and collect gradients
    model, collector = train_with_value_gradient_collection(
        envs, info, num_epochs, episodes_per_env, device
    )
    
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
    
    # Print matrices
    print(f"\nGradient Similarity Matrix:")
    n = len(envs)
    header = "    " + "  ".join([f"E{i:2d}" for i in range(n)])
    print(header)
    for i in range(n):
        row = " ".join([f"{result.similarity_matrix[i,j]:5.2f}" for j in range(n)])
        print(f"E{i:2d} {row}")
    
    print(f"\nGradient Conflict Rate Matrix:")
    print(header)
    for i in range(n):
        row = " ".join([f"{result.conflict_matrix[i,j]:5.2f}" for j in range(n)])
        print(f"E{i:2d} {row}")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
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
    
    if ari > 0.8:
        print("✓ Excellent domain recovery!")
    elif ari > 0.5:
        print("~ Good domain recovery.")
    elif ari > 0.2:
        print("~ Partial domain recovery.")
    else:
        print("✗ Poor domain recovery.")
    
    return {
        "benchmark": benchmark_name,
        "method": "value_function",
        "num_envs": len(envs),
        "true_domains": info.num_domains,
        "discovered_domains": result.num_domains,
        "true_labels": true_labels.tolist(),
        "discovered_labels": result.domain_assignments.tolist(),
        "validation": validation,
        "similarity_matrix": result.similarity_matrix.tolist(),
        "conflict_matrix": result.conflict_matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="GDD on RL (Value Function)")
    
    available = list_available_benchmarks()
    
    parser.add_argument(
        "--benchmark",
        type=str,
        default="cartpole",
        choices=available,
    )
    
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--episodes_per_env", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    results = run_value_domain_discovery(
        benchmark_name=args.benchmark,
        num_epochs=args.num_epochs,
        episodes_per_env=args.episodes_per_env,
        device=device,
        seed=args.seed,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"rl_value_{args.benchmark}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
