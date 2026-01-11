"""
Gradient Domain Discovery on RL Benchmarks

This script runs GDD on established RL benchmarks with known domain structure:
1. Synthetic RL domains (controlled experiment)
2. CartPole with physics variants (classic control)
3. Meta-World MT10 (if installed)
4. Procgen (if installed)

Usage:
    python run_rl_experiments.py --benchmark synthetic
    python run_rl_experiments.py --benchmark cartpole
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
    SimplePolicyMLP,
    SimpleDiscretePolicyMLP,
    RLBenchmarkInfo,
)


def collect_rl_gradients(
    envs: Dict[int, Any],
    model: nn.Module,
    collector: GradientCollector,
    num_episodes: int = 10,
    max_steps: int = 100,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Collect policy gradients from multiple RL environments.
    
    Uses REINFORCE-style gradient estimation.
    """
    for env_id, env in envs.items():
        for episode in range(num_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            log_probs = []
            rewards = []
            
            for step in range(max_steps):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                
                # Get action
                if hasattr(env.action_space, 'n'):
                    # Discrete action
                    logits = model(obs_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    action_np = action.item()
                else:
                    # Continuous action
                    mean, std = model(obs_t)
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(-1)
                    action_np = action.squeeze(0).detach().cpu().numpy()
                
                log_probs.append(log_prob)
                
                # Step environment
                result = env.step(action_np)
                if len(result) == 5:
                    next_obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, info = result
                
                rewards.append(reward)
                obs = next_obs
                
                if done:
                    break
            
            # Compute returns
            returns = []
            G = 0
            gamma = 0.99
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            
            returns = torch.FloatTensor(returns).to(device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Compute policy gradient loss
            log_probs_t = torch.stack(log_probs)
            loss = -(log_probs_t * returns).mean()
            
            # Backward to get gradients
            model.zero_grad()
            loss.backward()
            
            # Collect gradient
            collector.collect(env_id, model, episode, loss.item())


def train_rl_with_gradient_collection(
    envs: Dict[int, Any],
    info: RLBenchmarkInfo,
    num_epochs: int = 10,
    episodes_per_epoch: int = 5,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, GradientCollector]:
    """
    Train a policy on multiple RL environments while collecting gradients.
    """
    # Determine observation and action dimensions
    first_env = envs[0]
    obs_dim = first_env.observation_space.shape[0]
    
    if hasattr(first_env.action_space, 'n'):
        act_dim = first_env.action_space.n
        is_discrete = True
        model = SimpleDiscretePolicyMLP(obs_dim, act_dim).to(device)
    else:
        act_dim = first_env.action_space.shape[0]
        is_discrete = False
        model = SimplePolicyMLP(obs_dim, act_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Gradient collector
    collector = GradientCollector(
        num_sources=len(envs),
        max_samples_per_source=num_epochs * episodes_per_epoch,
        use_random_projection=True,
        projection_dim=256,
    )
    
    print(f"\nTraining policy on {len(envs)} environments...")
    print(f"  Obs dim: {obs_dim}, Act dim: {act_dim}, Discrete: {is_discrete}")
    print(f"  Epochs: {num_epochs}, Episodes/epoch: {episodes_per_epoch}")
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_rewards = {i: [] for i in envs.keys()}
        
        for env_id, env in envs.items():
            for episode in range(episodes_per_epoch):
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                
                log_probs = []
                rewards = []
                
                for step in range(200):
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    
                    if is_discrete:
                        logits = model(obs_t)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                        action_np = action.item()
                    else:
                        mean, std = model(obs_t)
                        dist = torch.distributions.Normal(mean, std)
                        action = dist.sample()
                        log_prob = dist.log_prob(action).sum(-1)
                        action_np = action.squeeze(0).detach().cpu().numpy()
                    
                    log_probs.append(log_prob)
                    
                    result = env.step(action_np)
                    if len(result) == 5:
                        next_obs, reward, terminated, truncated, _ = result
                        done = terminated or truncated
                    else:
                        next_obs, reward, done, _ = result
                    
                    rewards.append(reward)
                    obs = next_obs
                    
                    if done:
                        break
                
                epoch_rewards[env_id].append(sum(rewards))
                
                # Compute policy gradient
                returns = []
                G = 0
                for r in reversed(rewards):
                    G = r + 0.99 * G
                    returns.insert(0, G)
                
                returns = torch.FloatTensor(returns).to(device)
                if len(returns) > 1:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                
                log_probs_t = torch.stack(log_probs)
                loss = -(log_probs_t * returns).mean()
                
                # Collect gradient before optimizer step
                model.zero_grad()
                loss.backward()
                collector.collect(env_id, model, epoch * episodes_per_epoch + episode, loss.item())
                
                # Update
                optimizer.step()
        
        # Print progress
        if (epoch + 1) % max(1, num_epochs // 5) == 0:
            mean_rewards = {k: np.mean(v) for k, v in epoch_rewards.items()}
            print(f"  Epoch {epoch+1}: Mean rewards = {mean_rewards}")
    
    return model, collector


def run_rl_domain_discovery(
    benchmark_name: str,
    num_epochs: int = 15,
    episodes_per_epoch: int = 5,
    device: torch.device = torch.device("cpu"),
    **kwargs,
) -> Dict:
    """
    Run domain discovery on an RL benchmark.
    """
    print(f"\n{'='*60}")
    print(f"Gradient Domain Discovery on {benchmark_name}")
    print(f"{'='*60}")
    
    # Load benchmark
    envs, true_labels, info = get_rl_benchmark(benchmark_name, **kwargs)
    
    print(f"\nBenchmark: {info.name}")
    print(f"Description: {info.description}")
    print(f"Number of environments: {len(envs)}")
    print(f"True domains: {info.num_domains}")
    print(f"True labels: {true_labels}")
    print(f"Known conflicts: {info.known_conflicts}")
    
    # Print domain details
    print(f"\nDomains:")
    for d in info.domains:
        print(f"  {d.domain_id}: {d.name} (obs={d.obs_dim}, act={d.act_dim})")
    
    # Train and collect gradients
    model, collector = train_rl_with_gradient_collection(
        envs, info, num_epochs, episodes_per_epoch, device
    )
    
    print(f"\nCollected {collector.total_samples()} gradient samples")
    
    # Run domain discovery
    print("\nRunning Gradient Domain Discovery...")
    
    gdd = GradientDomainDiscovery(
        similarity_metric="cosine",
        clustering_method="spectral",
        num_domains=None,  # Auto-select
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
    elif cross_conf > within_conf:
        print("~ Slightly more cross-domain conflict.")
    else:
        print("✗ Within-domain conflict >= cross-domain conflict.")
    
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
    
    # Check known conflicts
    if info.known_conflicts:
        print(f"\nKnown Conflict Pairs (from literature):")
        for i, j in info.known_conflicts:
            sim = result.similarity_matrix[i, j]
            conf = result.conflict_matrix[i, j]
            print(f"  ({i}, {j}): similarity={sim:.3f}, conflict_rate={conf:.3f}")
    
    return {
        "benchmark": benchmark_name,
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
    parser = argparse.ArgumentParser(description="GDD on RL Benchmarks")
    
    available = list_available_benchmarks()
    
    parser.add_argument(
        "--benchmark",
        type=str,
        default="synthetic",
        choices=available,
        help=f"Benchmark to run. Available: {available}",
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=15,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--episodes_per_epoch",
        type=int,
        default=5,
        help="Episodes per environment per epoch",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Available benchmarks: {available}")
    
    # Run experiment
    results = run_rl_domain_discovery(
        benchmark_name=args.benchmark,
        num_epochs=args.num_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        device=device,
        seed=args.seed,
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"rl_{args.benchmark}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
