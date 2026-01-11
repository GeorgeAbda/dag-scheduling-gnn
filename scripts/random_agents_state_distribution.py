#!/usr/bin/env python3
"""
Run multiple randomly initialized agents on episodes from wide and longcp datasets,
collect state observations, and visualize using PCA and t-SNE.

Usage:
    python scripts/random_agents_state_distribution.py \
        --n-agents 5 \
        --n-episodes-per-domain 10 \
        --out-dir logs/random_agent_states
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import tyro

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.agents.gin_agent.agent import GinAgent


@dataclass
class Args:
    """Arguments for random agent state distribution analysis."""
    n_agents: int = 5
    """Number of randomly initialized agents to run"""
    n_episodes_per_domain: int = 10
    """Number of episodes to sample from each domain (wide, longcp)"""
    longcp_config: str = "data/rl_configs/train_long_cp_p08_seeds.json"
    """Path to longcp config JSON"""
    wide_config: str = "data/rl_configs/train_wide_p005_seeds.json"
    """Path to wide config JSON"""
    out_dir: str = "logs/random_agent_states"
    """Output directory for plots"""
    max_steps_per_episode: int = 10000
    """Maximum steps per episode (safety guard)"""
    pca_components: int = 2
    """Number of PCA components"""
    tsne_components: int = 2
    """Number of t-SNE components"""
    tsne_perplexity: float = 30.0
    """t-SNE perplexity parameter"""
    device: str = "cpu"
    """Device to run on (cpu/cuda)"""
    seed: int = 42
    """Random seed for reproducibility"""
    feature_mode: str = "summary"
    """Feature mode: 'mapped' uses full mapped observation; 'summary' uses compact features; 'features_only' aggregates raw task/VM feature stats"""


def _load_seedset(path: str, source: str = "eval") -> Tuple[List[int], dict]:
    """Load seeds and dataset config from JSON."""
    with open(path, "r") as f:
        cfg = json.load(f)
    section = cfg.get(source, {}) if isinstance(cfg, dict) else {}
    seeds: List[int] = list(section.get("seeds", []))
    ds: dict = dict(section.get("dataset", {}))
    return seeds, ds


def _mk_dataset(seed: int, ds: dict):
    """Generate a dataset from seed and config."""
    return generate_dataset(
        seed=seed,
        host_count=int(ds.get("host_count", 10)),
        vm_count=int(ds.get("vm_count", 10)),
        max_memory_gb=int(ds.get("max_memory_gb", 128)),
        min_cpu_speed_mips=int(ds.get("min_cpu_speed", 500)),
        max_cpu_speed_mips=int(ds.get("max_cpu_speed", 5000)),
        workflow_count=int(ds.get("workflow_count", 1)),
        dag_method=str(ds.get("dag_method", "gnp")),
        gnp_min_n=int(ds.get("gnp_min_n", 12)),
        gnp_max_n=int(ds.get("gnp_max_n", 24)),
        task_length_dist=str(ds.get("task_length_dist", "normal")),
        min_task_length=int(ds.get("min_task_length", 500)),
        max_task_length=int(ds.get("max_task_length", 100000)),
        task_arrival=str(ds.get("task_arrival", "static")),
        arrival_rate=float(ds.get("arrival_rate", 3)),
        vm_rng_seed=0,
        gnp_p=float(ds.get("gnp_p")) if ds.get("gnp_p") is not None else None,
    )


# -------------------------------
# Domain-invariant summary features
# -------------------------------

def _topo_layers(children: Dict[int, List[int]]) -> List[List[int]]:
    indeg: Dict[int, int] = {u: 0 for u in children}
    for u, vs in children.items():
        for v in vs:
            indeg[v] = indeg.get(v, 0) + 1
    frontier: List[int] = [u for u in children if indeg.get(u, 0) == 0]
    layers: List[List[int]] = []
    seen: set[int] = set()
    while frontier:
        cur = list(frontier)
        layers.append(cur)
        frontier = []
        for u in cur:
            if u in seen:
                continue
            seen.add(u)
            for v in children.get(u, []):
                indeg[v] = indeg.get(v, 0) - 1
                if indeg[v] == 0:
                    frontier.append(v)
    if not layers:
        return [list(children.keys())]
    return layers


def _max_width_from_dataset(dataset) -> int:
    wf = dataset.workflows[0]
    ch: Dict[int, List[int]] = {int(t.id): list(t.child_ids) for t in wf.tasks}
    layers = _topo_layers(ch)
    return max((len(L) for L in layers), default=1)


def _ready_count(prev_obs) -> int:
    T = len(prev_obs.task_observations)
    cnt = 0
    for idx, t in enumerate(prev_obs.task_observations):
        if idx == 0 or idx == T - 1:
            continue
        if bool(t.is_ready) and (t.assigned_vm_id is None):
            cnt += 1
    return int(cnt)


def _energy_intensity_index_prev(prev_obs) -> float:
    total_span = 0.0
    active = 0.0
    for v in prev_obs.vm_observations:
        idle = float(getattr(v, "host_power_idle_watt", 0.0))
        peak = float(getattr(v, "host_power_peak_watt", idle))
        span = max(0.0, peak - idle)
        total_span += span
        cores = max(1.0, float(getattr(v, "cpu_cores", 1.0)))
        avail = float(getattr(v, "available_cpu_cores", cores))
        used = max(0.0, cores - avail)
        u = min(1.0, max(0.0, used / cores))
        active += span * u
    if total_span <= 0.0:
        return 0.0
    return float(active / total_span)


def _summarize_state(prev_obs, width: int) -> np.ndarray:
    """Return compact features less tied to raw counts/positions.
    Features:
      [0] pi = ready_count / max(1,width)
      [1] ei = energy_intensity_index (0..1)
      [2] mean_used_cpu_fraction across VMs (0..1)
      [3] mean_used_mem_fraction across VMs (0..1)
      [4] compat_density = |compat|/(T*V) (0..1)
      [5] active_tasks_density = mean_active_tasks_count / max(1,V)
    """
    T = max(1, len(prev_obs.task_observations))
    V = max(1, len(prev_obs.vm_observations))
    # pi, ei
    pi = float(_ready_count(prev_obs)) / float(max(1, width))
    ei = _energy_intensity_index_prev(prev_obs)
    # VM stats
    used_cpu_fracs = []
    used_mem_fracs = []
    active_counts = []
    for v in prev_obs.vm_observations:
        cores = max(1.0, float(getattr(v, "cpu_cores", 1)))
        avail_cores = float(getattr(v, "available_cpu_cores", cores))
        used_cores = max(0.0, cores - avail_cores)
        used_cpu_fracs.append(min(1.0, max(0.0, used_cores / cores)))
        mem_mb = max(1.0, float(getattr(v, "memory_mb", 1)))
        avail_mb = float(getattr(v, "available_memory_mb", mem_mb))
        used_mem_fracs.append(min(1.0, max(0.0, (mem_mb - avail_mb) / mem_mb)))
        active_counts.append(float(getattr(v, "active_tasks_count", 0)))
    mean_used_cpu = float(np.mean(used_cpu_fracs)) if used_cpu_fracs else 0.0
    mean_used_mem = float(np.mean(used_mem_fracs)) if used_mem_fracs else 0.0
    active_tasks_density = float(np.mean(active_counts)) / float(max(1, V))
    # Compatibility density
    try:
        compat = getattr(prev_obs, "compatibilities", []) or []
        compat_density = float(len(compat)) / float(T * V)
    except Exception:
        compat_density = 0.0
    return np.array([pi, ei, mean_used_cpu, mean_used_mem, compat_density, active_tasks_density], dtype=np.float32)


# Features-only aggregation (permutation-invariant)
def _agg_stats(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros(7, dtype=np.float32)
    q10, q50, q90 = np.percentile(arr, [10, 50, 90])
    return np.array([
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(q10),
        float(q50),
        float(q90),
        float(np.max(arr)),
    ], dtype=np.float32)


def _features_only_state(prev_obs, width: int) -> np.ndarray:
    # Task features
    t_ready = []
    t_sched = []
    t_len = []
    t_mem = []
    t_cores = []
    for idx, t in enumerate(prev_obs.task_observations):
        # Skip dummy tasks if present (first/last)
        if idx == 0 or idx == len(prev_obs.task_observations) - 1:
            continue
        t_ready.append(1.0 if bool(getattr(t, "is_ready", False)) and (getattr(t, "assigned_vm_id", None) is None) else 0.0)
        t_sched.append(1.0 if getattr(t, "assigned_vm_id", None) is not None else 0.0)
        t_len.append(float(getattr(t, "length", 0.0)))
        t_mem.append(float(getattr(t, "req_memory_mb", 0.0)))
        t_cores.append(float(getattr(t, "req_cpu_cores", 0.0)))
    t_ready = np.asarray(t_ready, dtype=np.float32)
    t_sched = np.asarray(t_sched, dtype=np.float32)
    t_len = np.asarray(t_len, dtype=np.float32)
    t_mem = np.asarray(t_mem, dtype=np.float32)
    t_cores = np.asarray(t_cores, dtype=np.float32)

    # VM features
    v_speed = []
    v_idle = []
    v_peak = []
    v_cpu = []
    v_cpu_avail = []
    v_cpu_used_frac = []
    v_mem = []
    v_mem_avail = []
    v_mem_used_frac = []
    v_active = []
    for v in prev_obs.vm_observations:
        cpu_total = float(max(1, getattr(v, "cpu_cores", 1)))
        cpu_avail = float(getattr(v, "available_cpu_cores", cpu_total))
        mem_total = float(max(1, getattr(v, "memory_mb", 1)))
        mem_avail = float(getattr(v, "available_memory_mb", mem_total))
        v_speed.append(float(getattr(v, "cpu_speed_mips", 0.0)))
        v_idle.append(float(getattr(v, "host_power_idle_watt", 0.0)))
        v_peak.append(float(getattr(v, "host_power_peak_watt", 0.0)))
        v_cpu.append(cpu_total)
        v_cpu_avail.append(cpu_avail)
        v_cpu_used_frac.append(min(1.0, max(0.0, (cpu_total - cpu_avail) / cpu_total)))
        v_mem.append(mem_total)
        v_mem_avail.append(mem_avail)
        v_mem_used_frac.append(min(1.0, max(0.0, (mem_total - mem_avail) / mem_total)))
        v_active.append(float(getattr(v, "active_tasks_count", 0.0)))
    v_speed = np.asarray(v_speed, dtype=np.float32)
    v_idle = np.asarray(v_idle, dtype=np.float32)
    v_peak = np.asarray(v_peak, dtype=np.float32)
    v_cpu = np.asarray(v_cpu, dtype=np.float32)
    v_cpu_avail = np.asarray(v_cpu_avail, dtype=np.float32)
    v_cpu_used_frac = np.asarray(v_cpu_used_frac, dtype=np.float32)
    v_mem = np.asarray(v_mem, dtype=np.float32)
    v_mem_avail = np.asarray(v_mem_avail, dtype=np.float32)
    v_mem_used_frac = np.asarray(v_mem_used_frac, dtype=np.float32)
    v_active = np.asarray(v_active, dtype=np.float32)

    # Densities
    T = max(1, len(prev_obs.task_observations))
    V = max(1, len(prev_obs.vm_observations))
    pi = float(_ready_count(prev_obs)) / float(max(1, width))
    ei = _energy_intensity_index_prev(prev_obs)
    try:
        compat = getattr(prev_obs, "compatibilities", []) or []
        compat_density = float(len(compat)) / float(T * V)
    except Exception:
        compat_density = 0.0
    active_tasks_density = float(np.mean(v_active)) / float(max(1, V)) if v_active.size > 0 else 0.0

    # Build feature vector
    parts = [
        # Task stats
        _agg_stats(t_ready), _agg_stats(t_sched), _agg_stats(t_len), _agg_stats(t_mem), _agg_stats(t_cores),
        # VM stats
        _agg_stats(v_speed), _agg_stats(v_idle), _agg_stats(v_peak), _agg_stats(v_cpu), _agg_stats(v_cpu_avail),
        _agg_stats(v_cpu_used_frac), _agg_stats(v_mem), _agg_stats(v_mem_avail), _agg_stats(v_mem_used_frac), _agg_stats(v_active),
        # Densities
        np.array([pi, ei, compat_density, active_tasks_density], dtype=np.float32)
    ]
    return np.concatenate(parts, dtype=np.float32)


def _create_random_agent(device: torch.device, init_seed: int) -> GinAgent:
    """Create a randomly initialized agent with a specific seed."""
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    agent = GinAgent(device=device)
    agent.eval()
    return agent


def _rollout_collect_states(
    dataset,
    agent: GinAgent,
    max_steps: int = 10000,
    feature_mode: str = "summary"
) -> Tuple[List[np.ndarray], int]:
    """
    Rollout an agent on a dataset and collect state observations.
    
    Returns:
        states: List of state observation vectors (mapped observations)
        n_steps: Number of steps taken
    """
    env = GinAgentWrapper(
        CloudSchedulingGymEnvironment(
            dataset=dataset,
            compute_metrics=False,
            dataset_episode_mode="single"
        )
    )
    obs_np, _ = env.reset()

    states: List[np.ndarray] = []
    step_count = 0
    # Precompute width once for summary features
    width = _max_width_from_dataset(dataset)
    
    while step_count < max_steps:
        # Collect current state
        if feature_mode == "mapped":
            states.append(np.array(obs_np, dtype=np.float32))
        elif feature_mode == "features_only":
            prev = env.prev_obs
            states.append(_features_only_state(prev, width))
        else:
            prev = env.prev_obs
            states.append(_summarize_state(prev, width))
        
        # Agent action
        x = torch.from_numpy(obs_np.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            a, _, _, _ = agent.get_action_and_value(x)
        
        obs_np, _, terminated, truncated, _ = env.step(int(a.item()))
        step_count += 1
        
        if terminated or truncated:
            break
    
    return states, step_count


def _plot_pca_2d(
    X: np.ndarray,
    labels: np.ndarray,
    domain_labels: np.ndarray,
    out_path: str,
    title: str = "PCA: State Distribution"
):
    """Plot 2D PCA projection colored by agent and shaped by domain."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Unique agents and domains
    unique_agents = np.unique(labels)
    unique_domains = np.unique(domain_labels)
    
    # Color palette for agents
    colors = sns.color_palette("husl", len(unique_agents))
    agent_colors = {agent: colors[i] for i, agent in enumerate(unique_agents)}
    
    # Markers for domains
    domain_markers = {"wide": "o", "longcp": "s"}
    
    for agent in unique_agents:
        for domain in unique_domains:
            mask = (labels == agent) & (domain_labels == domain)
            if np.sum(mask) > 0:
                ax.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    c=[agent_colors[agent]],
                    marker=domain_markers.get(domain, "o"),
                    alpha=0.6,
                    s=20,
                    label=f"{agent} ({domain})"
                )
    
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA plot: {out_path}")


def _plot_tsne_2d(
    X: np.ndarray,
    labels: np.ndarray,
    domain_labels: np.ndarray,
    out_path: str,
    title: str = "t-SNE: State Distribution"
):
    """Plot 2D t-SNE projection colored by agent and shaped by domain."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Unique agents and domains
    unique_agents = np.unique(labels)
    unique_domains = np.unique(domain_labels)
    
    # Color palette for agents
    colors = sns.color_palette("husl", len(unique_agents))
    agent_colors = {agent: colors[i] for i, agent in enumerate(unique_agents)}
    
    # Markers for domains
    domain_markers = {"wide": "o", "longcp": "s"}
    
    for agent in unique_agents:
        for domain in unique_domains:
            mask = (labels == agent) & (domain_labels == domain)
            if np.sum(mask) > 0:
                ax.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    c=[agent_colors[agent]],
                    marker=domain_markers.get(domain, "o"),
                    alpha=0.6,
                    s=20,
                    label=f"{agent} ({domain})"
                )
    
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot: {out_path}")


def _plot_domain_comparison(
    X_pca: np.ndarray,
    domain_labels_pca: np.ndarray,
    X_tsne: np.ndarray,
    domain_labels_tsne: np.ndarray,
    out_path: str
):
    """Plot side-by-side PCA and t-SNE colored only by domain."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    unique_domains = np.unique(domain_labels_pca)
    colors = {"wide": "#1f77b4", "longcp": "#ff7f0e"}
    
    # PCA
    for domain in unique_domains:
        mask = domain_labels_pca == domain
        axes[0].scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=colors.get(domain, "gray"),
            alpha=0.5,
            s=15,
            label=domain
        )
    axes[0].set_xlabel("PC1", fontsize=12)
    axes[0].set_ylabel("PC2", fontsize=12)
    axes[0].set_title("PCA: Domain Comparison", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE
    # Use domains present in TSNE labels (may be subsampled)
    unique_domains_tsne = np.unique(domain_labels_tsne)
    for domain in unique_domains_tsne:
        mask = domain_labels_tsne == domain
        axes[1].scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            c=colors.get(domain, "gray"),
            alpha=0.5,
            s=15,
            label=domain
        )
    axes[1].set_xlabel("t-SNE 1", fontsize=12)
    axes[1].set_ylabel("t-SNE 2", fontsize=12)
    axes[1].set_title("t-SNE: Domain Comparison", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved domain comparison plot: {out_path}")


def main(args: Args):
    """Main execution."""
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device)
    
    # Load configs
    print(f"Loading configs...")
    seeds_long, ds_long = _load_seedset(args.longcp_config, "eval")
    seeds_wide, ds_wide = _load_seedset(args.wide_config, "eval")
    
    # Sample episodes
    n_ep = args.n_episodes_per_domain
    sampled_seeds_long = seeds_long[:n_ep] if len(seeds_long) >= n_ep else seeds_long
    sampled_seeds_wide = seeds_wide[:n_ep] if len(seeds_wide) >= n_ep else seeds_wide
    
    print(f"Sampled {len(sampled_seeds_long)} longcp episodes, {len(sampled_seeds_wide)} wide episodes")
    
    # Collect all states
    all_states: List[np.ndarray] = []
    all_labels: List[str] = []  # agent ID
    all_domains: List[str] = []  # "wide" or "longcp"
    
    print(f"\nRunning {args.n_agents} random agents on {len(sampled_seeds_long) + len(sampled_seeds_wide)} episodes...")
    
    for agent_idx in range(args.n_agents):
        agent_id = f"agent_{agent_idx}"
        print(f"\n=== {agent_id} ===")
        
        # Create random agent with unique seed
        agent = _create_random_agent(device, args.seed + agent_idx * 1000)
        
        # Run on longcp episodes
        for seed in tqdm(sampled_seeds_long, desc=f"{agent_id} longcp"):
            dataset = _mk_dataset(seed, ds_long)
            states, n_steps = _rollout_collect_states(dataset, agent, args.max_steps_per_episode, args.feature_mode)
            all_states.extend(states)
            all_labels.extend([agent_id] * len(states))
            all_domains.extend(["longcp"] * len(states))
        
        # Run on wide episodes
        for seed in tqdm(sampled_seeds_wide, desc=f"{agent_id} wide"):
            dataset = _mk_dataset(seed, ds_wide)
            states, n_steps = _rollout_collect_states(dataset, agent, args.max_steps_per_episode, args.feature_mode)
            all_states.extend(states)
            all_labels.extend([agent_id] * len(states))
            all_domains.extend(["wide"] * len(states))
    
    print(f"\nCollected {len(all_states)} total state observations")
    
    # Convert to numpy arrays
    X = np.array(all_states, dtype=np.float32)
    labels = np.array(all_labels)
    domains = np.array(all_domains)
    
    print(f"State shape: {X.shape}")
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    print(f"Running PCA (n_components={args.pca_components})...")
    pca = PCA(n_components=args.pca_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # t-SNE (subsample if too many points for speed)
    max_tsne_points = 10000
    if X_scaled.shape[0] > max_tsne_points:
        print(f"Subsampling {max_tsne_points} points for t-SNE...")
        idx = np.random.choice(X_scaled.shape[0], max_tsne_points, replace=False)
        X_tsne_input = X_scaled[idx]
        labels_tsne = labels[idx]
        domains_tsne = domains[idx]
    else:
        X_tsne_input = X_scaled
        labels_tsne = labels
        domains_tsne = domains
    
    print(f"Running t-SNE (n_components={args.tsne_components}, perplexity={args.tsne_perplexity})...")
    tsne = TSNE(
        n_components=args.tsne_components,
        perplexity=args.tsne_perplexity,
        random_state=args.seed,
        max_iter=1000
    )
    X_tsne = tsne.fit_transform(X_tsne_input)
    
    # Plot PCA (all agents + domains)
    print("\nGenerating plots...")
    _plot_pca_2d(
        X_pca,
        labels,
        domains,
        os.path.join(args.out_dir, "pca_all_agents.png"),
        title=f"PCA: State Distribution ({args.n_agents} Random Agents)"
    )
    
    # Plot t-SNE (all agents + domains)
    _plot_tsne_2d(
        X_tsne,
        labels_tsne,
        domains_tsne,
        os.path.join(args.out_dir, "tsne_all_agents.png"),
        title=f"t-SNE: State Distribution ({args.n_agents} Random Agents)"
    )
    
    # Plot domain comparison (PCA + t-SNE side by side)
    _plot_domain_comparison(
        X_pca,
        domains,
        X_tsne,
        domains_tsne,
        os.path.join(args.out_dir, "domain_comparison.png")
    )
    
    # Save summary stats
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Random Agent State Distribution Analysis\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Number of agents: {args.n_agents}\n")
        f.write(f"Episodes per domain: {args.n_episodes_per_domain}\n")
        f.write(f"Total states collected: {len(all_states)}\n")
        f.write(f"State dimensionality: {X.shape[1]}\n")
        f.write(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}\n")
        f.write(f"PCA cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}\n")
        f.write(f"\nStates per agent:\n")
        for agent_id in np.unique(labels):
            n = np.sum(labels == agent_id)
            f.write(f"  {agent_id}: {n}\n")
        f.write(f"\nStates per domain:\n")
        for domain in np.unique(domains):
            n = np.sum(domains == domain)
            f.write(f"  {domain}: {n}\n")
    
    print(f"\nSaved summary: {summary_path}")
    print(f"\nDone! Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
