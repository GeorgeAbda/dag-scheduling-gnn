#!/usr/bin/env python3
"""
Visualize the initial state (first observation) from training seeds across
all host configurations (AL, NAL, HP, HS) for LongCP and Wide domains using t-SNE.

This script:
1. Loads the 20 training seeds for LongCP and Wide domains
2. For each host configuration, generates the initial state for each seed
3. Extracts state features (mapped observation vector)
4. Projects all initial states to 2D using t-SNE
5. Creates visualizations colored by domain and/or host configuration

Usage:
    python scripts/visualize_initial_states_tsne.py \
        --out-dir release_new/initial_states_tsne \
        --device cpu
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
from sklearn.manifold import TSNE
from tqdm import tqdm
import tyro

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cogito.dataset_generator.core.gen_dataset import generate_dataset
from cogito.dataset_generator.core.gen_vm import generate_hosts, generate_vms, allocate_vms
from cogito.gnn_deeprl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from cogito.gnn_deeprl_model.agents.gin_agent.wrapper import GinAgentWrapper
from cogito.gnn_deeprl_model.agents.gin_agent.agent import GinAgent


@dataclass
class Args:
    """Arguments for initial state t-SNE visualization."""
    longcp_config: str = "release_new/data/rl_configs/train_long_cp_p08_seeds.json"
    """Path to longcp config JSON"""
    wide_config: str = "release_new/data/rl_configs/train_wide_p005_seeds.json"
    """Path to wide config JSON"""
    host_specs_al: str = "release_new/data/host_specs_AL.json"
    """Path to AL host specs"""
    host_specs_nal: str = "release_new/data/host_specs_NAL.json"
    """Path to NAL host specs"""
    host_specs_hp: str = "release_new/data/host_specs_homoPower.json"
    """Path to HP host specs"""
    host_specs_hs: str = "release_new/data/host_specs_homospeed.json"
    """Path to HS host specs"""
    out_dir: str = "release_new/initial_states_tsne"
    """Output directory for plots"""
    device: str = "cpu"
    """Device to run on (cpu/cuda)"""
    tsne_perplexity: float = 30.0
    """t-SNE perplexity parameter"""
    n_training_seeds: int = 20
    """Number of training seeds to use (first N from config)"""
    dpi: int = 300
    """Output DPI for PNG"""
    save_pdf: bool = True
    """Also save PDF versions"""


def _load_config(path: str) -> Tuple[List[int], dict]:
    """Load training seeds and dataset config from JSON."""
    with open(path, "r") as f:
        cfg = json.load(f)
    
    # Try both formats: all_seeds at top level or train.seeds
    if "all_seeds" in cfg:
        seeds = list(cfg["all_seeds"])
    elif "train" in cfg and "seeds" in cfg["train"]:
        seeds = list(cfg["train"]["seeds"])
    else:
        raise ValueError(f"Could not find seeds in config: {path}")
    
    # Get dataset config
    if "train" in cfg and "dataset" in cfg["train"]:
        dataset_cfg = dict(cfg["train"]["dataset"])
    elif "dataset" in cfg:
        dataset_cfg = dict(cfg["dataset"])
    else:
        raise ValueError(f"Could not find dataset config in: {path}")
    
    return seeds, dataset_cfg


def _compute_optimal_req_divisor(dataset_cfg: dict, seed: int) -> int:
    """Compute a conservative req_divisor for the dataset."""
    host_count = int(dataset_cfg.get("host_count", 10))
    vm_count = int(dataset_cfg.get("vm_count", 10))
    max_memory_gb = int(dataset_cfg.get("max_memory_gb", 10))
    min_cpu_speed = int(dataset_cfg.get("min_cpu_speed", 500))
    max_cpu_speed = int(dataset_cfg.get("max_cpu_speed", 5000))
    n_tasks = int(dataset_cfg.get("gnp_max_n", 40))

    if n_tasks <= 0:
        n_tasks = 1

    rng = np.random.RandomState(int(seed))
    hosts = generate_hosts(n=host_count, rng=rng)
    vms = generate_vms(
        n=vm_count,
        max_memory_gb=max_memory_gb,
        min_cpu_speed_mips=min_cpu_speed,
        max_cpu_speed_mips=max_cpu_speed,
        rng=rng,
    )
    allocate_vms(vms, hosts, rng)

    if not vms:
        return 1

    mem_caps = [int(getattr(vm, "memory_mb", 0)) for vm in vms]
    core_caps = [int(max(1, getattr(vm, "cpu_cores", 1))) for vm in vms]

    min_mem = max(1, min(mem_caps))
    max_mem = max(mem_caps)
    min_cores = max(1, min(core_caps))
    max_cores = max(core_caps)

    max_safe_mem_per_task = max(1024, min_mem // n_tasks)
    max_safe_cores_per_task = max(1, min_cores // n_tasks)

    req_div_mem = max(1, max_mem // max_safe_mem_per_task)
    req_div_core = max(1, max_cores // max_safe_cores_per_task)

    return int(max(req_div_mem, req_div_core))


def _generate_dataset_with_host_specs(seed: int, dataset_cfg: dict, host_specs_path: str):
    """Generate a dataset with specific host specs."""
    # Set environment variable for host specs
    os.environ["HOST_SPECS_PATH"] = host_specs_path
    
    req_div = _compute_optimal_req_divisor(dataset_cfg, seed)
    
    return generate_dataset(
        seed=seed,
        host_count=int(dataset_cfg.get("host_count", 10)),
        vm_count=int(dataset_cfg.get("vm_count", 10)),
        max_memory_gb=int(dataset_cfg.get("max_memory_gb", 128)),
        min_cpu_speed_mips=int(dataset_cfg.get("min_cpu_speed", 500)),
        max_cpu_speed_mips=int(dataset_cfg.get("max_cpu_speed", 5000)),
        workflow_count=int(dataset_cfg.get("workflow_count", 1)),
        dag_method=str(dataset_cfg.get("dag_method", "gnp")),
        gnp_min_n=int(dataset_cfg.get("gnp_min_n", 12)),
        gnp_max_n=int(dataset_cfg.get("gnp_max_n", 24)),
        task_length_dist=str(dataset_cfg.get("task_length_dist", "normal")),
        min_task_length=int(dataset_cfg.get("min_task_length", 500)),
        max_task_length=int(dataset_cfg.get("max_task_length", 100000)),
        task_arrival=str(dataset_cfg.get("task_arrival", "static")),
        arrival_rate=float(dataset_cfg.get("arrival_rate", 3)),
        vm_rng_seed=0,
        gnp_p=float(dataset_cfg.get("gnp_p")) if dataset_cfg.get("gnp_p") is not None else None,
        req_divisor=req_div,
    )


def collect_initial_states(args: Args):
    """Collect initial state observations from all configurations."""
    
    # Load configs
    longcp_seeds, longcp_cfg = _load_config(args.longcp_config)
    wide_seeds, wide_cfg = _load_config(args.wide_config)
    
    # Use only first N training seeds
    longcp_seeds = longcp_seeds[:args.n_training_seeds]
    wide_seeds = wide_seeds[:args.n_training_seeds]
    
    print(f"Using {len(longcp_seeds)} LongCP training seeds")
    print(f"Using {len(wide_seeds)} Wide training seeds")
    
    # Host configurations
    host_configs = {
        "AL": args.host_specs_al,
        "NAL": args.host_specs_nal,
        "HP": args.host_specs_hp,
        "HS": args.host_specs_hs,
    }
    
    # Storage for all initial states
    all_states = []
    all_labels = []
    
    # Create a dummy agent just for the mapper
    device = torch.device(args.device)
    dummy_agent = GinAgent(device)
    
    total_configs = len(host_configs) * (len(longcp_seeds) + len(wide_seeds))
    pbar = tqdm(total=total_configs, desc="Collecting initial states")
    
    # Collect states for each configuration
    for host_name, host_path in host_configs.items():
        # LongCP domain
        for seed in longcp_seeds:
            dataset = _generate_dataset_with_host_specs(seed, longcp_cfg, host_path)
            env = GinAgentWrapper(
                CloudSchedulingGymEnvironment(
                    dataset=dataset,
                    collect_timelines=False,
                    compute_metrics=False,
                    profile=False,
                    fixed_env_seed=True
                )
            )
            obs, _ = env.reset(seed=seed)
            all_states.append(obs)
            all_labels.append({
                "domain": "LongCP",
                "host": host_name,
                "seed": seed,
            })
            env.close()
            pbar.update(1)
        
        # Wide domain
        for seed in wide_seeds:
            dataset = _generate_dataset_with_host_specs(seed, wide_cfg, host_path)
            env = GinAgentWrapper(
                CloudSchedulingGymEnvironment(
                    dataset=dataset,
                    collect_timelines=False,
                    compute_metrics=False,
                    profile=False,
                    fixed_env_seed=True
                )
            )
            obs, _ = env.reset(seed=seed)
            all_states.append(obs)
            all_labels.append({
                "domain": "Wide",
                "host": host_name,
                "seed": seed,
            })
            env.close()
            pbar.update(1)
    
    pbar.close()
    
    # Convert to numpy array
    X = np.array(all_states, dtype=np.float32)
    print(f"\nCollected {X.shape[0]} initial states with {X.shape[1]} features each")
    
    return X, all_labels


def plot_tsne_by_domain(X_tsne, labels, out_dir, args):
    """Plot t-SNE colored by domain (LongCP vs Wide)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Separate by domain
    longcp_mask = np.array([l["domain"] == "LongCP" for l in labels])
    wide_mask = ~longcp_mask
    
    ax.scatter(
        X_tsne[longcp_mask, 0], X_tsne[longcp_mask, 1],
        c="#66BB6A", s=30, alpha=0.6, label="LongCP", edgecolors="none"
    )
    ax.scatter(
        X_tsne[wide_mask, 0], X_tsne[wide_mask, 1],
        c="#2E7D32", s=30, alpha=0.6, label="Wide", edgecolors="none"
    )
    
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("Initial States: LongCP vs Wide (all host configs)", fontsize=14)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    
    fig.tight_layout()
    out_path = out_dir / "initial_states_tsne_by_domain.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    if args.save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_tsne_by_host(X_tsne, labels, out_dir, args):
    """Plot t-SNE colored by host configuration."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    host_colors = {
        "AL": "#1976D2",
        "NAL": "#D32F2F",
        "HP": "#388E3C",
        "HS": "#F57C00",
    }
    
    for host_name, color in host_colors.items():
        mask = np.array([l["host"] == host_name for l in labels])
        ax.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=color, s=30, alpha=0.6, label=host_name, edgecolors="none"
        )
    
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("Initial States by Host Configuration", fontsize=14)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    
    fig.tight_layout()
    out_path = out_dir / "initial_states_tsne_by_host.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    if args.save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_tsne_grid(X_tsne, labels, out_dir, args):
    """Plot t-SNE in a 2x2 grid, one subplot per host configuration."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    host_names = ["AL", "NAL", "HP", "HS"]
    
    for idx, host_name in enumerate(host_names):
        ax = axes[idx]
        
        # Filter for this host
        host_mask = np.array([l["host"] == host_name for l in labels])
        X_host = X_tsne[host_mask]
        labels_host = [l for l in labels if l["host"] == host_name]
        
        # Separate by domain
        longcp_mask = np.array([l["domain"] == "LongCP" for l in labels_host])
        wide_mask = ~longcp_mask
        
        ax.scatter(
            X_host[longcp_mask, 0], X_host[longcp_mask, 1],
            c="#66BB6A", s=25, alpha=0.6, label="LongCP", edgecolors="none"
        )
        ax.scatter(
            X_host[wide_mask, 0], X_host[wide_mask, 1],
            c="#2E7D32", s=25, alpha=0.6, label="Wide", edgecolors="none"
        )
        
        ax.set_xlabel("t-SNE 1", fontsize=10)
        ax.set_ylabel("t-SNE 2", fontsize=10)
        ax.set_title(f"{host_name} Configuration", fontsize=12, fontweight="bold")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.15, linewidth=0.5)
    
    fig.suptitle("Initial States: LongCP vs Wide per Host Configuration", fontsize=14, y=0.995)
    fig.tight_layout()
    
    out_path = out_dir / "initial_states_tsne_grid.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    if args.save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    args = tyro.cli(Args)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Initial State t-SNE Visualization")
    print("=" * 60)
    
    # Collect initial states
    X, labels = collect_initial_states(args)
    
    # Run t-SNE
    print(f"\nRunning t-SNE (perplexity={args.tsne_perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        random_state=42,
        init="pca",
        n_iter=1000,
    )
    X_tsne = tsne.fit_transform(X)
    print("t-SNE complete")
    
    # Save embeddings
    np.savez(
        out_dir / "initial_states_tsne_embeddings.npz",
        X_tsne=X_tsne,
        labels=labels,
        X_original=X,
    )
    print(f"Saved embeddings to {out_dir / 'initial_states_tsne_embeddings.npz'}")
    
    # Create plots
    print("\nGenerating plots...")
    plot_tsne_by_domain(X_tsne, labels, out_dir, args)
    plot_tsne_by_host(X_tsne, labels, out_dir, args)
    plot_tsne_grid(X_tsne, labels, out_dir, args)
    
    print("\n" + "=" * 60)
    print("Done! All plots saved to:", out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
