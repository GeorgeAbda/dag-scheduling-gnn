#!/usr/bin/env python3
"""
Extract task→VM edge embeddings from trained agents and plot t-SNE projections with labels:
- Critical-path membership (by task)
- Indegree/Outdegree (task indegree as color, outdegree as marker size)
- Ready mask (by task)

Usage example:
  python scheduler/viz_results/plot_edge_embeddings_tsne.py \
    --model "NoGlobal:/path/to/no_global_actor_best_model.pt" \
    --variant noglobal \
    --gnp_p 0.8 \
    --hosts 4 --vms 10 --workflows 10 \
    --min_tasks 12 --max_tasks 24 \
    --device cpu \
    --episodes 3 \
    --out_dir csv_ablation/gnp_0.8/plots/embeddings

Repeat for MLP variant by changing --model and --variant.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Local imports from project
import os as _os, sys as _sys
_proj_root = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
if _proj_root not in _sys.path:
    _sys.path.insert(0, _proj_root)

from scheduler.rl_model.ablation_gnn import (
    AblationGinAgent,
    AblationVariant,
    Args as AblArgs,
    _make_test_env,
)
from scheduler.dataset_generator.gen_dataset import DatasetArgs


# ----------------------------
# Helpers to build eval args
# ----------------------------

def build_args_gnp(p: float, hosts: int, vms: int, workflows: int, min_tasks: int, max_tasks: int, device: str) -> AblArgs:
    a = AblArgs()
    a.device = device
    a.dataset = DatasetArgs(
        host_count=hosts,
        vm_count=vms,
        workflow_count=workflows,
        dag_method="gnp",
        gnp_min_n=min_tasks,
        gnp_max_n=max_tasks,
        gnp_p=p,
        max_memory_gb=10,
        min_cpu_speed=500,
        max_cpu_speed=5000,
        min_task_length=500,
        max_task_length=100_000,
        task_arrival="static",
    )
    # fewer iterations needed (we only extract at reset state)
    a.test_iterations = 1
    return a


def pick_variant(name: str) -> AblationVariant:
    n = name.strip().lower()
    if n in ("baseline",):
        return AblationVariant(name="baseline")
    if n in ("noglobal", "no_global_actor", "no_global"):
        return AblationVariant(name="no_global_actor", use_actor_global_embedding=False)
    if n in ("mlp_only", "mlp"):
        # Historical checkpoints trained with no global actor embedding
        return AblationVariant(name="mlp_only", mlp_only=True, gin_num_layers=0, use_actor_global_embedding=False)
    if n == "hetero":
        return AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True)
    # fallback
    return AblationVariant(name=n)


def load_agent(ckpt_path: str, device: torch.device, variant_name: str) -> AblationGinAgent:
    var = pick_variant(variant_name)
    # For our experiments: emb=16 for hetero; here variants are noglobal or mlp_only
    emb_dim = 16 if var.graph_type == "hetero" else 8
    agent = AblationGinAgent(device, var, embedding_dim=emb_dim)
    state = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(state, strict=False)
    agent.eval()
    return agent


# ----------------------------
# Graph/DAG label computations
# ----------------------------

def compute_task_degrees(dep_edges: torch.Tensor, num_tasks: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return indegree, outdegree arrays for task DAG (size num_tasks). dep_edges shape: (2, E)."""
    indeg = np.zeros(num_tasks, dtype=np.int32)
    outdeg = np.zeros(num_tasks, dtype=np.int32)
    if dep_edges is not None and dep_edges.numel() > 0:
        u = dep_edges[0].cpu().numpy().astype(int)
        v = dep_edges[1].cpu().numpy().astype(int)
        for a, b in zip(u, v):
            outdeg[a] += 1
            indeg[b] += 1
    return indeg, outdeg


def longest_path_nodes(dep_edges: torch.Tensor, num_tasks: int) -> np.ndarray:
    """Compute one set of nodes on a longest path in the DAG via DP on topological order."""
    # Build adjacency and indegrees for topo sort
    adj = [[] for _ in range(num_tasks)]
    indeg = np.zeros(num_tasks, dtype=np.int32)
    if dep_edges is not None and dep_edges.numel() > 0:
        u = dep_edges[0].cpu().numpy().astype(int)
        v = dep_edges[1].cpu().numpy().astype(int)
        for a, b in zip(u, v):
            adj[a].append(b)
            indeg[b] += 1
    # Kahn topo order
    from collections import deque
    dq = deque([i for i in range(num_tasks) if indeg[i] == 0])
    topo = []
    while dq:
        x = dq.popleft()
        topo.append(x)
        for y in adj[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                dq.append(y)
    # Longest path DP
    dist = np.full(num_tasks, -1, dtype=np.int32)
    parent = np.full(num_tasks, -1, dtype=np.int32)
    for i in topo:
        if dist[i] < 0:
            dist[i] = 0
        for j in adj[i]:
            if dist[i] + 1 > dist[j]:
                dist[j] = dist[i] + 1
                parent[j] = i
    # Find end node and recover path
    end = int(np.argmax(dist)) if dist.size > 0 else 0
    on_path = np.zeros(num_tasks, dtype=bool)
    cur = end
    while cur != -1 and cur >= 0:
        on_path[cur] = True
        cur = int(parent[cur]) if parent[cur] >= 0 else -1
    return on_path


# ----------------------------
# Extraction + Plotting
# ----------------------------

def extract_edge_embeddings_and_labels(agent: AblationGinAgent, args: AblArgs, rng_seed: int = 1):
    env = _make_test_env(args)
    obs_np, _ = env.reset(seed=rng_seed)
    # Map observation back to structured tensors
    obs_tensor = torch.tensor(np.asarray(obs_np, dtype=np.float32))
    obs = agent.mapper.unmap(obs_tensor)

    # Get raw edge embeddings from actor network (before optional global concat)
    with torch.no_grad():
        _, edge_embeddings, _ = agent.actor.network(obs)
    # Select only task→VM edges (first E entries)
    E = int(obs.compatibilities.shape[1])
    edge_emb_tensor = edge_embeddings[:E]
    edge_emb = edge_emb_tensor.cpu().numpy()

    # Split halves: [task_emb || vm_emb]
    emb_dim = int(agent.actor.embedding_dim)
    vm_half = edge_emb_tensor[:, -emb_dim:].cpu().numpy()

    # Labels per-edge derived from the source task id and vm id for that edge
    task_ids = obs.compatibilities[0][:E].cpu().numpy().astype(int)
    vm_ids = obs.compatibilities[1][:E].cpu().numpy().astype(int)

    num_tasks = int(obs.task_state_ready.shape[0])
    dep_edges = obs.task_dependencies  # shape (2, Edep)

    indeg, outdeg = compute_task_degrees(dep_edges, num_tasks)
    on_cp = longest_path_nodes(dep_edges, num_tasks)
    ready_mask = obs.task_state_ready.cpu().numpy().astype(bool)

    # Edge-level labels
    labels = {
        'cp': on_cp[task_ids],
        'indeg': indeg[task_ids],
        'outdeg': outdeg[task_ids],
        'ready': ready_mask[task_ids],
        'vm': vm_ids,
        'task': task_ids,
    }

    # Compute per-edge action scores from the actor (after masking), aligned with compat edges
    with torch.no_grad():
        action_scores = agent.actor(obs)  # shape [num_tasks, num_vms], invalid set to large negative
    edge_scores = action_scores[obs.compatibilities[0][:E], obs.compatibilities[1][:E]].cpu().numpy()

    env.close()
    return edge_emb, vm_half, labels, edge_scores


def tsne_project(X: np.ndarray, random_state: int = 0) -> np.ndarray:
    n = X.shape[0]
    perplex = max(5, min(30, n // 3))
    tsne = TSNE(n_components=2, perplexity=perplex, init='random', learning_rate='auto', random_state=random_state)
    Y = tsne.fit_transform(X)
    return Y


def plot_tsne(Y: np.ndarray, labels: dict, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Critical path membership
    plt.figure(figsize=(6,5))
    c = np.where(labels['cp'], 1, 0)
    plt.scatter(Y[:,0], Y[:,1], c=c, cmap='coolwarm', s=10, alpha=0.9)
    plt.title(f"{prefix} — t-SNE (color: critical-path)\nRed=True, Blue=False")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_tsne_cp.png", dpi=200)
    plt.close()

    # 2) Indegree/Outdegree: color = indegree, size = outdegree
    plt.figure(figsize=(6,5))
    indeg = labels['indeg'].astype(float)
    outdeg = labels['outdeg'].astype(float)
    sizes = 10 + 5.0 * (outdeg - outdeg.min()) / max(1e-6, (outdeg.max() - outdeg.min())) if outdeg.max()>outdeg.min() else np.full_like(outdeg, 12.0)
    sc = plt.scatter(Y[:,0], Y[:,1], c=indeg, cmap='viridis', s=sizes, alpha=0.9)
    plt.colorbar(sc, label='indegree (task)')
    plt.title(f"{prefix} — t-SNE (color: indegree, size: outdegree)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_tsne_deg.png", dpi=200)
    plt.close()

    # 3) Ready mask
    plt.figure(figsize=(6,5))
    c = np.where(labels['ready'], 1, 0)
    plt.scatter(Y[:,0], Y[:,1], c=c, cmap='coolwarm', s=10, alpha=0.9)
    plt.title(f"{prefix} — t-SNE (color: ready mask)\nRed=True, Blue=False")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_tsne_ready.png", dpi=200)
    plt.close()

    # 4) VM id coloring
    plt.figure(figsize=(6,5))
    vm = labels['vm'].astype(int)
    # Use a categorical colormap with enough distinct colors
    scatter = plt.scatter(Y[:,0], Y[:,1], c=vm, cmap='tab20', s=10, alpha=0.9)
    cbar = plt.colorbar(scatter, ticks=sorted(np.unique(vm)))
    cbar.set_label('VM id')
    plt.title(f"{prefix} — t-SNE (color: VM id)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_tsne_vm.png", dpi=200)
    plt.close()

    # 5) Task id coloring
    plt.figure(figsize=(6,5))
    task = labels['task'].astype(int)
    scatter = plt.scatter(Y[:,0], Y[:,1], c=task, cmap='tab20', s=10, alpha=0.9)
    # Show a compact colorbar with unique task ids sampled if too many
    unique_tasks = np.unique(task)
    if len(unique_tasks) <= 20:
        cbar = plt.colorbar(scatter, ticks=unique_tasks)
    else:
        cbar = plt.colorbar(scatter)
    cbar.set_label('Task id')
    plt.title(f"{prefix} — t-SNE (color: Task id)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_tsne_task.png", dpi=200)
    plt.close()


def plot_tsne_topk(Y: np.ndarray, scores: np.ndarray, out_dir: Path, prefix: str, k: int = 3):
    """Overlay the top-k highest-scoring edges on the t-SNE projection.
    Args:
        Y: (N,2) t-SNE coordinates aligned with the edge embeddings used for projection.
        scores: (N,) per-edge action scores aligned with Y.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,5))
    # Base scatter in light gray
    plt.scatter(Y[:,0], Y[:,1], c='#BBBBBB', s=8, alpha=0.5, edgecolors='none')
    # Top-k indices by score
    order = np.argsort(scores)[::-1]
    topk = order[:min(k, scores.shape[0])]
    # Use distinct markers/colors
    markers = ['*', 'X', 'P']
    colors = ['red', 'orange', 'gold']
    sizes = [120, 90, 75]
    for r, idx in enumerate(topk):
        m = markers[r % len(markers)]
        c = colors[r % len(colors)]
        s = sizes[r % len(sizes)]
        plt.scatter([Y[idx,0]], [Y[idx,1]], marker=m, c=c, s=s, edgecolors='k', linewidths=0.6, label=f"top{r+1}")
    plt.legend(loc='best', fontsize=8, frameon=True)
    plt.title(f"{prefix} — t-SNE with top-{min(k, scores.shape[0])} actions")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_tsne_top{min(k, scores.shape[0])}.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Plot t-SNE of task→VM edge embeddings with labels.')
    ap.add_argument('--model', type=str, required=True, help='Label:/path/to/ckpt.pt')
    ap.add_argument('--variant', type=str, required=True, choices=['baseline','noglobal','no_global_actor','no_global','mlp_only','mlp','hetero'])
    ap.add_argument('--gnp_p', type=float, default=0.8)
    ap.add_argument('--hosts', type=int, default=4)
    ap.add_argument('--vms', type=int, default=10)
    ap.add_argument('--workflows', type=int, default=10)
    ap.add_argument('--min_tasks', type=int, default=12)
    ap.add_argument('--max_tasks', type=int, default=24)
    ap.add_argument('--episodes', type=int, default=1, help='Number of episodes to sample (we use the reset state)')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--vm_half_only', action='store_true', help='Use only the VM half of edge embeddings for t-SNE and clustering metrics')
    ap.add_argument('--out_dir', type=Path, default=Path('csv_ablation/gnp_0.8/plots/embeddings'))
    args = ap.parse_args()

    if ':' not in args.model:
        raise SystemExit('Use --model Label:/full/path/to/ckpt.pt')
    label, ckpt = args.model.split(':', 1)

    device = torch.device(args.device)
    eval_args = build_args_gnp(args.gnp_p, args.hosts, args.vms, args.workflows, args.min_tasks, args.max_tasks, args.device)

    agent = load_agent(ckpt, device, args.variant)

    # Collect embeddings across episodes (concatenate)
    X_list = []
    lab_collect = []
    vm_half_list = []
    scores_list = []
    for s in range(args.episodes):
        X_full, X_vm, lab, sc = extract_edge_embeddings_and_labels(agent, eval_args, rng_seed=1_000_000_001 + s)
        X_list.append(X_full)
        vm_half_list.append(X_vm)
        lab_collect.append(lab)
        scores_list.append(sc)
    X_full = np.concatenate(X_list, axis=0)
    X_vm = np.concatenate(vm_half_list, axis=0)
    scores_all = np.concatenate(scores_list, axis=0)
    X = X_vm if args.vm_half_only else X_full
    # Concatenate labels correspondingly
    labels = {
        'cp': np.concatenate([d['cp'] for d in lab_collect], axis=0),
        'indeg': np.concatenate([d['indeg'] for d in lab_collect], axis=0),
        'outdeg': np.concatenate([d['outdeg'] for d in lab_collect], axis=0),
        'ready': np.concatenate([d['ready'] for d in lab_collect], axis=0),
        'vm': np.concatenate([d['vm'] for d in lab_collect], axis=0),
        'task': np.concatenate([d['task'] for d in lab_collect], axis=0),
    }

    # Project and plot
    Y = tsne_project(X, random_state=0)
    prefix = f"{label}_{args.variant}_p{args.gnp_p:.2f}"
    plot_tsne(Y, labels, args.out_dir, prefix)
    # Overlay top-3 highest scoring edges in the same embedding space (based on full-edge scores)
    try:
        plot_tsne_topk(Y, scores_all, args.out_dir, prefix, k=3)
    except Exception as e:
        print(f"[topk] Warning: failed to plot top-3: {e}")

    # Clustering metric: silhouette score for VM labels
    try:
        sil = silhouette_score(X, labels['vm'])
        print(f"[silhouette] VM clustering (X={'vm_half' if args.vm_half_only else 'edge_full'}) for {label} ({args.variant}): {sil:.4f}")
    except Exception as e:
        print(f"[silhouette] could not compute: {e}")

    print(f"Saved t-SNE plots for {label} ({args.variant}) at: {args.out_dir}")


if __name__ == '__main__':
    main()
