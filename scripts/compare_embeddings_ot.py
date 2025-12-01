#!/usr/bin/env python3
"""
Compare GNN vs MLP embeddings using Optimal Transport (OT) between train-like and evaluation domains.

- Loads ablation-trained checkpoints (model.pt or *_iterXXXX.pt) for two variants: a GNN-based model and an MLP-only model
- Runs short evaluations to collect embeddings from AblationBaseNetwork.forward()
- Computes:
  * Sinkhorn (Wasserstein) distance between train-like (GNP p) and OOD (e.g., linear) node embeddings
  * Gromov–Wasserstein (GW) discrepancy based on intra-domain distances
  * Barycentric projection displacements of OOD embeddings into the train manifold
- Writes a CSV summary and optional plots

Example:
  python scripts/compare_embeddings_ot.py \
    --gnn_ckpt logs/1758400896_gnn_ablation_gnp_p03_baseline/ablation/per_variant/baseline_model.pt \
    --mlp_ckpt logs/1758400896_gnn_ablation_gnp_p03_baseline/ablation/per_variant/mlp_only_model.pt \
    --train_base gnp --train_gnp_p 0.3 \
    --eval_base linear \
    --episodes 8 \
    --device cpu \
    --out_dir logs/ot_compare

Notes:
- Requires POT (Python Optimal Transport): pip install POT
- For large graphs, this script sub-samples a fixed number of node embeddings per episode to keep GW tractable
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch

# Optional deps
try:
    import ot
except Exception as _e:
    ot = None

try:
    from sklearn.metrics.pairwise import pairwise_distances
except Exception:
    pairwise_distances = None

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

try:
    from sklearn.manifold import TSNE
except Exception:
    TSNE = None

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Local imports mirroring other scripts
import sys as _sys
_grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _grandparent_dir not in _sys.path:
    _sys.path.insert(0, _grandparent_dir)

from scheduler.rl_model.ablation_gnn import (
    AblationGinAgent,
    AblationVariant,
    Args as AblArgs,
    _make_test_env,
)
from scheduler.dataset_generator.gen_dataset import DatasetArgs


def _pick_variant(name: str) -> AblationVariant:
    name = name.strip().lower()
    if name in ("baseline",):
        return AblationVariant(name="baseline")
    if name in ("mlp_only", "mlp"):
        return AblationVariant(name="mlp_only", mlp_only=True, gin_num_layers=0)
    if name in ("no_global_actor", "noglobal"):
        return AblationVariant(name="no_global_actor", use_actor_global_embedding=False)
    # Fallback to baseline
    return AblationVariant(name=name)


def _load_agent(ckpt_path: Path, device: torch.device, variant_name: str) -> AblationGinAgent:
    var = _pick_variant(variant_name)
    sd = torch.load(str(ckpt_path), map_location=device)

    # Try initial construction
    agent = AblationGinAgent(device, var)
    try:
        agent.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        # Heuristic: infer whether use_actor_global_embedding should be False from edge_scorer weight shape
        try:
            w = sd.get('actor.edge_scorer.0.weight', None)
            if isinstance(w, torch.Tensor):
                in_dim_ckpt = w.shape[1]
                # embedding_dim default is 8 in AblationGinAgent; in_dim formula: (2 + (1 if use_actor_global_embedding else 0)) * embedding_dim
                # If in_dim is 16, likely use_actor_global_embedding=False with embedding_dim=8
                # If in_dim is 24, likely use_actor_global_embedding=True with embedding_dim=8
                if in_dim_ckpt % 8 == 0:
                    emb_guess = 8
                    k = in_dim_ckpt // emb_guess
                    use_global = (k >= 3)
                    # rebuild with inferred flag if it differs
                    if use_global != var.use_actor_global_embedding:
                        var2 = _pick_variant(variant_name)
                        var2.use_actor_global_embedding = use_global
                        agent = AblationGinAgent(device, var2)
                        agent.load_state_dict(sd, strict=False)
                    else:
                        raise e
                else:
                    raise e
            else:
                raise e
        except Exception:
            raise e
    agent.eval()
    return agent


def _build_args(base: str, gnp_p: float | None, device: str) -> AblArgs:
    a = AblArgs()
    a.device = device
    # Configure dataset for domain
    if base == 'gnp':
        a.dataset = DatasetArgs(
            host_count=4,
            vm_count=10,
            workflow_count=10,
            dag_method="gnp",
            gnp_min_n=12,
            gnp_max_n=24,
            gnp_p=gnp_p,
            max_memory_gb=10,
            min_cpu_speed=500,
            max_cpu_speed=5000,
            min_task_length=500,
            max_task_length=100_000,
            task_arrival="static",
        )
    elif base == 'linear':
        a.dataset = DatasetArgs(
            host_count=4,
            vm_count=10,
            workflow_count=10,
            dag_method="linear",
            gnp_min_n=12,
            gnp_max_n=24,
            gnp_p=None,
            max_memory_gb=10,
            min_cpu_speed=500,
            max_cpu_speed=5000,
            min_task_length=500,
            max_task_length=100_000,
            task_arrival="static",
        )
    else:
        raise ValueError(f"Unknown base: {base}")
    return a


def _parse_eval_specs(specs: list[str]) -> list[tuple[str, float | None, str]]:
    """
    Parse eval specs like:
      - 'linear' -> (base='linear', p=None, tag='linear')
      - 'gnp:p=0.3' -> (base='gnp', p=0.3, tag='gnp_p0.3')
    Returns list of (base, p, tag) tuples used to build dataset args and subdirs.
    """
    out: list[tuple[str, float | None, str]] = []
    for s in specs:
        s = s.strip()
        if not s:
            continue
        if s.startswith('gnp'):
            p = None
            tag = 'gnp'
            # allow formats 'gnp', 'gnp:p=0.3'
            if ':' in s:
                _, rest = s.split(':', 1)
                for part in rest.split(','):
                    part = part.strip()
                    if part.startswith('p='):
                        try:
                            p = float(part[2:])
                        except Exception:
                            p = None
            if p is not None:
                tag = f'gnp_p{p}'
            out.append(('gnp', p, tag))
        elif s == 'linear':
            out.append(('linear', None, 'linear'))
        else:
            # fallback: treat as base string directly
            out.append((s, None, s))
    return out


def _parse_size_specs(specs: list[str]) -> list[tuple[int | None, int | None, int | None, str]]:
    """
    Parse size specs like:
      - 'h=6,v=12,t=18' (hosts, vms, tasks per workflow)
      - 'h=6' or 'v=12' or 't=18' (partial overrides)
    Returns list of (hosts, vms, tasks, tag) where tag is like 'h6_v12_t18' (omits parts that are None).
    """
    out: list[tuple[int | None, int | None, int | None, str]] = []
    for s in specs:
        s = s.strip()
        if not s:
            continue
        h = v = t = None
        for part in s.split(','):
            part = part.strip()
            if not part:
                continue
            if part.startswith('h='):
                try:
                    h = int(part[2:])
                except Exception:
                    h = None
            elif part.startswith('v='):
                try:
                    v = int(part[2:])
                except Exception:
                    v = None
            elif part.startswith('t='):
                try:
                    t = int(part[2:])
                except Exception:
                    t = None
        tag_parts = []
        if h is not None: tag_parts.append(f'h{h}')
        if v is not None: tag_parts.append(f'v{v}')
        if t is not None: tag_parts.append(f't{t}')
        tag = '_'.join(tag_parts) if tag_parts else 'size_default'
        out.append((h, v, t, tag))
    return out


def _collect_node_embeddings(
    agent: AblationGinAgent,
    args: AblArgs,
    episodes: int,
    max_nodes_per_ep: int = 256,
    frames_per_ep: int = 1,
) -> np.ndarray:
    """Run episodes and collect a sub-sample of node embeddings per episode.
    Returns array of shape (K, d) where K is total sampled nodes and d is embedding_dim.
    """
    device = agent.device
    rng = np.random.default_rng(1234)
    all_nodes: List[np.ndarray] = []

    for ep in range(episodes):
        env = _make_test_env(args)
        obs, _ = env.reset()
        with torch.no_grad():
            frame = 0
            while True:
                # Unmap obs and get node embeddings from the base network
                x = torch.tensor(obs, dtype=torch.float32, device=device)
                # agent.mapper -> GinAgentMapper
                go = agent.mapper.unmap(x)  # GinAgentObsTensor
                node_Z, _, _gZ = agent.actor.network(go)
                node_Z_np = node_Z.detach().cpu().numpy()
                # Sub-sample nodes to bound complexity
                if node_Z_np.shape[0] > max_nodes_per_ep:
                    idx = rng.choice(node_Z_np.shape[0], size=max_nodes_per_ep, replace=False)
                    node_Z_np = node_Z_np[idx]
                all_nodes.append(node_Z_np)

                # Take a step using the agent policy
                action, _, _, _ = agent.get_action_and_value(x.unsqueeze(0))
                nxt, _, term, trunc, info = env.step(action.cpu().numpy())
                obs = nxt
                frame += 1
                if frame >= frames_per_ep:
                    # Only collect limited frames per episode to reduce runtime/memory
                    break
                if bool(term) or bool(trunc):
                    break
        env.close()

    if not all_nodes:
        return np.zeros((0, agent.actor.network.embedding_dim), dtype=np.float32)
    return np.concatenate(all_nodes, axis=0)


def _normalize(X: np.ndarray, kind: str = "l2") -> np.ndarray:
    if X.size == 0:
        return X
    if kind == "l2":
        n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        return X / n
    if kind == "z":
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        return (X - mu) / sd
    return X


def _extract_metrics_from_info(info: dict) -> tuple[float | None, float | None, dict]:
    """Best-effort extraction of (energy_delta, makespan_at_step) from an env info dict.
    Returns (energy_increment, makespan_value_or_None) for this step. Makespan may only be known at terminal.
    """
    energy = None
    makespan = None
    extras: dict = {}
    # Common patterns
    for k in ('energy_delta', 'delta_energy', 'energy'):  # step delta preferred
        if isinstance(info, dict) and k in info and isinstance(info[k], (int, float)):
            energy = float(info[k])
            break
    # makespan sometimes in 'episode' or 'metrics'
    for container_key in ('episode', 'metrics', 'summary'):
        d = info.get(container_key) if isinstance(info, dict) else None
        if isinstance(d, dict):
            for mk in ('makespan', 'makespan_time', 'total_time'):
                if mk in d and isinstance(d[mk], (int, float)):
                    makespan = float(d[mk])
                    break
    # direct keys
    if makespan is None and isinstance(info, dict):
        for mk in ('makespan', 'makespan_time', 'total_time'):
            if mk in info and isinstance(info[mk], (int, float)):
                makespan = float(info[mk])
                break
    # Collect totals if available
    for k in ('total_energy', 'total_energy_active', 'total_energy_idle'):
        if isinstance(info, dict) and k in info and isinstance(info[k], (int, float)):
            extras[k] = float(info[k])
    return energy, makespan, extras


def _eval_policy_metrics(agent: AblationGinAgent, args: AblArgs, episodes: int) -> tuple[list[dict], dict]:
    """Run episodes and compute per-episode and summary metrics for energy and makespan.
    We accumulate energy deltas if present; on termination, if the info contains totals, they override the sum.
    """
    rng = np.random.default_rng(123)
    device = agent.device
    per_ep: list[dict] = []
    for ep in range(episodes):
        env = _make_test_env(args)
        obs, _ = env.reset()
        energy_sum = 0.0
        final_makespan: float | None = None
        with torch.no_grad():
            while True:
                x = torch.tensor(obs, dtype=torch.float32, device=device)
                action, _, _, _ = agent.get_action_and_value(x.unsqueeze(0))
                nxt, reward, term, trunc, info = env.step(action.cpu().numpy())
                e_delta, m_val, extras = _extract_metrics_from_info(info if isinstance(info, dict) else {})
                if e_delta is not None:
                    energy_sum += float(e_delta)
                if m_val is not None:
                    final_makespan = float(m_val)
                obs = nxt
                if bool(term) or bool(trunc):
                    # try to read totals at end
                    if isinstance(info, dict):
                        for k in ('total_energy', 'energy_total'):
                            if k in info and isinstance(info[k], (int, float)):
                                energy_sum = float(info[k])
                        for container_key in ('episode', 'metrics', 'summary'):
                            d = info.get(container_key)
                            if isinstance(d, dict):
                                for mk in ('total_energy', 'energy_total'):
                                    if mk in d and isinstance(d[mk], (int, float)):
                                        energy_sum = float(d[mk])
                                for mk in ('makespan', 'makespan_time', 'total_time'):
                                    if mk in d and isinstance(d[mk], (int, float)):
                                        final_makespan = float(d[mk])
                        # fallback: reward is negative makespan by env contract
                        if final_makespan is None and isinstance(reward, (int, float)):
                            final_makespan = float(-reward)
                    break
        env.close()
        te = float(extras.get('total_energy', np.nan))
        tea = float(extras.get('total_energy_active', np.nan))
        tei = float(extras.get('total_energy_idle', np.nan))
        teh = float(tea + tei) if (not np.isnan(tea) and not np.isnan(tei)) else np.nan
        per_ep.append({'episode': ep,
                       'energy': float(energy_sum),
                       'makespan': (float(final_makespan) if final_makespan is not None else np.nan),
                       'total_energy': te,
                       'total_energy_active': tea,
                       'total_energy_idle': tei,
                       'total_energy_host': teh})
    # summary
    energies = np.array([r['energy'] for r in per_ep], dtype=float)
    makespans = np.array([r['makespan'] for r in per_ep], dtype=float)
    total_energy_active = np.array([r.get('total_energy_active', np.nan) for r in per_ep], dtype=float)
    total_energy_idle = np.array([r.get('total_energy_idle', np.nan) for r in per_ep], dtype=float)
    total_energy_host = np.array([r.get('total_energy_host', np.nan) for r in per_ep], dtype=float)
    summary = {
        'episodes': episodes,
        'energy_mean': float(np.nanmean(energies)) if energies.size else np.nan,
        'energy_std': float(np.nanstd(energies)) if energies.size else np.nan,
        'makespan_mean': float(np.nanmean(makespans)) if makespans.size else np.nan,
        'makespan_std': float(np.nanstd(makespans)) if makespans.size else np.nan,
        'total_energy_active_mean': float(np.nanmean(total_energy_active)) if total_energy_active.size else np.nan,
        'total_energy_active_std': float(np.nanstd(total_energy_active)) if total_energy_active.size else np.nan,
        'total_energy_idle_mean': float(np.nanmean(total_energy_idle)) if total_energy_idle.size else np.nan,
        'total_energy_idle_std': float(np.nanstd(total_energy_idle)) if total_energy_idle.size else np.nan,
        'total_energy_host_mean': float(np.nanmean(total_energy_host)) if total_energy_host.size else np.nan,
        'total_energy_host_std': float(np.nanstd(total_energy_host)) if total_energy_host.size else np.nan,
    }
    return per_ep, summary


def _sinkhorn_distance(X: np.ndarray, Y: np.ndarray, reg: float = 1e-2, numItermax: int = 5000) -> float:
    if ot is None:
        raise RuntimeError("POT not installed. Please `pip install POT`. ")
    n, m = X.shape[0], Y.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m
    C = ot.utils.dist(X, Y, metric='euclidean')  # (n, m)
    C /= (C.max() + 1e-12)
    W = ot.sinkhorn(a, b, C, reg, numItermax=numItermax)
    return float((W * C).sum())


def _gw_distance(X: np.ndarray, Y: np.ndarray, metric: str = 'cosine', epsilon: float = 5e-3) -> Tuple[float, np.ndarray]:
    if ot is None or pairwise_distances is None:
        raise RuntimeError("POT and scikit-learn are required. `pip install POT scikit-learn`. ")
    n, m = X.shape[0], Y.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m
    DX = pairwise_distances(X, X, metric=metric)
    DY = pairwise_distances(Y, Y, metric=metric)
    gw_dist, log = ot.gromov.gromov_wasserstein2(DX, DY, a, b, loss_fun='square_loss', epsilon=epsilon, log=True)
    Pi = log['T']  # (n, m)
    return float(gw_dist), Pi


def _barycentric_displacements(X: np.ndarray, Y: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    """Project OOD (Y) into train space (X) and compute displacement to top-match in X."""
    col = Pi.sum(axis=0, keepdims=True) + 1e-12
    X_bary_from_Y = (X.T @ (Pi / col)).T  # (m, d)
    top_i = Pi.argmax(axis=0)
    nn_X = X[top_i]
    disp = np.linalg.norm(X_bary_from_Y - nn_X, axis=1)
    return disp


def _plot_transport_arrows(
    X: np.ndarray,
    Y: np.ndarray,
    Pi: np.ndarray,
    out_path: Path,
    title: str = "",
    max_arrows: int = 500,
    project: str = "pca",
    tsne_perplexity: float = 30.0,
    tsne_n_iter: int = 1000,
):
    """Project to 2D with PCA, draw Y points and arrows to their barycentric projections in X.
    Saves a PNG at out_path.
    """
    if PCA is None or plt is None:
        print("[ot] Skipping plot: scikit-learn or matplotlib not available.")
        return
    m = Y.shape[0]
    # Compute barycentric projections
    col = Pi.sum(axis=0, keepdims=True) + 1e-12
    X_bary_from_Y = (X.T @ (Pi / col)).T  # (m, d)

    # Reduce to 2D jointly for stable geometry
    Z = np.vstack([X, Y, X_bary_from_Y])
    if project == "tsne":
        if TSNE is None:
            print("[ot] TSNE not available; falling back to PCA")
            projector = PCA(n_components=2, random_state=0)
            Z2 = projector.fit_transform(Z)
        else:
            if tsne_n_iter != 1000:
                print("[ot] Note: tsne_n_iter flag is ignored for compatibility with your sklearn TSNE version")
            Z2 = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=0, init="pca").fit_transform(Z)
    else:
        projector = PCA(n_components=2, random_state=0)
        Z2 = projector.fit_transform(Z)
    n = X.shape[0]
    X2 = Z2[:n]
    Y2 = Z2[n:n + m]
    Xb2 = Z2[n + m:]

    # Sample arrows for clarity
    rng = np.random.default_rng(42)
    if m > max_arrows:
        idx = rng.choice(m, size=max_arrows, replace=False)
        Y2s = Y2[idx]
        Xb2s = Xb2[idx]
    else:
        idx = np.arange(m)
        Y2s = Y2
        Xb2s = Xb2

    plt.figure(figsize=(7, 6), dpi=140)
    # Plot train cloud lightly
    plt.scatter(X2[:, 0], X2[:, 1], s=6, c="#1f77b4", alpha=0.25, label="Train X")
    # Plot eval points
    plt.scatter(Y2[:, 0], Y2[:, 1], s=10, c="#d62728", alpha=0.6, label="Eval Y")
    # Arrows from Y to barycentric projection in X
    for j in range(Y2s.shape[0]):
        plt.arrow(
            Y2s[j, 0], Y2s[j, 1],
            (Xb2s[j, 0] - Y2s[j, 0]), (Xb2s[j, 1] - Y2s[j, 1]),
            length_includes_head=True,
            head_width=0.02 * (np.ptp(Z2[:, 0]) + np.ptp(Z2[:, 1])),
            head_length=0.02 * (np.ptp(Z2[:, 0]) + np.ptp(Z2[:, 1])),
            color="#2ca02c",
            alpha=0.4,
            linewidth=0.5,
        )
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_cluster_symbols(
    X: np.ndarray,
    Y: np.ndarray,
    Pi: np.ndarray,
    out_path: Path,
    title: str = "",
    k: int = 20,
    top_pairs: int = 3,
    mass_min: float = 0.0,
    embed_points: bool = False,
    embed_points_max: int = 4000,
    project: str = "pca",
    tsne_perplexity: float = 30.0,
    tsne_n_iter: int = 1000,
):
    """PCA 2D visualization of cluster centroids with symbol/opacity encoding transport mass.
    - KMeans on X and Y (k clusters each)
    - For each Y cluster j, draw markers at its centroid with shapes corresponding to the top source X clusters i by mass M[i,j].
      Opacity encodes relative mass; filter by mass_min and top_pairs.
    """
    if plt is None or KMeans is None or PCA is None:
        print("[ot] Skipping cluster symbol plot: matplotlib or scikit-learn not available.")
        return
    kmX = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmY = KMeans(n_clusters=k, n_init=10, random_state=0)
    labX = kmX.fit_predict(X)
    labY = kmY.fit_predict(Y)
    CX = kmX.cluster_centers_
    CY = kmY.cluster_centers_

    # Aggregate coupling to cluster level
    M = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        idx_i = np.where(labX == i)[0]
        if idx_i.size == 0:
            continue
        Pi_i = Pi[idx_i]
        for j in range(k):
            idx_j = np.where(labY == j)[0]
            if idx_j.size == 0:
                continue
            M[i, j] = Pi_i[:, idx_j].sum()
    if M.sum() > 0:
        M /= M.sum()

    # Joint PCA of embeddings (sampled) + centroids for stable geometry
    rng = np.random.default_rng(123)
    Xs = X
    Ys = Y
    if embed_points and X.shape[0] > embed_points_max:
        X_idx = rng.choice(X.shape[0], size=embed_points_max, replace=False)
        Xs = X[X_idx]
    if embed_points and Y.shape[0] > embed_points_max:
        Y_idx = rng.choice(Y.shape[0], size=embed_points_max, replace=False)
        Ys = Y[Y_idx]
    Z = np.vstack([Xs, Ys, CX, CY])
    if project == "tsne":
        if TSNE is None:
            print("[ot] TSNE not available; falling back to PCA")
            projector = PCA(n_components=2, random_state=0)
            Z2 = projector.fit_transform(Z)
        else:
            if tsne_n_iter != 1000:
                print("[ot] Note: tsne_n_iter flag is ignored for compatibility with your sklearn TSNE version")
            Z2 = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=0, init="pca").fit_transform(Z)
    else:
        projector = PCA(n_components=2, random_state=0)
        Z2 = projector.fit_transform(Z)
    nXs = Xs.shape[0]
    nYs = Ys.shape[0]
    X2 = Z2[:nXs]
    Y2 = Z2[nXs:nXs + nYs]
    CZ2 = Z2[nXs + nYs:]
    CX2 = CZ2[:k]
    CY2 = CZ2[k:]

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', 'H', 'p', 'x', '+']
    def mkr(i: int) -> str:
        return markers[i % len(markers)]

    plt.figure(figsize=(7, 6), dpi=140)
    # Plot sampled raw embeddings faintly for context
    if embed_points:
        plt.scatter(X2[:, 0], X2[:, 1], c="#1f77b4", s=6, alpha=0.15, label="X embeddings")
        plt.scatter(Y2[:, 0], Y2[:, 1], c="#d62728", s=6, alpha=0.15, label="Y embeddings")
    # Plot X centroids (train)
    for i in range(k):
        plt.scatter(CX2[i, 0], CX2[i, 1], marker=mkr(i), c="#1f77b4", s=70, alpha=0.8, label="X centroid" if i == 0 else None)
    # Plot Y centroids with opacity per incoming mass from top source clusters
    for j in range(k):
        masses = M[:, j]
        if masses.sum() <= 0:
            continue
        order = np.argsort(masses)[::-1]
        drawn = 0
        for i in order:
            mass = float(masses[i])
            if mass < mass_min:
                continue
            alpha = float(np.clip(mass / (masses.max() + 1e-12), 0.05, 1.0))
            plt.scatter(CY2[j, 0], CY2[j, 1], marker=mkr(i), c="#d62728", s=90, alpha=alpha,
                        label="Y centroid (opacity~mass)" if (j == 0 and drawn == 0) else None)
            drawn += 1
            if drawn >= top_pairs:
                break
    plt.title(title)
    plt.legend(frameon=False, loc='best')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_cluster_transport(
    X: np.ndarray,
    Y: np.ndarray,
    Pi: np.ndarray,
    out_path: Path,
    title: str = "",
    k: int = 20,
):
    """Cluster X and Y separately with KMeans(k) and aggregate Pi into a kxk transport matrix.
    Saves a heatmap of cluster-to-cluster transport masses.
    """
    if plt is None or KMeans is None:
        print("[ot] Skipping cluster heatmap: matplotlib or scikit-learn not available.")
        return
    # Fit KMeans on each domain separately (on normalized embeddings)
    kmX = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmY = KMeans(n_clusters=k, n_init=10, random_state=0)
    labX = kmX.fit_predict(X)
    labY = kmY.fit_predict(Y)
    # Aggregate Pi
    M = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        idx_i = np.where(labX == i)[0]
        if idx_i.size == 0:
            continue
        Pi_i = Pi[idx_i]  # (|i|, m)
        # For each j cluster in Y
        for j in range(k):
            idx_j = np.where(labY == j)[0]
            if idx_j.size == 0:
                continue
            M[i, j] = Pi_i[:, idx_j].sum()
    # Normalize to probabilities
    if M.sum() > 0:
        M /= M.sum()
    plt.figure(figsize=(7, 6), dpi=140)
    im = plt.imshow(M, cmap='viridis', aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04, label='transport mass')
    plt.xlabel('Y clusters')
    plt.ylabel('X clusters')
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_displacement_hist(
    X: np.ndarray,
    Y: np.ndarray,
    Pi: np.ndarray,
    out_path: Path,
    title: str = "",
):
    """Plot histogram + CDF of ||Y - X_bary(Y)|| under Pi."""
    if plt is None:
        print("[ot] Skipping displacement plot: matplotlib not available.")
        return
    col = Pi.sum(axis=0, keepdims=True) + 1e-12
    XbY = (X.T @ (Pi / col)).T  # (m, d)
    d = np.linalg.norm(Y - XbY, axis=1)
    d_sorted = np.sort(d)
    cdf = np.arange(1, len(d_sorted)+1) / len(d_sorted)
    plt.figure(figsize=(7, 5), dpi=140)
    plt.subplot(1, 2, 1)
    plt.hist(d, bins=30, color="#2ca02c", alpha=0.8)
    plt.xlabel("||Y - X_bary(Y)||")
    plt.ylabel("count")
    plt.title("Displacement histogram")
    plt.subplot(1, 2, 2)
    plt.plot(d_sorted, cdf, color="#1f77b4")
    plt.xlabel("||Y - X_bary(Y)||")
    plt.ylabel("CDF")
    plt.title("Displacement CDF")
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gnn_ckpt', type=str, required=True, help='Path to GNN variant checkpoint (e.g., baseline_model.pt)')
    p.add_argument('--mlp_ckpt', type=str, required=True, help='Path to MLP-only checkpoint (mlp_only_model.pt or iter)')
    p.add_argument('--gnn_variant', type=str, default='baseline', help='Variant name for GNN (default: baseline)')
    p.add_argument('--mlp_variant', type=str, default='mlp_only', help='Variant name for MLP (default: mlp_only)')
    p.add_argument('--train_base', type=str, default='gnp', choices=['gnp','linear'])
    p.add_argument('--train_gnp_p', type=float, default=0.3)
    p.add_argument('--eval_base', type=str, default='linear', choices=['gnp','linear'])
    p.add_argument('--eval_gnp_p', type=float, default=None)
    p.add_argument('--eval_specs', nargs='+', default=None, help="Optional list of eval specs, e.g. linear gnp:p=0.2 gnp:p=0.4. If set, overrides --eval_base/--eval_gnp_p and evaluates each spec in a subfolder.")
    p.add_argument('--size_specs', nargs='+', default=None, help="Optional list of size specs, e.g. h=6,v=12,t=18 h=6 t=30. Evaluates each size override per eval spec.")
    p.add_argument('--episodes', type=int, default=6)
    p.add_argument('--max_nodes_per_ep', type=int, default=256)
    p.add_argument('--frames_per_ep', type=int, default=1)
    p.add_argument('--max_total_nodes', type=int, default=20000, help='Cap total nodes kept per domain after concatenation')
    p.add_argument('--sinkhorn_reg', type=float, default=1e-2)
    p.add_argument('--sinkhorn_iter', type=int, default=5000)
    p.add_argument('--gw_epsilon', type=float, default=5e-3)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--out_dir', type=str, default='logs/ot_compare')
    p.add_argument('--make_plots', action='store_true', help='If set, saves OT arrow plots for GNN and MLP')
    p.add_argument('--max_arrows', type=int, default=400, help='Max arrows to draw in the OT plot')
    p.add_argument('--plot_displacements', action='store_true', help='If set, also saves displacement hist/CDF plots')
    p.add_argument('--cluster_heatmaps', action='store_true', help='If set, saves K-means cluster-level transport heatmaps')
    p.add_argument('--kmeans_k', type=int, default=20, help='K for KMeans in cluster heatmaps')
    p.add_argument('--cluster_symbols', action='store_true', help='If set, draws centroid symbols with opacity ~ transport mass')
    p.add_argument('--top_pairs', type=int, default=3, help='Top-X source clusters per target centroid to draw')
    p.add_argument('--mass_min', type=float, default=0.0, help='Min mass to visualize a pair')
    p.add_argument('--embed_points', action='store_true', help='If set, also scatter sampled raw embeddings behind centroids')
    p.add_argument('--embed_points_max', type=int, default=4000, help='Max sampled embeddings per domain to display')
    p.add_argument('--viz_project', type=str, default='tsne', choices=['pca','tsne'], help='2D projection method for plots')
    p.add_argument('--tsne_perplexity', type=float, default=30.0)
    p.add_argument('--tsne_n_iter', type=int, default=1000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Load agents
    gnn = _load_agent(Path(args.gnn_ckpt), device, args.gnn_variant)
    mlp = _load_agent(Path(args.mlp_ckpt), device, args.mlp_variant)

    # Build train/eval args
    train_args = _build_args(args.train_base, args.train_gnp_p if args.train_base=='gnp' else None, args.device)
    # Build eval specs
    if args.eval_specs:
        eval_specs = _parse_eval_specs(args.eval_specs)
    else:
        # single fallback
        eval_specs = [(args.eval_base, (args.eval_gnp_p if args.eval_base=='gnp' else None), args.eval_base if args.eval_base!='gnp' or args.eval_gnp_p is None else f"gnp_p{args.eval_gnp_p}")]

    # Build size specs
    if args.size_specs:
        size_specs = _parse_size_specs(args.size_specs)
    else:
        size_specs = [(None, None, None, 'size_default')]

    # Collect embeddings
    print('[ot] Collecting train embeddings (GNN) ...')
    X_gnn = _collect_node_embeddings(gnn, train_args, args.episodes, args.max_nodes_per_ep, args.frames_per_ep)
    print('[ot] Collecting train embeddings (MLP) ...')
    X_mlp = _collect_node_embeddings(mlp, train_args, args.episodes, args.max_nodes_per_ep, args.frames_per_ep)
    
    # Subsample globally if needed to cap memory (for train only here)
    rng = np.random.default_rng(0)
    def _cap(A: np.ndarray) -> np.ndarray:
        if A.shape[0] > args.max_total_nodes:
            idx = rng.choice(A.shape[0], size=args.max_total_nodes, replace=False)
            return A[idx]
        return A
    X_gnn = _cap(X_gnn); X_mlp = _cap(X_mlp)

    # Normalize
    X_gnn = _normalize(X_gnn, 'l2')
    X_mlp = _normalize(X_mlp, 'l2')

    # Compute distances
    rows = []
    def _compute_block(tag: str, X: np.ndarray, Y: np.ndarray):
        if X.size == 0 or Y.size == 0:
            return {
                'tag': tag,
                'sinkhorn': np.nan,
                'gw': np.nan,
                'disp_mean': np.nan,
                'disp_median': np.nan,
                'n_train': X.shape[0],
                'n_eval': Y.shape[0],
            }
        try:
            sink = _sinkhorn_distance(X, Y, reg=args.sinkhorn_reg, numItermax=args.sinkhorn_iter)
        except Exception as e:
            sink = float('nan')
            print(f"[ot] Sinkhorn failed for {tag}: {e}")
        try:
            gw, Pi = _gw_distance(X, Y, metric='cosine', epsilon=args.gw_epsilon)
            disp = _barycentric_displacements(X, Y, Pi)
            disp_mean = float(np.mean(disp))
            disp_median = float(np.median(disp))
        except Exception as e:
            gw, disp_mean, disp_median = float('nan'), float('nan'), float('nan')
            print(f"[ot] GW failed for {tag}: {e}")
        return {
            'tag': tag,
            'sinkhorn': sink,
            'gw': gw,
            'disp_mean': disp_mean,
            'disp_median': disp_median,
            'n_train': X.shape[0],
            'n_eval': Y.shape[0],
        }
    # Loop over evaluation specs
    for base, pval, tag in eval_specs:
        for (h, v, t, size_tag) in size_specs:
            eval_args = _build_args(base, pval if base=='gnp' else None, args.device)
            # apply size overrides
            if h is not None:
                eval_args.dataset.host_count = h
            if v is not None:
                eval_args.dataset.vm_count = v
            if t is not None:
                eval_args.dataset.gnp_min_n = t
                eval_args.dataset.gnp_max_n = t

            combined_tag = f"{tag}_{size_tag}" if size_tag != 'size_default' else tag
            spec_dir = out_dir / combined_tag
            spec_dir.mkdir(parents=True, exist_ok=True)

            print(f"[ot] Collecting eval embeddings (GNN) for {combined_tag} ...")
            Y_gnn = _collect_node_embeddings(gnn, eval_args, args.episodes, args.max_nodes_per_ep, args.frames_per_ep)
            print(f"[ot] Collecting eval embeddings (MLP) for {combined_tag} ...")
            Y_mlp = _collect_node_embeddings(mlp, eval_args, args.episodes, args.max_nodes_per_ep, args.frames_per_ep)

            # Cap and normalize for eval sets
            Y_gnn = _cap(Y_gnn); Y_mlp = _cap(Y_mlp)
            Y_gnn = _normalize(Y_gnn, 'l2'); Y_mlp = _normalize(Y_mlp, 'l2')

            # Metrics & plots
            gnn_metrics = _compute_block('GNN', X_gnn, Y_gnn)
            gnn_metrics['spec'] = combined_tag
            rows.append(gnn_metrics)
            if args.make_plots and X_gnn.size and Y_gnn.size:
                try:
                    gw_gnn, Pi_gnn = _gw_distance(X_gnn, Y_gnn, metric='cosine', epsilon=args.gw_epsilon)
                    _plot_transport_arrows(
                        X_gnn, Y_gnn, Pi_gnn,
                        spec_dir / 'ot_plan_gnn.png',
                        title=f"GNN OT plan {combined_tag} (gw={gw_gnn:.3g})",
                        max_arrows=args.max_arrows,
                        project=args.viz_project,
                        tsne_perplexity=args.tsne_perplexity,
                        tsne_n_iter=args.tsne_n_iter,
                    )
                    if args.plot_displacements:
                        _plot_displacement_hist(
                            X_gnn, Y_gnn, Pi_gnn,
                            spec_dir / 'ot_disp_hist_gnn.png',
                            title=f"GNN displacement {combined_tag} (||Y - X_bary(Y)||)",
                        )
                    if args.cluster_heatmaps:
                        _plot_cluster_transport(
                            X_gnn, Y_gnn, Pi_gnn,
                            spec_dir / 'ot_cluster_heatmap_gnn.png',
                            title=f"GNN cluster transport {combined_tag} (k={args.kmeans_k})",
                            k=args.kmeans_k,
                        )
                    if args.cluster_symbols:
                        _plot_cluster_symbols(
                            X_gnn, Y_gnn, Pi_gnn,
                            spec_dir / 'ot_cluster_symbols_gnn.png',
                            title=f"GNN centroid symbols {combined_tag} (opacity~mass, k={args.kmeans_k})",
                            k=args.kmeans_k,
                            top_pairs=args.top_pairs,
                            mass_min=args.mass_min,
                            embed_points=args.embed_points,
                            embed_points_max=args.embed_points_max,
                            project=args.viz_project,
                            tsne_perplexity=args.tsne_perplexity,
                            tsne_n_iter=args.tsne_n_iter,
                        )
                except Exception as e:
                    print(f"[ot] Plot (GNN,{combined_tag}) skipped: {e}")

            # Evaluate policy-level metrics (energy, makespan) and write CSVs
            per_gnn, sum_gnn = _eval_policy_metrics(gnn, eval_args, args.episodes)
            per_mlp, sum_mlp = _eval_policy_metrics(mlp, eval_args, args.episodes)
            # Per-episode CSVs
            import csv as _csv
            with (spec_dir / 'metrics_gnn.csv').open('w', newline='') as f:
                w = _csv.DictWriter(f, fieldnames=['episode','energy','makespan','total_energy','total_energy_active','total_energy_idle','total_energy_host'])
                w.writeheader(); [w.writerow(r) for r in per_gnn]
            with (spec_dir / 'metrics_mlp.csv').open('w', newline='') as f:
                w = _csv.DictWriter(f, fieldnames=['episode','energy','makespan','total_energy','total_energy_active','total_energy_idle','total_energy_host'])
                w.writeheader(); [w.writerow(r) for r in per_mlp]
            # Summary CSV comparing variants
            with (spec_dir / 'metrics_summary.csv').open('w', newline='') as f:
                w = _csv.DictWriter(f, fieldnames=[
                    'variant','episodes',
                    'energy_mean','energy_std',
                    'makespan_mean','makespan_std',
                    'total_energy_active_mean','total_energy_active_std',
                    'total_energy_idle_mean','total_energy_idle_std',
                    'total_energy_host_mean','total_energy_host_std',
                ])
                w.writeheader()
                w.writerow({'variant':'GNN', **sum_gnn})
                w.writerow({'variant':'MLP', **sum_mlp})

            mlp_metrics = _compute_block('MLP', X_mlp, Y_mlp)
            mlp_metrics['spec'] = combined_tag
            rows.append(mlp_metrics)
            if args.make_plots and X_mlp.size and Y_mlp.size:
                try:
                    gw_mlp, Pi_mlp = _gw_distance(X_mlp, Y_mlp, metric='cosine', epsilon=args.gw_epsilon)
                    _plot_transport_arrows(
                        X_mlp, Y_mlp, Pi_mlp,
                        spec_dir / 'ot_plan_mlp.png',
                        title=f"MLP OT plan {combined_tag} (gw={gw_mlp:.3g})",
                        max_arrows=args.max_arrows,
                        project=args.viz_project,
                        tsne_perplexity=args.tsne_perplexity,
                        tsne_n_iter=args.tsne_n_iter,
                    )
                    if args.plot_displacements:
                        _plot_displacement_hist(
                            X_mlp, Y_mlp, Pi_mlp,
                            spec_dir / 'ot_disp_hist_mlp.png',
                            title=f"MLP displacement {combined_tag} (||Y - X_bary(Y)||)",
                        )
                    if args.cluster_heatmaps:
                        _plot_cluster_transport(
                            X_mlp, Y_mlp, Pi_mlp,
                            spec_dir / 'ot_cluster_heatmap_mlp.png',
                            title=f"MLP cluster transport {combined_tag} (k={args.kmeans_k})",
                            k=args.kmeans_k,
                        )
                    if args.cluster_symbols:
                        _plot_cluster_symbols(
                            X_mlp, Y_mlp, Pi_mlp,
                            spec_dir / 'ot_cluster_symbols_mlp.png',
                            title=f"MLP centroid symbols {combined_tag} (opacity~mass, k={args.kmeans_k})",
                            k=args.kmeans_k,
                            top_pairs=args.top_pairs,
                            mass_min=args.mass_min,
                            embed_points=args.embed_points,
                            embed_points_max=args.embed_points_max,
                            project=args.viz_project,
                            tsne_perplexity=args.tsne_perplexity,
                            tsne_n_iter=args.tsne_n_iter,
                        )
                except Exception as e:
                    print(f"[ot] Plot (MLP,{combined_tag}) skipped: {e}")

    # Write CSV
    import csv
    csv_path = out_dir / 'ot_compare_summary.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['spec','tag','sinkhorn','gw','disp_mean','disp_median','n_train','n_eval'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ot] Wrote summary: {csv_path}")

    # Simple printout for quick compare
    for r in rows:
        print(f"[ot] {r['tag']}: sinkhorn={r['sinkhorn']:.6g}, gw={r['gw']:.6g}, disp_mean={r['disp_mean']:.6g}, disp_median={r['disp_median']:.6g}, n=({r['n_train']},{r['n_eval']})")


if __name__ == '__main__':
    main()
