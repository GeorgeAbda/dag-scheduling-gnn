#!/usr/bin/env python3
"""
Compare actor MLP decision regions across three ablation variants by visualizing
unique ReLU activation patterns ("linear regions") over the 2D PCA subspace of
scorer inputs (edge embeddings [|| graph embedding]).

- Models: Homogeneous (no-global), Heterogeneous (with global embedding), MLP-only
- Dataset: GNP with configurable p, tasks/VMs, etc. (matches embedding panels)
- State: single reset state controlled by --seed (same for all models)
- Output: 3x1 panel with activation-region tilings and probability-colored points

Usage example:

python -m scheduler.viz_results.decision_boundaries.compare_activation_regions \
  --homogeneous-model logs/.../no_global_actor_best_model.pt \
  --heterogeneous-model logs/.../hetero_best_model.pt \
  --mlp-model logs/.../mlp_only_best_model.pt \
  --gnp-p 0.8 --hosts 4 --vms 10 --workflows 10 --min-tasks 12 --max-tasks 24 \
  --seed 1000000001 --device cpu --out-dir scheduler/viz_results/decision_boundaries
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.path import Path as MplPath

# Project path hack (this file is three levels deep from project root)
import os as _os, sys as _sys
_proj_root = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))
if _proj_root not in _sys.path:
    _sys.path.insert(0, _proj_root)

from cogito.gnn_deeprl_model.ablation_gnn import (
    AblationGinAgent,
    AblationVariant,
    Args as AblArgs,
    _make_test_env,
)
from cogito.dataset_generator.gen_dataset import DatasetArgs


# ----------------------------
# Helpers (args, loading, style)
# ----------------------------

def register_spectral_11_cmap():
    """Register the custom 'Spectral-11' colormap locally if not present."""
    try:
        plt.colormaps.get_cmap('Spectral-11')
        return
    except Exception:
        pass
    spectral_11_rgba = [
        (158/255, 1/255, 66/255, 1),
        (213/255, 62/255, 79/255, 1),
        (244/255, 109/255, 67/255, 1),
        (253/255, 174/255, 97/255, 1),
        (254/255, 224/255, 139/255, 1),
        (255/255, 255/255, 191/255, 1),
        (230/255, 245/255, 152/255, 1),
        (171/255, 221/255, 164/255, 1),
        (102/255, 194/255, 165/255, 1),
        (50/255, 136/255, 189/255, 1),
        (94/255, 79/255, 162/255, 1),
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list('Spectral-11', spectral_11_rgba)
    try:
        plt.colormaps.register(cmap)
    except Exception:
        # If registration fails (older MPL), fall back to using it by object in get_cmap calls
        pass

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
    a.test_iterations = 1
    return a


def pick_variant(name: str) -> AblationVariant:
    """Create variant configuration by name."""
    n = name.strip().lower()    
    if n in ("noglobal", "no_global_actor", "no_global"):
        return AblationVariant(name="no_global_actor", use_actor_global_embedding=False)
    elif n == "homo":
        return AblationVariant(name="homogeneous")
    elif n in ("mlp_only", "mlp"):
        return AblationVariant(name="mlp_only", mlp_only=True, gin_num_layers=0, use_actor_global_embedding=False)
    elif n == "hetero":
        return AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True)
    elif n == "hetero_noglobal":
        return AblationVariant(name="hetero", graph_type="hetero", use_task_dependencies=True, use_actor_global_embedding=False)
    else:
        return AblationVariant(name="baseline")



def load_agent(ckpt_path: str, device: torch.device, variant_name: str) -> AblationGinAgent:
    var = pick_variant(variant_name)
    emb_dim =8
    agent = AblationGinAgent(device, var, embedding_dim=emb_dim)
    state = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(state, strict=False)
    agent.eval()
    return agent


def setup_publication_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['patch.linewidth'] = 0.5
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['legend.frameon'] = False
    register_spectral_11_cmap()


# ----------------------------
# Core: scorer inputs, probabilities, activation patterns
# ----------------------------

def extract_scorer_inputs_and_probs(agent: AblationGinAgent, args: AblArgs, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_in, Y2, P) for a single state.
    X_in: scorer inputs (E, D) = [edge_emb || graph_emb?]
    Y2: PCA projection of X_in to 2D (E,2)
    P: per-edge action probability under the agent for this state (E,)
    """
    env = _make_test_env(args)
    try:
        obs_np, _ = env.reset(seed=seed)
        obs_tensor = torch.tensor(np.asarray(obs_np, dtype=np.float32))
        obs = agent.mapper.unmap(obs_tensor)
        with torch.no_grad():
            _, edge_embeddings, graph_embedding = agent.actor.network(obs)
            E = int(obs.compatibilities.shape[1])
            edge_embeddings = edge_embeddings[:E]
            scorer_in = edge_embeddings
            if getattr(agent.variant, 'use_actor_global_embedding', False):
                rep_graph = graph_embedding.expand(edge_embeddings.shape[0], agent.actor.embedding_dim)
                scorer_in = torch.cat([scorer_in, rep_graph], dim=1)
            # Edge scores (compat edges only)
            scores = agent.actor.edge_scorer(scorer_in).flatten()
            # Mask for readiness/not-scheduled
            t_idx = obs.compatibilities[0][:E].to(torch.long)
            valid_task_mask = (obs.task_state_ready[t_idx] == 1) & (obs.task_state_scheduled[t_idx] == 0)
            masked_scores = scores.clone(); masked_scores[~valid_task_mask] = -1e8
            probs = torch.softmax(masked_scores, dim=0).cpu().numpy()
        X = scorer_in.cpu().numpy()
        # PCA to 2D (no sklearn dependency)
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        comp = Vt[:2].T
        Y = (Xc @ comp).astype(np.float64)
        return X, Y, probs
    finally:
        try:
            env.close()
        except Exception:
            pass


def activation_signatures(edge_scorer: torch.nn.Sequential, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ReLU activation masks for the two hidden layers of the edge_scorer.
    Returns (mask1, mask2, logits), where mask1 has shape (N, H1), mask2 (N, H2).
    """
    edge_scorer.eval()
    with torch.no_grad():
        h = X
        pre1 = edge_scorer[0](h)
        bn1 = edge_scorer[1](pre1)
        mask1 = (bn1 > 0)
        h1 = torch.relu(bn1)
        pre2 = edge_scorer[3](h1)
        bn2 = edge_scorer[4](pre2)
        mask2 = (bn2 > 0)
        h2 = torch.relu(bn2)
        logits = edge_scorer[6](h2).flatten()
    return mask1.cpu().numpy().astype(bool), mask2.cpu().numpy().astype(bool), logits.cpu().numpy()


def pca_from_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp = Vt[:2].T
    Y = Xc @ comp
    return Y.astype(np.float64), mu.squeeze().astype(np.float64), comp.astype(np.float64)


def grid_activation_regions(edge_scorer: torch.nn.Sequential, mu: np.ndarray, comp: np.ndarray,
                            bounds: Tuple[np.ndarray, np.ndarray], grid_res: int = 300):
    """Tile the 2D PCA plane with activation region ids by probing the scorer on the plane.
    bounds: (min_xy, max_xy) in the 2D PCA space.
    Returns (XX, YY, Z_id) where Z_id is int region id per cell.
    """
    xmin, ymin = bounds[0]
    xmax, ymax = bounds[1]
    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    G = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (g,2)
    # Map back to scorer input space: x = mu + G @ comp.T
    Xg = (G @ comp.T) + mu
    Xg_t = torch.tensor(Xg, dtype=torch.float32)
    m1, m2, _ = activation_signatures(edge_scorer, Xg_t)
    # Hash activation bits into compact ints
    def hash_bits(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
        total = b1.shape[1] + b2.shape[1]
        out = np.zeros((b1.shape[0],), dtype=np.uint64)
        pos = 0
        for j in range(b1.shape[1]):
            out |= (b1[:, j].astype(np.uint64) << np.uint64(pos)); pos += 1
        for j in range(b2.shape[1]):
            out |= (b2[:, j].astype(np.uint64) << np.uint64(pos)); pos += 1
        return out
    zid = hash_bits(m1, m2)
    Z = zid.reshape(grid_res, grid_res)
    return XX, YY, Z


# ----------------------------
# Geometry and fragmentation within convex hull
# ----------------------------

def _convex_hull_monotone_chain(P: np.ndarray) -> np.ndarray:
    """Compute 2D convex hull (counter-clockwise) using Andrew's monotone chain.
    P: (N,2) array
    Returns (H,2) hull vertices.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] <= 3:
        return P.copy()
    pts = P[np.lexsort((P[:, 1], P[:, 0]))]
    def cross(o, a, b):
        return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull


def _grid_mask_inside_hull(XX: np.ndarray, YY: np.ndarray, hull_xy: np.ndarray) -> np.ndarray:
    """Return boolean mask of grid cell centers that lie within the convex hull polygon."""
    path = MplPath(hull_xy)
    G = np.stack([XX.ravel(), YY.ravel()], axis=1)
    inside = path.contains_points(G)
    return inside.reshape(XX.shape)


def fragmentation_metrics_within_hull(Y: np.ndarray, XX: np.ndarray, YY: np.ndarray, Z: np.ndarray):
    """Compute fragmentation metrics restricted to the convex hull of observed points Y.
    Returns dict with: region_count_in_hull, boundary_count, adjacency_pairs, fragmentation_ratio, hull_xy
    """
    hull_xy = _convex_hull_monotone_chain(Y)
    mask = _grid_mask_inside_hull(XX, YY, hull_xy)
    # Horizontal adjacencies
    diff_h = (Z[:, 1:] != Z[:, :-1])
    valid_h = mask[:, 1:] & mask[:, :-1]
    bcount_h = int(np.sum(diff_h & valid_h))
    total_h = int(np.sum(valid_h))
    # Vertical adjacencies
    diff_v = (Z[1:, :] != Z[:-1, :])
    valid_v = mask[1:, :] & mask[:-1, :]
    bcount_v = int(np.sum(diff_v & valid_v))
    total_v = int(np.sum(valid_v))
    boundary_count = bcount_h + bcount_v
    adjacency_pairs = total_h + total_v
    fragmentation_ratio = float(boundary_count) / float(max(1, adjacency_pairs))
    region_count_in_hull = int(np.unique(Z[mask]).size)
    return {
        'region_count_in_hull': region_count_in_hull,
        'boundary_count': boundary_count,
        'adjacency_pairs': adjacency_pairs,
        'fragmentation_ratio': fragmentation_ratio,
        'hull_xy': hull_xy,
    }


# ----------------------------
# Plotting
# ----------------------------

def plot_activation_regions_panel(results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]],
                                  out_path: Path,
                                  prob_cmap: str = 'Spectral-11',
                                  grid_res: int = 300,
                                  show_hull: bool = True):
    setup_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    model_names = ['Homogeneous GNN', 'Heterogeneous GNN', 'MLP-only']
    model_keys = ['homogeneous', 'heterogeneous', 'mlp']
    panel_labels = ['(a)', '(b)', '(c)']

    # Normalize probabilities across panels for consistent color scaling
    vmax = 0.0
    for key in model_keys:
        if key in results:
            _X, Y, P, _tile, _hull = results[key]
            if P.size > 0:
                vmax = max(vmax, float(np.max(P)))
    norm = mcolors.Normalize(vmin=0.0, vmax=max(1e-9, vmax))
    sm = plt.cm.ScalarMappable(cmap=plt.colormaps.get_cmap(prob_cmap), norm=norm)
    sm.set_array([])

    for i, (ax, name, key) in enumerate(zip(axes, model_names, model_keys)):
        if key not in results:
            ax.text(0.5, 0.5, f'{name}\nNot Available', ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        X, Y, P, tile, hull_xy = results[key]
        XX, YY, Z = tile
        # Remap region ids to compact 0..K-1 for colormap stability
        uniq = np.unique(Z)
        remap = {u: idx for idx, u in enumerate(uniq)}
        Zc = np.vectorize(remap.get)(Z)
        K = len(uniq)
        region_cmap = plt.colormaps.get_cmap('tab20')
        ax.pcolormesh(XX, YY, Zc, cmap=region_cmap, shading='nearest', alpha=0.35)
        # Overlay convex hull of observed points (optional)
        if show_hull:
            try:
                ax.plot(np.r_[hull_xy[:,0], hull_xy[0,0]], np.r_[hull_xy[:,1], hull_xy[0,1]],
                        color='black', linewidth=1.0, alpha=0.6)
            except Exception:
                pass
        # Overlay samples colored by probability
        colors = plt.colormaps.get_cmap(prob_cmap)(norm(P))
        ax.scatter(Y[:, 0], Y[:, 1], c=colors, s=9, alpha=0.85, edgecolors='none')
        ax.text(0.01, 0.98, panel_labels[i], transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top', ha='left')
        ax.set_xticks([]); ax.set_yticks([])

    # Shared horizontal colorbar for probabilities
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.18, fraction=0.05, aspect=30)
    cbar.set_label('Action probability (per edge)')
    plt.tight_layout(rect=[0, 0.16, 1, 0.97])
    # Save PDF
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    # Save PNG alongside
    png_path = out_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved activation-region comparison plots: {out_path} and {png_path}")


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Compare actor decision regions via ReLU activation patterns across models')
    parser.add_argument('--homogeneous-model', type=str, required=True)
    parser.add_argument('--heterogeneous-model', type=str, required=True)
    parser.add_argument('--mlp-model', type=str, required=True)
    parser.add_argument('--gnp-p', type=float, default=0.8)
    parser.add_argument('--gnp-ps', type=str, default='', help='Optional comma-separated list of gnp p values to sweep, e.g., "0.3,0.5,0.8"')
    parser.add_argument('--hosts', type=int, default=4)
    parser.add_argument('--vms', type=int, default=10)
    parser.add_argument('--workflows', type=int, default=10)
    parser.add_argument('--min-tasks', type=int, default=12)
    parser.add_argument('--max-tasks', type=int, default=24)
    parser.add_argument('--seed', type=int, default=1_000_000_001)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out-dir', type=Path, default=Path('scheduler/viz_results/decision_boundaries'))
    parser.add_argument('--grid-res', type=int, default=300)
    parser.add_argument('--prob-cmap', type=str, default='Spectral-11')
    parser.add_argument('--no-hull', action='store_true',
                        help='If set, do not draw convex hull lines around observed points in each panel')
    args = parser.parse_args()

    device = torch.device(args.device)
    ps = []
    if args.gnp_ps.strip():
        try:
            ps = [float(x) for x in args.gnp_ps.split(',') if x.strip()]
        except Exception:
            print(f"Warning: failed to parse --gnp-ps='{args.gnp_ps}', falling back to --gnp-p={args.gnp_p}")
            ps = [args.gnp_p]
    else:
        ps = [args.gnp_p]

    model_specs = [
        ('homogeneous', args.homogeneous_model, 'homogeneous'),
        ('heterogeneous', args.heterogeneous_model, 'hetero'),
        ('mlp', args.mlp_model, 'mlp_only'),
    ]

    for p in ps:
        print(f"\n=== GNP p = {p:.3f} ===")
        eval_args = build_args_gnp(p, args.hosts, args.vms, args.workflows, args.min_tasks, args.max_tasks, args.device)
        results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {}
        for key, path, variant in model_specs:
            if not (path and Path(path).exists()):
                print(f"Skipping {key} (path not found: {path})")
                continue
            print(f"Processing {key} model for activation regions...")
            try:
                agent = load_agent(path, device, variant)
                X_in, Y2, P = extract_scorer_inputs_and_probs(agent, eval_args, args.seed)
                # PCA fit from data to build plane for tiling
                Y, mu, comp = pca_from_data(X_in)
                # Tile plane with activation regions
                pad = 0.05
                mins = Y.min(axis=0); maxs = Y.max(axis=0)
                bounds = (mins - pad * (maxs - mins + 1e-9), maxs + pad * (maxs - mins + 1e-9))
                XX, YY, Z = grid_activation_regions(agent.actor.edge_scorer, mu, comp, bounds, grid_res=args.grid_res)
                # Fragmentation metrics inside convex hull of observed Y
                fr = fragmentation_metrics_within_hull(Y, XX, YY, Z)
                results[key] = (X_in, Y, P, (XX, YY, Z), fr['hull_xy'])
                # Summary metrics
                m1, m2, _ = activation_signatures(agent.actor.edge_scorer, torch.tensor(X_in, dtype=torch.float32))
                pat_ids = np.concatenate([m1, m2], axis=1)
                uniq_patterns = np.unique(pat_ids, axis=0).shape[0]
                uniq_regions = np.unique(Z).size
                print(f"  Unique activation patterns (observed edges): {uniq_patterns}; Tiled regions (global): {uniq_regions}")
                print(f"  In-hull regions: {fr['region_count_in_hull']}; boundary_count={fr['boundary_count']}, adjacency_pairs={fr['adjacency_pairs']}, fragmentation_ratio={fr['fragmentation_ratio']:.4f}")
            except Exception as e:
                print(f"  Error processing {key}: {e}")

        if not results:
            print("No models processed for this p. Skipping.")
            continue

        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / f"activation_regions_gnp{p:.1f}_seed{args.seed}.pdf"
        plot_activation_regions_panel(results, out_path,
                                      prob_cmap=args.prob_cmap,
                                      grid_res=args.grid_res,
                                      show_hull=(not args.no_hull))


if __name__ == '__main__':
    main()
