#!/usr/bin/env python3
"""
Compute Optimal Transport distances between edge-embedding distributions across GNP p values
for three models (Homogeneous, Heterogeneous, MLP-only), using p_ref (default 0.8) as the
reference domain. Produces a CSV summary and a plot of distance vs p per model.

- Embeddings: edge embeddings extracted as in compare_embeddings_panel.py
- Distance: Sinkhorn (POT) if available; otherwise falls back to Sliced Wasserstein Distance (SWD)

Example:
  python -m scheduler.viz_results.decision_boundaries.compute_ot_mass_vs_p 
    --homogeneous-model logs/.../no_global_actor_best_model.pt 
    --heterogeneous-model logs/.../hetero_best_model.pt 
    --mlp-model logs/.../mlp_only_best_model.pt 
    --ps "0.3,0.4,0.5,0.6,0.7,0.8" 
    --p-ref 0.8 
    --hosts 4 --vms 10 --workflows 10 --min-tasks 12 --max-tasks 24 
    --episodes 3 --device cpu 
    --out-dir scheduler/viz_results/decision_boundaries/ot_results
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Optional dependency: POT
try:
    import ot  # type: ignore
except Exception:
    ot = None  # Fallback to SWD

# Optional dependency: scikit-learn pairwise distances for GW
try:
    from sklearn.metrics.pairwise import pairwise_distances  # type: ignore
except Exception:
    pairwise_distances = None  # type: ignore

# Optional dimensionality reduction for arrow plots
try:
    from sklearn.decomposition import PCA  # type: ignore
except Exception:
    PCA = None  # type: ignore
try:
    from sklearn.manifold import TSNE  # type: ignore
except Exception:
    TSNE = None  # type: ignore

# Project imports
from scheduler.viz_results.compare_embeddings_panel import (
    build_args_gnp,
    load_agent,
    extract_embeddings_and_labels,
)
from scheduler.rl_model.ablation_gnn import AblationGinAgent


def normalize(X: np.ndarray, kind: str = "l2") -> np.ndarray:
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


def sliced_wasserstein_distance(X: np.ndarray, Y: np.ndarray, n_projections: int = 128, seed: int = 0) -> float:
    """Approximate 2-Wasserstein via Sliced Wasserstein using random 1D projections.
    This does not require POT and is reasonably fast.
    """
    if X.size == 0 or Y.size == 0:
        return float('nan')
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    proj = rng.standard_normal(size=(n_projections, d))
    proj /= (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
    swd2 = 0.0
    for v in proj:
        Xp = X @ v
        Yp = Y @ v
        Xp_sorted = np.sort(Xp)
        Yp_sorted = np.sort(Yp)
        # Match lengths by linear interpolation if unequal sizes
        n = Xp_sorted.shape[0]
        m = Yp_sorted.shape[0]
        if n != m:
            t = np.linspace(0, 1, max(n, m))
            Xp_sorted = np.interp(t, np.linspace(0, 1, n), Xp_sorted)
            Yp_sorted = np.interp(t, np.linspace(0, 1, m), Yp_sorted)
        swd2 += np.mean((Xp_sorted - Yp_sorted) ** 2)
    swd2 /= n_projections
    return float(np.sqrt(max(0.0, swd2)))


def sinkhorn_distance(X: np.ndarray, Y: np.ndarray, reg: float = 1e-2, num_iter: int = 3000) -> float:
    if ot is None:
        return float('nan')
    if X.size == 0 or Y.size == 0:
        return float('nan')
    a = np.ones(X.shape[0], dtype=np.float64) / X.shape[0]
    b = np.ones(Y.shape[0], dtype=np.float64) / Y.shape[0]
    C = ot.utils.dist(X, Y, metric='euclidean')
    C /= (C.max() + 1e-12)
    try:
        W = ot.sinkhorn(a, b, C, reg, numItermax=num_iter)
        cost = float((W * C).sum())
    except Exception:
        # Fallback to emd2 if sinkhorn fails (may be slow for large sets)
        try:
            cost = float(ot.emd2(a, b, C))
        except Exception:
            cost = float('nan')
    return cost


def gw_distance(X: np.ndarray, Y: np.ndarray, metric: str = 'cosine', epsilon: float = 5e-3) -> tuple[float, np.ndarray | None]:
    """Compute Gromov–Wasserstein-2 discrepancy and return (gw_value, coupling Pi).
    Requires POT and scikit-learn. Returns (nan, None) if unavailable or on failure.
    """
    if ot is None or pairwise_distances is None:
        return float('nan'), None
    if X.size == 0 or Y.size == 0:
        return float('nan'), None
    try:
        n, m = X.shape[0], Y.shape[0]
        a = np.ones(n, dtype=np.float64) / max(1, n)
        b = np.ones(m, dtype=np.float64) / max(1, m)
        DX = pairwise_distances(X, X, metric=metric)
        DY = pairwise_distances(Y, Y, metric=metric)
        gw, log = ot.gromov.gromov_wasserstein2(DX, DY, a, b, loss_fun='square_loss', epsilon=epsilon, log=True)
        Pi = log.get('T', None)
        return float(gw), (Pi if isinstance(Pi, np.ndarray) else None)
    except Exception:
        return float('nan'), None


def barycentric_displacements(X: np.ndarray, Y: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    """Project Y into X via barycentric map under coupling Pi and return per-point displacement norms.
    X: (n,d), Y: (m,d), Pi: (n,m) coupling
    """
    if Pi is None or Pi.size == 0:
        return np.zeros((0,), dtype=np.float64)
    col = Pi.sum(axis=0, keepdims=True) + 1e-12
    X_bary_from_Y = (X.T @ (Pi / col)).T  # (m, d)
    d = np.linalg.norm(Y - X_bary_from_Y, axis=1)
    return d


def plot_displacement_hist(X: np.ndarray, Y: np.ndarray, Pi: np.ndarray, out_path: Path, title: str = "") -> None:
    """Save histogram + CDF of ||Y - X_bary(Y)|| under coupling Pi to PNG and PDF.
    """
    if Pi is None or Pi.size == 0:
        return
    d = barycentric_displacements(X, Y, Pi)
    if d.size == 0:
        return
    d_sorted = np.sort(d)
    cdf = np.arange(1, len(d_sorted) + 1) / len(d_sorted)
    plt.figure(figsize=(7, 5), dpi=140)
    plt.subplot(1, 2, 1)
    plt.hist(d, bins=30, color="#2ca02c", alpha=0.85)
    plt.xlabel("||Y - X_bary(Y)||")
    plt.ylabel("count")
    plt.title("Displacement histogram")
    plt.subplot(1, 2, 2)
    plt.plot(d_sorted, cdf, color="#1f77b4")
    plt.xlabel("||Y - X_bary(Y)||")
    plt.ylabel("CDF")
    plt.title("Displacement CDF")
    if title:
        plt.suptitle(title) 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    pdf_path = out_path.with_suffix('.pdf')
    plt.savefig(pdf_path)
    plt.close()


def plot_transport_arrows(
    X: np.ndarray,
    Y: np.ndarray,
    Pi: np.ndarray,
    out_path: Path,
    title: str = "",
    max_arrows: int = 400,
    project: str = "pca",
    tsne_perplexity: float = 30.0,
    tsne_n_iter: int = 1000,
):
    """Project to 2D and draw arrows from each sampled Y point to its barycentric projection in X under Pi.
    Saves to out_path (PNG) and alongside PDF.
    """
    if Pi is None or Pi.size == 0:
        return
    # Barycentric projections of Y in X space
    col = Pi.sum(axis=0, keepdims=True) + 1e-12
    XbY = (X.T @ (Pi / col)).T  # (m, d)
    # Displacements in original space for coloring
    disp = np.linalg.norm(Y - XbY, axis=1)

    # Joint 2D projection for stable geometry
    Z = np.vstack([X, Y, XbY])
    # Choose projector
    if project == "tsne" and TSNE is not None:
        projector = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=0, init="pca")
        Z2 = projector.fit_transform(Z)
    else:
        # PCA fallback (sklearn if available, else NumPy SVD)
        try:
            if PCA is not None:
                Z2 = PCA(n_components=2, random_state=0).fit_transform(Z)
            else:
                # NumPy SVD PCA
                Zc = Z - Z.mean(axis=0, keepdims=True)
                U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
                Z2 = Zc @ Vt[:2].T
        except Exception:
            # Final fallback: take first two dims
            Z2 = Z[:, :2]

    n = X.shape[0]
    m = Y.shape[0]
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
        Y2s = Y2
        Xb2s = Xb2

    plt.figure(figsize=(7.2, 6.2), dpi=140)
    # Train/reference cloud lightly
    plt.scatter(X2[:, 0], X2[:, 1], s=6, c="#1f77b4", alpha=0.25, label="Ref X (p_ref)")
    # Eval points colored by displacement
    sc = plt.scatter(
        Y2[:, 0], Y2[:, 1], s=10, c=disp, cmap="viridis", alpha=0.9,
        label="Eval Y (p) — colored by ||Y−X_bary(Y)||"
    )
    cb = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cb.set_label("||Y − X_bary(Y)||")
    # Line segments (no arrowheads) from Y to barycentric projection in X
    ax = plt.gca()
    segments = np.stack([Y2s, Xb2s], axis=1)  # (k, 2, 2)
    lc = LineCollection(segments, colors="#2ca02c", linewidths=0.6, alpha=0.35)
    ax.add_collection(lc)
    if title:
        plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    pdf_path = out_path.with_suffix('.pdf')
    plt.savefig(pdf_path)
    plt.close()


def collect_edge_embeddings(agent: AblationGinAgent, p: float, hosts: int, vms: int, workflows: int,
                            min_tasks: int, max_tasks: int, device: str, episodes: int,
                            seed_base: int = 1_000_000_001) -> np.ndarray:
    args = build_args_gnp(p, hosts, vms, workflows, min_tasks, max_tasks, device)
    X, _labels = extract_embeddings_and_labels(agent, args, episodes=episodes, seed_base=seed_base)
    return X.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='OT analysis of edge embeddings across GNP p values w.r.t. p_ref')
    # Model paths
    parser.add_argument('--homogeneous-model', type=str, required=True)
    parser.add_argument('--heterogeneous-model', type=str, required=True)
    parser.add_argument('--mlp-model', type=str, required=True)
    # Dataset
    parser.add_argument('--p-ref', type=float, default=0.8)
    parser.add_argument('--ps', type=str, default='0.3,0.4,0.5,0.6,0.7,0.8', help='Comma-separated p values to compare to p_ref')
    parser.add_argument('--hosts', type=int, default=4)
    parser.add_argument('--vms', type=int, default=10)
    parser.add_argument('--workflows', type=int, default=10)
    parser.add_argument('--min-tasks', type=int, default=12)
    parser.add_argument('--max-tasks', type=int, default=24)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out-dir', type=Path, default=Path('scheduler/viz_results/decision_boundaries/ot_results'))
    # Dist options
    parser.add_argument('--normalize', type=str, default='l2', choices=['none','l2','z'])
    parser.add_argument('--sinkhorn-reg', type=float, default=1e-2)
    parser.add_argument('--sinkhorn-iter', type=int, default=3000)
    parser.add_argument('--gw-epsilon', type=float, default=5e-3)
    parser.add_argument('--swd-proj', type=int, default=128)
    parser.add_argument('--swd-seed', type=int, default=0)
    parser.add_argument('--plot-displacements', action='store_true', help='If set, saves GW barycentric displacement hist/CDF per model and p')
    parser.add_argument('--make-arrow-plots', action='store_true', help='If set, saves OT arrow plots (Y → barycentric(X)) per model and p')
    parser.add_argument('--viz-project', type=str, default='pca', choices=['pca','tsne'], help='2D projection method for arrow plots')
    parser.add_argument('--tsne-perplexity', type=float, default=30.0)
    parser.add_argument('--tsne-n-iter', type=int, default=1000)
    parser.add_argument('--max-arrows', type=int, default=400)
    # Sampling caps
    parser.add_argument('--max-ref', type=int, default=20000, help='Max embeddings to keep from reference set (per model)')
    parser.add_argument('--max-eval', type=int, default=20000, help='Max embeddings to keep from eval set (per model per p)')

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device

    # Load models via compare_embeddings_panel.load_agent
    model_specs = [
        ('homogeneous', args.homogeneous_model, 'homogeneous'),
        ('heterogeneous', args.heterogeneous_model, 'hetero'),
        ('mlp', args.mlp_model, 'mlp_only'),
    ]

    agents: Dict[str, AblationGinAgent] = {}
    for key, path, variant in model_specs:
        pth = Path(path)
        if not pth.exists():
            print(f"[ot] Skipping {key}: checkpoint not found: {pth}")
            continue
        try:
            agents[key] = load_agent(str(pth), device=device, variant_name=variant)  # type: ignore[arg-type]
        except Exception as e:
            print(f"[ot] Failed to load {key}: {e}")

    if not agents:
        print("[ot] No models loaded. Exiting.")
        return

    # Reference embeddings at p_ref
    print(f"[ot] Collecting reference embeddings at p_ref={args.p_ref} ...")
    X_ref: Dict[str, np.ndarray] = {}
    for key, _, _ in model_specs:
        if key not in agents:
            continue
        X = collect_edge_embeddings(
            agents[key], args.p_ref, args.hosts, args.vms, args.workflows,
            args.min_tasks, args.max_tasks, args.device, args.episodes
        )
        # Subsample cap
        if X.shape[0] > args.max_ref:
            idx = np.random.default_rng(0).choice(X.shape[0], size=args.max_ref, replace=False)
            X = X[idx]
        X_ref[key] = normalize(X, args.normalize) if args.normalize != 'none' else X
        print(f"  - {key}: ref embeddings shape = {X.shape}")

    # Parse p list
    try:
        ps_list = [float(x) for x in args.ps.split(',') if x.strip()]
    except Exception:
        print(f"[ot] Failed to parse --ps '{args.ps}'.")
        return

    # Prepare CSV writer
    import csv
    csv_path = args.out_dir / f"ot_vs_p_pref_{args.p_ref:.1f}.csv"
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model','p_ref','p_eval','sinkhorn','gw','disp_mean','disp_median','swd','n_ref','n_eval'])
        w.writeheader()
        # For each p, compute distances per model
        records: List[dict] = []
        for p in ps_list:
            print(f"\n[ot] Evaluating p={p:.3f}")
            for key, _, _ in model_specs:
                if key not in agents or key not in X_ref:
                    continue
                Xr = X_ref[key]
                # Collect eval embeddings
                Xe = collect_edge_embeddings(
                    agents[key], p, args.hosts, args.vms, args.workflows,
                    args.min_tasks, args.max_tasks, args.device, args.episodes
                )
                if Xe.shape[0] > args.max_eval:
                    idx = np.random.default_rng(1).choice(Xe.shape[0], size=args.max_eval, replace=False)
                    Xe = Xe[idx]
                Xe = normalize(Xe, args.normalize) if args.normalize != 'none' else Xe

                # Compute distances
                sink = sinkhorn_distance(Xr, Xe, reg=args.sinkhorn_reg, num_iter=args.sinkhorn_iter)
                gw, Pi = gw_distance(Xr, Xe, metric='cosine', epsilon=args.gw_epsilon)
                swd = sliced_wasserstein_distance(Xr, Xe, n_projections=args.swd_proj, seed=args.swd_seed)
                # Displacement stats and plots (if GW available)
                disp_mean = float('nan'); disp_median = float('nan')
                if Pi is not None:
                    d = barycentric_displacements(Xr, Xe, Pi)
                    if d.size:
                        disp_mean = float(np.mean(d))
                        disp_median = float(np.median(d))
                        if args.plot_displacements:
                            out_hist = args.out_dir / f"disp_hist_{key}_p{p:.1f}_pref{args.p_ref:.1f}.png"
                            plot_displacement_hist(Xr, Xe, Pi, out_hist, title=f"{key} | p_ref={args.p_ref:.1f} → p={p:.1f}")
                        if args.make_arrow_plots:
                            out_arw = args.out_dir / f"ot_plan_{key}_p{p:.1f}_pref{args.p_ref:.1f}.png"
                            plot_transport_arrows(
                                Xr, Xe, Pi, out_arw,
                                title=f"{key} | OT plan (p_ref={args.p_ref:.1f} → p={p:.1f})",
                                max_arrows=args.max_arrows,
                                project=args.viz_project,
                                tsne_perplexity=args.tsne_perplexity,
                                tsne_n_iter=args.tsne_n_iter,
                            )
                rec = {
                    'model': key,
                    'p_ref': args.p_ref,
                    'p_eval': p,
                    'sinkhorn': sink,
                    'gw': gw,
                    'disp_mean': disp_mean,
                    'disp_median': disp_median,
                    'swd': swd,
                    'n_ref': Xr.shape[0],
                    'n_eval': Xe.shape[0],
                }
                w.writerow(rec)
                records.append(rec)
                print(f"  {key:13s} | sinkhorn={sink:.5g} | gw={gw:.5g} | disp_mean={disp_mean:.5g} | swd={swd:.5g} | n=({Xr.shape[0]},{Xe.shape[0]})")

    print(f"[ot] Wrote: {csv_path}")

    # Make a simple plot: distance vs p per model
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ot] Failed to read CSV for plotting: {e}")
        return

    plt.figure(figsize=(11.0, 3.6))
    plt.subplot(1, 3, 1)
    for key, _, _ in model_specs:
        if key not in agents:
            continue
        dsub = df[df['model'] == key]
        if dsub.empty:
            continue
        plt.plot(dsub['p_eval'], dsub['sinkhorn'], 'o-', label=key.capitalize(), linewidth=2, markersize=6)
    plt.xlabel('GNP p (eval)')
    plt.ylabel('Sinkhorn distance (ref p={:.1f})'.format(args.p_ref))
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 2)
    for key, _, _ in model_specs:
        if key not in agents:
            continue
        dsub = df[df['model'] == key]
        if dsub.empty:
            continue
        plt.plot(dsub['p_eval'], dsub['swd'], 'o-', label=key.capitalize(), linewidth=2, markersize=6)
    plt.xlabel('GNP p (eval)')
    plt.ylabel('Sliced W2 distance (ref p={:.1f})'.format(args.p_ref))
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    for key, _, _ in model_specs:
        if key not in agents:
            continue
        dsub = df[df['model'] == key]
        if dsub.empty:
            continue
        plt.plot(dsub['p_eval'], dsub['gw'], 'o-', label=key.capitalize(), linewidth=2, markersize=6)
    plt.xlabel('GNP p (eval)')
    plt.ylabel('GW2 (cosine) (ref p={:.1f})'.format(args.p_ref))
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = args.out_dir / f"ot_vs_p_p{args.p_ref:.1f}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plot_pdf = plot_path.with_suffix('.pdf')
    plt.savefig(plot_pdf, bbox_inches='tight')
    print(f"[ot] Saved plots: {plot_path} and {plot_pdf}")


if __name__ == '__main__':
    main()
