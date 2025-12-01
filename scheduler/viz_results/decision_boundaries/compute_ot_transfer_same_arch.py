#!/usr/bin/env python3
"""
Transfer OT analysis between two checkpoints of the SAME architecture
(one trained at GNP p=0.8 and the other at GNP p=0.3), evaluated on
p_eval in {0.3, 0.8} (configurable).

For each architecture (homogeneous, heterogeneous, mlp), we:
  1) Load two agents: one trained @ p=0.8 (A), one trained @ p=0.3 (B).
  2) For each eval p in --ps (default "0.3,0.8"), collect edge embeddings
     from both agents on that eval distribution using the same seeds.
  3) Compute OT metrics between A and B embeddings:
     - Sinkhorn (if POT is installed)
     - Gromov–Wasserstein (GW) and barycentric displacement stats
     - Sliced W2 (SWD) fallback/auxiliary
  4) Optionally, save displacement histograms and line-segment OT plans
     (Y → barycentric(X)) in 2D (PCA or TSNE projection).

Outputs:
  - CSV summary: ot_transfer_same_arch.csv
  - Optional: ot_plan_<arch>_<AtoB|BtoA>_p{p_eval}.png/.pdf
  - Optional: disp_hist_<arch>_<AtoB|BtoA>_p{p_eval}.png/.pdf
  - Summary plot: ot_transfer_summary.png/.pdf (metrics vs p)

Example:
  python -m scheduler.viz_results.decision_boundaries.compute_ot_transfer_same_arch \
    --homogeneous-p08 logs/.../no_global_actor_best_model.pt \
    --homogeneous-p03 logs/.../no_global_actor_best_model.pt \
    --heterogeneous-p08 logs/.../hetero_best_model.pt \
    --heterogeneous-p03 logs/.../hetero_best_model.pt \
    --mlp-p08 logs/.../mlp_only_best_model.pt \
    --mlp-p03 logs/.../mlp_only_best_model.pt \
    --ps "0.3,0.8" --hosts 4 --vms 10 --workflows 10 --min-tasks 12 --max-tasks 24 \
    --episodes 3 --device cpu --out-dir scheduler/viz_results/decision_boundaries/ot_results
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Optional: POT and sklearn
try:
    import ot  # type: ignore
except Exception:
    ot = None
try:
    from sklearn.metrics.pairwise import pairwise_distances  # type: ignore
except Exception:
    pairwise_distances = None  # type: ignore
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
        try:
            cost = float(ot.emd2(a, b, C))
        except Exception:
            cost = float('nan')
    return cost


def gw_distance(X: np.ndarray, Y: np.ndarray, metric: str = 'cosine', epsilon: float = 5e-3):
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
    if Pi is None or Pi.size == 0:
        return np.zeros((0,), dtype=np.float64)
    col = Pi.sum(axis=0, keepdims=True) + 1e-12
    X_bary_from_Y = (X.T @ (Pi / col)).T
    d = np.linalg.norm(Y - X_bary_from_Y, axis=1)
    return d


def plot_disp_hist(X: np.ndarray, Y: np.ndarray, Pi: np.ndarray, out_path: Path, title: str = ""):
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
    plt.xlabel("||Y - X_bary(Y)||"); plt.ylabel("count"); plt.title("Disp histogram")
    plt.subplot(1, 2, 2)
    plt.plot(d_sorted, cdf, color="#1f77b4")
    plt.xlabel("||Y - X_bary(Y)||"); plt.ylabel("CDF"); plt.title("Disp CDF")
    if title:
        plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path); plt.savefig(out_path.with_suffix('.pdf')); plt.close()


def plot_ot_lines(
    X: np.ndarray, Y: np.ndarray, Pi: np.ndarray, out_path: Path,
    title: str = "", project: str = "pca", tsne_perplexity: float = 30.0, tsne_n_iter: int = 1000,
    max_lines: int = 400
):
    if Pi is None or Pi.size == 0:
        return
    col = Pi.sum(axis=0, keepdims=True) + 1e-12
    XbY = (X.T @ (Pi / col)).T
    disp = np.linalg.norm(Y - XbY, axis=1)
    Z = np.vstack([X, Y, XbY])
    # Project
    if project == "tsne" and TSNE is not None:
        # Handle sklearn TSNE API differences across versions: some do not accept n_iter in __init__
        try:
            Z2 = TSNE(n_components=2, perplexity=tsne_perplexity, n_iter=tsne_n_iter, random_state=0, init="pca").fit_transform(Z)
        except TypeError:
            # Fallback: use default number of iterations if n_iter is not supported in this version
            Z2 = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=0, init="pca").fit_transform(Z)
    else:
        try:
            if PCA is not None:
                Z2 = PCA(n_components=2, random_state=0).fit_transform(Z)
            else:
                Zc = Z - Z.mean(axis=0, keepdims=True)
                U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
                Z2 = Zc @ Vt[:2].T
        except Exception:
            Z2 = Z[:, :2]
    n = X.shape[0]; m = Y.shape[0]
    X2 = Z2[:n]; Y2 = Z2[n:n+m]; Xb2 = Z2[n+m:]
    # Subsample
    idx = np.arange(m)
    if m > max_lines:
        idx = np.random.default_rng(42).choice(m, size=max_lines, replace=False)
    Y2s = Y2[idx]; Xb2s = Xb2[idx]
    plt.figure(figsize=(7.2, 6.2), dpi=140)
    plt.scatter(X2[:,0], X2[:,1], s=6, c="#1f77b4", alpha=0.25, label="Ref X")
    sc = plt.scatter(Y2[:,0], Y2[:,1], s=10, c=disp, cmap="viridis", alpha=0.9, label="Eval Y (colored by disp)")
    cb = plt.colorbar(sc, fraction=0.046, pad=0.04); cb.set_label("||Y - X_bary(Y)||")
    segs = np.stack([Y2s, Xb2s], axis=1)
    lc = LineCollection(segs, colors="#2ca02c", linewidths=0.6, alpha=0.35)
    plt.gca().add_collection(lc)
    if title: plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path); plt.savefig(out_path.with_suffix('.pdf')); plt.close()


def collect_edge_embeddings(agent: AblationGinAgent, p: float, hosts: int, vms: int, workflows: int,
                            min_tasks: int, max_tasks: int, device: str, episodes: int,
                            seed_base: int = 1_000_000_001) -> np.ndarray:
    args = build_args_gnp(p, hosts, vms, workflows, min_tasks, max_tasks, device)
    X, _ = extract_embeddings_and_labels(agent, args, episodes=episodes, seed_base=seed_base)
    return X.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="OT transfer between same-arch checkpoints trained at p=0.8 vs p=0.3")
    # Checkpoint paths: allow missing ones (we'll skip if not all provided for an arch)
    parser.add_argument('--homogeneous-p08', type=str, default="")
    parser.add_argument('--homogeneous-p03', type=str, default="")
    parser.add_argument('--baseline-p08', type=str, default="")
    parser.add_argument('--baseline-p03', type=str, default="")
    parser.add_argument('--heterogeneous-p08', type=str, default="")
    parser.add_argument('--heterogeneous-p03', type=str, default="")
    parser.add_argument('--mlp-p08', type=str, default="")
    parser.add_argument('--mlp-p03', type=str, default="")

    parser.add_argument('--ps', type=str, default='0.3,0.8')
    parser.add_argument('--hosts', type=int, default=4)
    parser.add_argument('--vms', type=int, default=10)
    parser.add_argument('--workflows', type=int, default=10)
    parser.add_argument('--min-tasks', type=int, default=12)
    parser.add_argument('--max-tasks', type=int, default=24)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed-base', type=int, default=1_000_000_001)
    parser.add_argument('--out-dir', type=Path, default=Path('scheduler/viz_results/decision_boundaries/ot_results'))

    # Dist/viz options
    parser.add_argument('--normalize', type=str, default='l2', choices=['none','l2','z'])
    parser.add_argument('--sinkhorn-reg', type=float, default=1e-2)
    parser.add_argument('--sinkhorn-iter', type=int, default=3000)
    parser.add_argument('--gw-epsilon', type=float, default=5e-3)
    parser.add_argument('--swd-proj', type=int, default=128)
    parser.add_argument('--swd-seed', type=int, default=0)
    parser.add_argument('--plot-displacements', action='store_true')
    parser.add_argument('--make-arrow-plots', action='store_true')
    parser.add_argument('--viz-project', type=str, default='pca', choices=['pca','tsne'])
    parser.add_argument('--tsne-perplexity', type=float, default=30.0)
    parser.add_argument('--tsne-n-iter', type=int, default=1000)
    parser.add_argument('--max-arrows', type=int, default=400)

    # Caps
    parser.add_argument('--max-emb', type=int, default=20000)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build model mapping
    arch_specs = {
        'homogeneous': (args.homogeneous_p08, args.homogeneous_p03, 'homogeneous'),
        'baseline': (args.baseline_p08, args.baseline_p03, 'baseline'),
        'heterogeneous': (args.heterogeneous_p08, args.heterogeneous_p03, 'hetero'),
        'mlp': (args.mlp_p08, args.mlp_p03, 'mlp_only'),
    }

    # Load agents where both checkpoints exist
    agents_p08: Dict[str, AblationGinAgent] = {}
    agents_p03: Dict[str, AblationGinAgent] = {}
    for arch, (pth08, pth03, variant) in arch_specs.items():
        if not pth08 or not pth03:
            print(f"[transfer] Skipping {arch}: missing checkpoints")
            continue
        p08 = Path(pth08); p03 = Path(pth03)
        if not (p08.exists() and p03.exists()):
            print(f"[transfer] Skipping {arch}: checkpoint(s) not found")
            continue
        try:
            agents_p08[arch] = load_agent(str(p08), device=args.device, variant_name=variant)  # type: ignore[arg-type]
            agents_p03[arch] = load_agent(str(p03), device=args.device, variant_name=variant)  # type: ignore[arg-type]
        except Exception as e:
            print(f"[transfer] Failed to load {arch}: {e}")

    if not agents_p08:
        print("[transfer] No architectures loaded. Exiting.")
        return

    # Parse eval p list
    try:
        p_list = [float(x) for x in args.ps.split(',') if x.strip()]
    except Exception:
        print(f"[transfer] Failed to parse --ps '{args.ps}'.")
        return

    # CSV writer
    import csv
    csv_path = args.out_dir / 'ot_transfer_same_arch.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'arch','p_eval','direction','sinkhorn','gw','disp_mean','disp_median','swd','n_ref','n_eval'
        ])
        w.writeheader()
        rows: List[dict] = []

        for p_eval in p_list:
            print(f"\n[transfer] Evaluating on p={p_eval}")
            for arch in agents_p08.keys():
                if arch not in agents_p03:
                    continue
                # Collect embeddings with same seeds for both agents
                X08 = collect_edge_embeddings(agents_p08[arch], p_eval, args.hosts, args.vms, args.workflows,
                                              args.min_tasks, args.max_tasks, args.device, args.episodes,
                                              seed_base=args.seed_base)
                X03 = collect_edge_embeddings(agents_p03[arch], p_eval, args.hosts, args.vms, args.workflows,
                                              args.min_tasks, args.max_tasks, args.device, args.episodes,
                                              seed_base=args.seed_base)
                if X08.shape[0] > args.max_emb:
                    idx = np.random.default_rng(0).choice(X08.shape[0], size=args.max_emb, replace=False)
                    X08 = X08[idx]
                if X03.shape[0] > args.max_emb:
                    idx = np.random.default_rng(0).choice(X03.shape[0], size=args.max_emb, replace=False)
                    X03 = X03[idx]
                if args.normalize != 'none':
                    X08 = normalize(X08, args.normalize)
                    X03 = normalize(X03, args.normalize)

                # A) A:=p08 as REF, B:=p03 as EVAL
                sink = sinkhorn_distance(X08, X03, reg=args.sinkhorn_reg, num_iter=args.sinkhorn_iter)
                gw, Pi = gw_distance(X08, X03, metric='cosine', epsilon=args.gw_epsilon)
                swd = sliced_wasserstein_distance(X08, X03, n_projections=args.swd_proj, seed=args.swd_seed)
                disp_mean = float('nan'); disp_median = float('nan')
                if Pi is not None:
                    d = barycentric_displacements(X08, X03, Pi)
                    if d.size:
                        disp_mean = float(np.mean(d)); disp_median = float(np.median(d))
                        if args.plot_displacements:
                            out_hist = args.out_dir / f"disp_hist_{arch}_p08to_p03_p{p_eval:.1f}.png"
                            plot_disp_hist(X08, X03, Pi, out_hist, title=f"{arch} p08→p03 | p_eval={p_eval}")
                        if args.make_arrow_plots:
                            out_plan = args.out_dir / f"ot_plan_{arch}_p08to_p03_p{p_eval:.1f}.png"
                            plot_ot_lines(X08, X03, Pi, out_plan, title=f"{arch} p08→p03 | p_eval={p_eval}",
                                          project=args.viz_project, tsne_perplexity=args.tsne_perplexity,
                                          tsne_n_iter=args.tsne_n_iter, max_lines=args.max_arrows)
                rec = {
                    'arch': arch,
                    'p_eval': p_eval,
                    'direction': 'p08_to_p03',
                    'sinkhorn': sink,
                    'gw': gw,
                    'disp_mean': disp_mean,
                    'disp_median': disp_median,
                    'swd': swd,
                    'n_ref': X08.shape[0],
                    'n_eval': X03.shape[0],
                }
                w.writerow(rec); rows.append(rec)
                print(f"  [{arch}] p08→p03 | sink={sink:.5g} gw={gw:.5g} disp_mean={disp_mean:.5g} swd={swd:.5g} n=({X08.shape[0]},{X03.shape[0]})")

                # B) B:=p03 as REF, A:=p08 as EVAL (reverse direction)
                sink_r = sinkhorn_distance(X03, X08, reg=args.sinkhorn_reg, num_iter=args.sinkhorn_iter)
                gw_r, Pi_r = gw_distance(X03, X08, metric='cosine', epsilon=args.gw_epsilon)
                swd_r = sliced_wasserstein_distance(X03, X08, n_projections=args.swd_proj, seed=args.swd_seed)
                disp_mean_r = float('nan'); disp_median_r = float('nan')
                if Pi_r is not None:
                    d_r = barycentric_displacements(X03, X08, Pi_r)
                    if d_r.size:
                        disp_mean_r = float(np.mean(d_r)); disp_median_r = float(np.median(d_r))
                        if args.plot_displacements:
                            out_hist = args.out_dir / f"disp_hist_{arch}_p03to_p08_p{p_eval:.1f}.png"
                            plot_disp_hist(X03, X08, Pi_r, out_hist, title=f"{arch} p03→p08 | p_eval={p_eval}")
                        if args.make_arrow_plots:
                            out_plan = args.out_dir / f"ot_plan_{arch}_p03to_p08_p{p_eval:.1f}.png"
                            plot_ot_lines(X03, X08, Pi_r, out_plan, title=f"{arch} p03→p08 | p_eval={p_eval}",
                                          project=args.viz_project, tsne_perplexity=args.tsne_perplexity,
                                          tsne_n_iter=args.tsne_n_iter, max_lines=args.max_arrows)
                rec_r = {
                    'arch': arch,
                    'p_eval': p_eval,
                    'direction': 'p03_to_p08',
                    'sinkhorn': sink_r,
                    'gw': gw_r,
                    'disp_mean': disp_mean_r,
                    'disp_median': disp_median_r,
                    'swd': swd_r,
                    'n_ref': X03.shape[0],
                    'n_eval': X08.shape[0],
                }
                w.writerow(rec_r); rows.append(rec_r)
                print(f"  [{arch}] p03→p08 | sink={sink_r:.5g} gw={gw_r:.5g} disp_mean={disp_mean_r:.5g} swd={swd_r:.5g} n=({X03.shape[0]},{X08.shape[0]})")

    print(f"[transfer] Wrote summary: {csv_path}")

    # Make a simple 3-panel summary per metric vs p with both directions
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[transfer] Could not load CSV for plotting: {e}")
        return

    plt.figure(figsize=(10.5, 3.6))
    metrics = [('sinkhorn','Sinkhorn'), ('swd','Sliced W2'), ('gw','GW2 (cosine)')]
    for i, (col, label) in enumerate(metrics, start=1):
        plt.subplot(1, 3, i)
        for arch in sorted(df['arch'].unique()):
            for direction, ls in [('p08_to_p03','-'), ('p03_to_p08','--')]:
                sub = df[(df['arch']==arch) & (df['direction']==direction)]
                if sub.empty:
                    continue
                plt.plot(sub['p_eval'], sub[col], ls, marker='o', label=f"{arch} {direction}")
        plt.xlabel('GNP p (eval)'); plt.ylabel(label); plt.grid(True, alpha=0.3)
    # One legend outside
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', borderaxespad=0., fontsize=8)
    plt.tight_layout(rect=[0,0,0.8,1])
    plot_path = args.out_dir / 'ot_transfer_summary.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"[transfer] Saved summary plots: {plot_path} and {plot_path.with_suffix('.pdf')}")


if __name__ == '__main__':
    main()
