#!/usr/bin/env python3
"""
Compute Optimal Transport distances between edge-embedding distributions across graph sizes n
for three models (Homogeneous, Heterogeneous, MLP-only), using n_ref as the reference domain.
This mirrors compute_ot_mass_vs_p.py but sweeps sizes instead of probabilities.

Example:
  python -m scheduler.viz_results.decision_boundaries.compute_ot_mass_vs_n \
    --homogeneous-model logs/.../no_global_actor_best_model.pt \
    --heterogeneous-model logs/.../hetero_best_model.pt \
    --mlp-model logs/.../mlp_only_best_model.pt \
    --ns "12,16,20,24,32" \
    --n-ref 24 \
    --p 0.8 \
    --hosts 4 --vms 10 --workflows 10 \
    --episodes 3 --device cpu \
    --out-dir scheduler/viz_results/decision_boundaries/ot_results_sizes
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Reuse utilities from the p-sweep script
from cogito.viz_results.decision_boundaries.compute_ot_mass_vs_p import (
    normalize,
    sliced_wasserstein_distance,
    sinkhorn_distance,
    gw_distance,
    barycentric_displacements,
    plot_displacement_hist,
    plot_transport_arrows,
)
from cogito.viz_results.compare_embeddings_panel import (
    build_args_gnp,
    load_agent,
    extract_embeddings_and_labels,
)
from cogito.gnn_deeprl_model.ablation_gnn import AblationGinAgent


def collect_edge_embeddings_n(
    agent: AblationGinAgent,
    p: float,
    n: int,
    hosts: int,
    vms: int,
    workflows: int,
    device: str,
    episodes: int,
    seed_base: int = 1_000_000_001,
) -> np.ndarray:
    """Collect edge embeddings for fixed size n by setting gnp_min_n=gnp_max_n=n."""
    args = build_args_gnp(p, hosts, vms, workflows, n, n, device)
    X, _labels = extract_embeddings_and_labels(agent, args, episodes=episodes, seed_base=seed_base)
    return X.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='OT analysis of edge embeddings across sizes n w.r.t. n_ref')
    # Model paths
    parser.add_argument('--homogeneous-model', type=str, required=True)
    parser.add_argument('--heterogeneous-model', type=str, required=True)
    parser.add_argument('--mlp-model', type=str, required=True)
    # Dataset (fixed p treated as constant graphon)
    parser.add_argument('--p', type=float, default=0.8)
    parser.add_argument('--n-ref', type=int, default=24)
    parser.add_argument('--ns', type=str, default='12,16,20,24,32', help='Comma-separated n values to compare to n_ref')
    parser.add_argument('--hosts', type=int, default=4)
    parser.add_argument('--vms', type=int, default=10)
    parser.add_argument('--workflows', type=int, default=10)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out-dir', type=Path, default=Path('scheduler/viz_results/decision_boundaries/ot_results_sizes'))
    # Dist options (forwarded to reused functions)
    parser.add_argument('--normalize', type=str, default='l2', choices=['none','l2','z'])
    parser.add_argument('--sinkhorn-reg', type=float, default=1e-2)
    parser.add_argument('--sinkhorn-iter', type=int, default=3000)
    parser.add_argument('--gw-epsilon', type=float, default=5e-3)
    parser.add_argument('--swd-proj', type=int, default=128)
    parser.add_argument('--swd-seed', type=int, default=0)
    parser.add_argument('--plot-displacements', action='store_true', help='If set, saves GW barycentric displacement hist/CDF per model and n')
    parser.add_argument('--make-arrow-plots', action='store_true', help='If set, saves OT arrow plots (Y → barycentric(X)) per model and n')
    parser.add_argument('--viz-project', type=str, default='pca', choices=['pca','tsne'], help='2D projection method for arrow plots')
    parser.add_argument('--tsne-perplexity', type=float, default=30.0)
    parser.add_argument('--tsne-n-iter', type=int, default=1000)
    parser.add_argument('--max-arrows', type=int, default=400)
    # Sampling caps
    parser.add_argument('--max-ref', type=int, default=20000, help='Max embeddings to keep from reference set (per model)')
    parser.add_argument('--max-eval', type=int, default=20000, help='Max embeddings to keep from eval set (per model per n)')

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
            print(f"[ot-n] Skipping {key}: checkpoint not found: {pth}")
            continue
        try:
            agents[key] = load_agent(str(pth), device=device, variant_name=variant)  # type: ignore[arg-type]
        except Exception as e:
            print(f"[ot-n] Failed to load {key}: {e}")

    if not agents:
        print("[ot-n] No models loaded. Exiting.")
        return

    # Reference embeddings at n_ref
    print(f"[ot-n] Collecting reference embeddings at n_ref={args.n_ref} (p={args.p}) ...")
    X_ref: Dict[str, np.ndarray] = {}
    for key, _, _ in model_specs:
        if key not in agents:
            continue
        X = collect_edge_embeddings_n(
            agents[key], args.p, int(args.n_ref), args.hosts, args.vms, args.workflows,
            args.device, args.episodes
        )
        if X.shape[0] > args.max_ref:
            idx = np.random.default_rng(0).choice(X.shape[0], size=args.max_ref, replace=False)
            X = X[idx]
        X_ref[key] = normalize(X, args.normalize) if args.normalize != 'none' else X
        print(f"  - {key}: ref embeddings shape = {X.shape}")

    # Parse n list
    try:
        ns_list = [int(x) for x in args.ns.split(',') if x.strip()]
    except Exception:
        print(f"[ot-n] Failed to parse --ns '{args.ns}'.")
        return

    # Prepare CSV writer
    import csv
    csv_path = args.out_dir / f"ot_vs_n_nref_{int(args.n_ref)}.csv"
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model','n_ref','n_eval','sinkhorn','gw','disp_mean','disp_median','swd','n_ref_emb','n_eval_emb'])
        w.writeheader()
        for n in ns_list:
            print(f"\n[ot-n] Evaluating n={n}")
            for key, _, _ in model_specs:
                if key not in agents or key not in X_ref:
                    continue
                Xr = X_ref[key]
                Xe = collect_edge_embeddings_n(
                    agents[key], args.p, int(n), args.hosts, args.vms, args.workflows,
                    args.device, args.episodes
                )
                if Xe.shape[0] > args.max_eval:
                    idx = np.random.default_rng(1).choice(Xe.shape[0], size=args.max_eval, replace=False)
                    Xe = Xe[idx]
                Xe = normalize(Xe, args.normalize) if args.normalize != 'none' else Xe

                sink = sinkhorn_distance(Xr, Xe, reg=args.sinkhorn_reg, num_iter=args.sinkhorn_iter)
                gw, Pi = gw_distance(Xr, Xe, metric='cosine', epsilon=args.gw_epsilon)
                swd = sliced_wasserstein_distance(Xr, Xe, n_projections=args.swd_proj, seed=args.swd_seed)

                disp_mean = float('nan'); disp_median = float('nan')
                if Pi is not None:
                    d = barycentric_displacements(Xr, Xe, Pi)
                    if d.size:
                        disp_mean = float(np.mean(d))
                        disp_median = float(np.median(d))
                        if args.plot_displacements:
                            out_hist = args.out_dir / f"disp_hist_{key}_n{int(n)}_nref{int(args.n_ref)}.png"
                            plot_displacement_hist(Xr, Xe, Pi, out_hist, title=f"{key} | n_ref={int(args.n_ref)} → n={int(n)}")
                        if args.make_arrow_plots:
                            out_arw = args.out_dir / f"ot_plan_{key}_n{int(n)}_nref{int(args.n_ref)}.png"
                            plot_transport_arrows(
                                Xr, Xe, Pi, out_arw,
                                title=f"{key} | OT plan (n_ref={int(args.n_ref)} → n={int(n)})",
                                max_arrows=args.max_arrows,
                                project=args.viz_project,
                                tsne_perplexity=args.tsne_perplexity,
                                tsne_n_iter=args.tsne_n_iter,
                            )
                rec = {
                    'model': key,
                    'n_ref': int(args.n_ref),
                    'n_eval': int(n),
                    'sinkhorn': sink,
                    'gw': gw,
                    'disp_mean': disp_mean,
                    'disp_median': disp_median,
                    'swd': swd,
                    'n_ref_emb': Xr.shape[0],
                    'n_eval_emb': Xe.shape[0],
                }
                w.writerow(rec)
                print(f"  {key:13s} | sinkhorn={sink:.5g} | gw={gw:.5g} | disp_mean={disp_mean:.5g} | swd={swd:.5g} | n_emb=({Xr.shape[0]},{Xe.shape[0]})")

    print(f"[ot-n] Wrote: {csv_path}")

    # Simple plots: distance vs n per model
    import pandas as pd
    import matplotlib.pyplot as plt
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ot-n] Failed to read CSV for plotting: {e}")
        return

    plt.figure(figsize=(11.0, 3.6))
    plt.subplot(1, 3, 1)
    for key, _, _ in model_specs:
        if key not in agents:
            continue
        dsub = df[df['model'] == key]
        if dsub.empty:
            continue
        plt.plot(dsub['n_eval'], dsub['sinkhorn'], 'o-', label=key.capitalize(), linewidth=2, markersize=6)
    plt.xlabel('n (eval)')
    plt.ylabel('Sinkhorn distance (ref n={})'.format(int(args.n_ref)))
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 2)
    for key, _, _ in model_specs:
        if key not in agents:
            continue
        dsub = df[df['model'] == key]
        if dsub.empty:
            continue
        plt.plot(dsub['n_eval'], dsub['swd'], 'o-', label=key.capitalize(), linewidth=2, markersize=6)
    plt.xlabel('n (eval)')
    plt.ylabel('Sliced W2 distance (ref n={})'.format(int(args.n_ref)))
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    for key, _, _ in model_specs:
        if key not in agents:
            continue
        dsub = df[df['model'] == key]
        if dsub.empty:
            continue
        plt.plot(dsub['n_eval'], dsub['gw'], 'o-', label=key.capitalize(), linewidth=2, markersize=6)
    plt.xlabel('n (eval)')
    plt.ylabel('GW2 (cosine) (ref n={})'.format(int(args.n_ref)))
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = args.out_dir / f"ot_vs_n_nref_{int(args.n_ref)}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plot_pdf = plot_path.with_suffix('.pdf')
    plt.savefig(plot_pdf, bbox_inches='tight')
    print(f"[ot-n] Saved plots: {plot_path} and {plot_pdf}")


if __name__ == '__main__':
    main()
