#!/usr/bin/env python3
"""
Compare low-pass regularization (lambda) effect on size and density generalization.

For each lambda model:
- Evaluate size generalization by scaling task range (e.g., 12-24 → 18-30)
- Compute OT distances (Sinkhorn, GW, displacement) vs n
- Fit linear model: distance ~ a + b/sqrt(n) to test graphon-style scaling
- Evaluate density generalization (vs p)
- Generate comparative plots across lambdas

Example:
  python -m scheduler.viz_results.decision_boundaries.compare_lambda_generalization \
    --architecture hetero \
    --lambda-models "0:logs/.../hetero_baseline/ablation/per_variant/hetero_best_model.pt,1e-5:logs/.../hetero_lp1e-5_cache/ablation/per_variant/hetero_best_model.pt,1e-4:logs/.../hetero_lp1e-4_cache/ablation/per_variant/hetero_best_model.pt,1e-3:logs/.../hetero_lp1e-3_cache/ablation/per_variant/hetero_best_model.pt,1e-2:logs/.../hetero_lp1e-2_cache/ablation/per_variant/hetero_best_model.pt" \
    --n-min-base 12 --n-max-base 24 \
    --n-ref 24 \
    --n-scales "1.0,1.25,1.5" \
    --p-base 0.8 \
    --ps "0.3,0.5,0.8" \
    --hosts 4 --vms 10 --workflows 10 \
    --episodes 3 --device cpu \
    --out-dir scheduler/viz_results/decision_boundaries/lambda_comparison
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
_proj_root = Path(__file__).resolve().parents[3]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from scheduler.viz_results.decision_boundaries.compute_ot_mass_vs_n import (
    collect_edge_embeddings_n,
)
from scheduler.viz_results.decision_boundaries.compute_ot_mass_vs_p import (
    normalize,
    sinkhorn_distance,
    gw_distance,
    barycentric_displacements,
    sliced_wasserstein_distance,
    collect_edge_embeddings,
)
from scheduler.viz_results.compare_embeddings_panel import load_agent


def load_lambda_models(lambda_paths_str: str, device: str, architecture: str = "hetero") -> Dict[float, any]:
    """Parse 'lambda:path,lambda:path,...' and load agents."""
    models = {}
    for entry in lambda_paths_str.split(','):
        lam_str, path_str = entry.split(':', 1)
        lam = float(lam_str)
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: checkpoint not found for lambda={lam}: {path}")
            continue
        agent = load_agent(str(path), device=device, variant_name=architecture)
        models[lam] = agent
        print(f"[load] Lambda={lam}: {path}")
    return models


def evaluate_size_generalization(
    agent, 
    n_min: int, 
    n_max: int, 
    n_ref: int,
    n_eval_list: List[int],
    p: float,
    hosts: int,
    vms: int,
    workflows: int,
    episodes: int,
    device: str,
    normalize_kind: str = "l2",
    sinkhorn_reg: float = 1e-2,
    sinkhorn_iter: int = 3000,
    max_samples: int = 20000,
    seed_ref: int = 1_000_000_001,
    seed_eval: int = 1_000_000_002,
    gw_epsilon: float = 5e-3,
    disable_gw: bool = False,
) -> pd.DataFrame:
    """Evaluate agent on size shifts; return DataFrame with OT metrics vs n."""
    # Collect reference embeddings at n_ref
    X_ref = collect_edge_embeddings_n(
        agent, p, n_ref, hosts, vms, workflows, device, episodes, seed_base=seed_ref
    )
    if normalize_kind != "none":
        X_ref = normalize(X_ref, normalize_kind)
    # Subsample reference if needed
    if isinstance(X_ref, np.ndarray) and X_ref.shape[0] > max_samples:
        idx = np.random.default_rng(123).choice(X_ref.shape[0], size=max_samples, replace=False)
        X_ref = X_ref[idx]

    rows = []
    for n_eval in n_eval_list:
        if n_eval < n_min or n_eval > n_max:
            continue
        X_eval = collect_edge_embeddings_n(
            agent, p, n_eval, hosts, vms, workflows, device, episodes, seed_base=seed_eval
        )
        if normalize_kind != "none":
            X_eval = normalize(X_eval, normalize_kind)
        # Subsample eval if needed
        if isinstance(X_eval, np.ndarray) and X_eval.shape[0] > max_samples:
            idx = np.random.default_rng(456).choice(X_eval.shape[0], size=max_samples, replace=False)
            X_eval = X_eval[idx]

        # Compute OT distances
        sink = sinkhorn_distance(X_ref, X_eval, reg=sinkhorn_reg, num_iter=sinkhorn_iter)
        gw_dist, Pi = (float('nan'), None)
        if not disable_gw:
            gw_dist, Pi = gw_distance(X_ref, X_eval, metric='cosine', epsilon=gw_epsilon)
        # Sliced-Wasserstein (projection robust proxy)
        try:
            swd = sliced_wasserstein_distance(X_ref, X_eval, n_projections=128, seed=0)
        except Exception:
            swd = float('nan')
        
        # Barycentric displacements
        disp_stats = {}
        if Pi is not None and Pi.size > 0:
            disps = barycentric_displacements(X_ref, X_eval, Pi)
            disp_stats = {
                'disp_mean': float(np.mean(disps)),
                'disp_median': float(np.median(disps)),
                'disp_std': float(np.std(disps)),
            }
        else:
            disp_stats = {'disp_mean': np.nan, 'disp_median': np.nan, 'disp_std': np.nan}

        rows.append({
            'n_ref': n_ref,
            'n_eval': n_eval,
            'inv_sqrt_n': 1.0 / np.sqrt(n_eval),
            'sinkhorn': sink,
            'gw': gw_dist,
            'swd': swd,
            **disp_stats
        })
    
    return pd.DataFrame(rows)


def evaluate_density_generalization(
    agent,
    n: int,
    p_ref: float,
    p_eval_list: List[float],
    hosts: int,
    vms: int,
    workflows: int,
    episodes: int,
    device: str,
    normalize_kind: str = "l2",
    sinkhorn_reg: float = 1e-2,
    sinkhorn_iter: int = 3000,
    max_samples: int = 20000,
    seed_ref: int = 1_000_000_001,
    seed_eval: int = 1_000_000_002,
) -> pd.DataFrame:
    """Evaluate agent on density (p) shifts; return DataFrame with OT metrics vs p."""
    X_ref = collect_edge_embeddings(
        agent, p_ref, hosts, vms, workflows, n, n, device, episodes, seed_base=seed_ref
    )
    if normalize_kind != "none":
        X_ref = normalize(X_ref, normalize_kind)

    rows = []
    for p_eval in p_eval_list:
        X_eval = collect_edge_embeddings(
            agent, p_eval, hosts, vms, workflows, n, n, device, episodes, seed_base=seed_eval
        )
        if normalize_kind != "none":
            X_eval = normalize(X_eval, normalize_kind)

        sink = sinkhorn_distance(X_ref, X_eval, reg=sinkhorn_reg, num_iter=sinkhorn_iter)
        gw_dist, Pi = gw_distance(X_ref, X_eval, metric='cosine', epsilon=5e-3)
        try:
            swd = sliced_wasserstein_distance(X_ref, X_eval, n_projections=128, seed=0)
        except Exception:
            swd = float('nan')

        disp_stats = {}
        if Pi is not None and Pi.size > 0:
            disps = barycentric_displacements(X_ref, X_eval, Pi)
            disp_stats = {
                'disp_mean': float(np.mean(disps)),
                'disp_median': float(np.median(disps)),
                'disp_std': float(np.std(disps)),
            }
        else:
            disp_stats = {'disp_mean': np.nan, 'disp_median': np.nan, 'disp_std': np.nan}

        rows.append({
            'p_ref': p_ref,
            'p_eval': p_eval,
            'sinkhorn': sink,
            'gw': gw_dist,
            'swd': swd,
            **disp_stats
        })

    return pd.DataFrame(rows)


def fit_sqrt_scaling(df: pd.DataFrame, metric: str) -> Dict:
    """Fit metric ~ a + b/sqrt(n) and return coefficients + R^2."""
    try:
        import statsmodels.api as sm
        X = sm.add_constant(df['inv_sqrt_n'].values)
        y = df[metric].values
        model = sm.OLS(y, X).fit()
        return {
            'intercept': model.params[0],
            'slope': model.params[1],
            'r2': model.rsquared,
            'pvalue': model.pvalues[1]
        }
    except Exception as e:
        print(f"Warning: fit failed for {metric}: {e}")
        return {'intercept': np.nan, 'slope': np.nan, 'r2': np.nan, 'pvalue': np.nan}


def plot_size_comparison_agg(agg_df: pd.DataFrame, alphas: List[float], out_dir: Path):
    """Plot mean±std vs 1/sqrt(n) per alpha, colored by lambda."""
    sns.set_style("whitegrid")
    metrics = ['sinkhorn', 'gw', 'swd', 'disp_mean']
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    # Determine x column name depending on aggregation naming
    x_col = 'inv_sqrt_n' if 'inv_sqrt_n' in agg_df.columns else ('inv_sqrt_n_mean' if 'inv_sqrt_n_mean' in agg_df.columns else None)
    if x_col is None:
        raise KeyError("Expected 'inv_sqrt_n' or 'inv_sqrt_n_mean' in aggregated DataFrame")
    for ax, metric in zip(axes, metrics):
        for alpha in alphas:
            sub = agg_df[agg_df['alpha'] == alpha]
            # Plot each lambda with error bars
            for lam, g in sub.groupby('lambda'):
                ax.errorbar(g[x_col], g[f'{metric}_mean'], yerr=g[f'{metric}_std'], fmt='o-', capsize=3, label=f'λ={lam}' if alpha == alphas[0] else None, alpha=0.8)
        ax.set_xlabel('1/√n_eval')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs 1/√n (faceted by α in color legend)')
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    fig.savefig(out_dir / 'size_generalization_comparison.png', dpi=150)
    plt.close(fig)
    print(f"[plot] Saved size comparison: {out_dir / 'size_generalization_comparison.png'}")


def plot_size_comparison_by_alpha(agg_df: pd.DataFrame, alphas: List[float], out_dir: Path):
    """Facet size plots by alpha: rows=alpha, cols=metrics; error bars per lambda."""
    sns.set_style("whitegrid")
    metrics = ['sinkhorn', 'gw', 'swd', 'disp_mean']
    n_rows = len(alphas)
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3.5*n_rows), sharex=True)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for i, alpha in enumerate(alphas):
        sub = agg_df[agg_df['alpha'] == alpha]
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for lam, g in sub.groupby('lambda'):
                ax.errorbar(g['inv_sqrt_n'], g[f'{metric}_mean'], yerr=g[f'{metric}_std'], fmt='o-', capsize=3, label=f'λ={lam}', alpha=0.85)
            ax.set_ylabel(metric if j == 0 else "")
            ax.set_title(f"α={alpha} | {metric}")
            ax.grid(True, alpha=0.3)
            if i == n_rows - 1:
                ax.set_xlabel('1/√n_eval')
    # Single legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    out_path = out_dir / 'size_generalization_comparison_by_alpha.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved size by-alpha comparison: {out_path}")


def plot_density_comparison_agg(agg_df: pd.DataFrame, out_dir: Path):
    """Plot mean±std vs p for each lambda."""
    sns.set_style("whitegrid")
    metrics = ['sinkhorn', 'gw', 'swd', 'disp_mean']
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    for ax, metric in zip(axes, metrics):
        for lam, sub in agg_df.groupby('lambda'):
            ax.errorbar(sub['p_eval'], sub[f'{metric}_mean'], yerr=sub[f'{metric}_std'], fmt='o-', capsize=3, label=f'λ={lam}', alpha=0.8)
        ax.set_xlabel('p_eval')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs p')
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    fig.savefig(out_dir / 'density_generalization_comparison.png', dpi=150)
    plt.close(fig)
    print(f"[plot] Saved density comparison: {out_dir / 'density_generalization_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Compare lambda effect on size/density generalization')
    parser.add_argument('--architecture', type=str, default='hetero', help='Architecture name')
    parser.add_argument('--lambda-models', type=str, required=True, 
                       help='Comma-separated lambda:path pairs, e.g., "0:path1,1e-5:path2"')
    
    # Size sweep
    parser.add_argument('--n-min-base', type=int, default=12, help='Base min tasks')
    parser.add_argument('--n-max-base', type=int, default=24, help='Base max tasks')
    parser.add_argument('--n-ref', type=int, default=24, help='Reference n for OT')
    parser.add_argument('--n-scales', type=str, default='1.0,1.25,1.5', 
                       help='Scales to apply to [n_min_base, n_max_base]')
    
    # Density sweep
    parser.add_argument('--p-base', type=float, default=0.8, help='Base p for size sweep')
    parser.add_argument('--ps', type=str, default='0.3,0.5,0.8', help='p values for density sweep')
    
    # Environment
    parser.add_argument('--hosts', type=int, default=4)
    parser.add_argument('--vms', type=int, default=10)
    parser.add_argument('--workflows', type=int, default=10)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--normalize', type=str, default='l2', choices=['none', 'l2', 'z'])
    parser.add_argument('--out-dir', type=str, default='scheduler/viz_results/decision_boundaries/lambda_comparison')
    # OT/Performance controls
    parser.add_argument('--sinkhorn-reg', type=float, default=1e-2)
    parser.add_argument('--sinkhorn-iter', type=int, default=3000)
    parser.add_argument('--max-samples', type=int, default=20000, help='Max embeddings per set before OT (subsampled)')
    parser.add_argument('--repeats', type=int, default=1, help='Repeat embedding collection with different seeds to get error bars')
    parser.add_argument('--gw-epsilon', type=float, default=5e-3, help='Entropic regularization for GW (higher=faster but smoother)')
    parser.add_argument('--disable-gw', action='store_true', help='Skip GW and barycentric displacement to avoid slowdowns')
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    models = load_lambda_models(args.lambda_models, args.device, args.architecture)
    if not models:
        print("Error: no models loaded")
        return
    
    # Parse scales and ps
    scales = [float(s) for s in args.n_scales.split(',')]
    ps = [float(p) for p in args.ps.split(',')]
    
    # ========== SIZE GENERALIZATION ==========
    print("\n[size] Evaluating size generalization...")
    all_size_rows = []
    size_fit_rows = []
    
    for lam, agent in sorted(models.items()):
        print(f"\n[size] Lambda={lam}")
        for alpha in scales:
            n_min = int(args.n_min_base * alpha)
            n_max = int(args.n_max_base * alpha)
            n_eval_list = list(range(n_min, n_max + 1, 4))  # Sample every 4 tasks
            
            print(f"  Scale α={alpha}: n_range=[{n_min},{n_max}], n_ref={args.n_ref}")
            # Repeats for error bars
            for r in range(args.repeats):
                seed_ref = 1_000_000_001 + 1000*r
                seed_eval = 1_000_000_002 + 1000*r
                df = evaluate_size_generalization(
                    agent, n_min, n_max, args.n_ref, n_eval_list,
                    args.p_base, args.hosts, args.vms, args.workflows,
                    args.episodes, args.device, args.normalize,
                    args.sinkhorn_reg, args.sinkhorn_iter, args.max_samples,
                    seed_ref, seed_eval, args.gw_epsilon, args.disable_gw
                )
                df['lambda'] = lam
                df['alpha'] = alpha
                df['repeat'] = r
                all_size_rows.append(df)
            
            # Fit sqrt scaling for each metric
            df_fit = pd.concat([d for d in all_size_rows if ('alpha' in d and isinstance(d, pd.DataFrame))]) if False else None
            # Using the last computed df for fitting at this alpha (still indicative)
            for metric in ['sinkhorn', 'gw', 'swd', 'disp_mean']:
                fit_res = fit_sqrt_scaling(df, metric)
                size_fit_rows.append({
                    'lambda': lam,
                    'alpha': alpha,
                    'metric': metric,
                    **fit_res
                })
    
    all_size_df = pd.concat(all_size_rows, ignore_index=True)
    size_fit_df = pd.DataFrame(size_fit_rows)
    
    # Save CSVs
    all_size_df.to_csv(out_dir / 'size_generalization_raw.csv', index=False)
    size_fit_df.to_csv(out_dir / 'size_generalization_fits.csv', index=False)
    print(f"\n[size] Saved raw data: {out_dir / 'size_generalization_raw.csv'}")
    print(f"[size] Saved fits: {out_dir / 'size_generalization_fits.csv'}")
    
    # Print fit summary
    print("\n[size] 1/√n scaling fits (slope, R²):")
    print(size_fit_df[['lambda', 'alpha', 'metric', 'slope', 'r2']].to_string(index=False))
    
    # Aggregate for error bars
    size_agg = (all_size_df
        .groupby(['lambda','alpha','n_eval'], as_index=False)
        .agg({
            'inv_sqrt_n':'mean',
            'sinkhorn':['mean','std'],
            'gw':['mean','std'],
            'swd':['mean','std'],
            'disp_mean':['mean','std'],
        }))
    # Flatten columns
    size_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in size_agg.columns.values]
    # Plot
    plot_size_comparison_agg(size_agg, scales, out_dir)
    
    # ========== DENSITY GENERALIZATION ==========
    print("\n[density] Evaluating density generalization...")
    all_density_rows = []
    
    for lam, agent in sorted(models.items()):
        print(f"\n[density] Lambda={lam}")
        for r in range(args.repeats):
            seed_ref = 1_000_000_001 + 1000*r
            seed_eval = 1_000_000_002 + 1000*r
            df = evaluate_density_generalization(
                agent, args.n_ref, args.p_base, ps,
                args.hosts, args.vms, args.workflows,
                args.episodes, args.device, args.normalize,
                args.sinkhorn_reg, args.sinkhorn_iter, args.max_samples,
                seed_ref, seed_eval, args.gw_epsilon, args.disable_gw
            )
            df['lambda'] = lam
            df['repeat'] = r
            all_density_rows.append(df)
    
    all_density_df = pd.concat(all_density_rows, ignore_index=True)
    all_density_df.to_csv(out_dir / 'density_generalization.csv', index=False)
    print(f"\n[density] Saved: {out_dir / 'density_generalization.csv'}")
    
    # Aggregate for error bars
    density_agg = (all_density_df
        .groupby(['lambda','p_eval'], as_index=False)
        .agg({
            'sinkhorn':['mean','std'],
            'gw':['mean','std'],
            'swd':['mean','std'],
            'disp_mean':['mean','std'],
        }))
    density_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in density_agg.columns.values]
    # Plot
    plot_density_comparison_agg(density_agg, out_dir)
    
    print(f"\n[done] All results in: {out_dir}")


if __name__ == '__main__':
    main()
