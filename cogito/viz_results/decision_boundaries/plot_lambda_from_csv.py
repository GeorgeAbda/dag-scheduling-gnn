#!/usr/bin/env python3
"""
Plot lambda generalization results from existing CSVs without recomputing OT.

Usage examples:
  python -m scheduler.viz_results.decision_boundaries.plot_lambda_from_csv \
    --size-raw scheduler/viz_results/decision_boundaries/lambda_comparison_tuned_v2/size_generalization_raw.csv \
    --size-fits scheduler/viz_results/decision_boundaries/lambda_comparison_tuned_v2/size_generalization_fits.csv \
    --out-dir scheduler/viz_results/decision_boundaries/lambda_comparison_tuned_v2

  # Optionally include density CSV to re-plot density figure
  python -m scheduler.viz_results.decision_boundaries.plot_lambda_from_csv \
    --size-raw .../size_generalization_raw.csv \
    --density-raw .../density_generalization.csv \
    --out-dir ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_size_agg(size_raw: pd.DataFrame, out_dir: Path, scales: List[float] | None = None) -> None:
    # Ensure expected columns
    needed_cols = {'lambda', 'alpha', 'n_eval', 'inv_sqrt_n'}
    if not needed_cols.issubset(size_raw.columns):
        raise ValueError(f"size_raw missing columns: {needed_cols - set(size_raw.columns)}")

    # Determine which metrics exist
    metric_candidates = ['sinkhorn', 'gw', 'swd', 'disp_mean']
    metrics = [m for m in metric_candidates if m in size_raw.columns]
    if not metrics:
        raise ValueError("No known metric columns found in size_raw.csv")

    # Aggregate mean/std across repeats/episodes per (lambda, alpha, n_eval)
    agg = (
        size_raw
        .groupby(['lambda','alpha','n_eval'], as_index=False)
        .agg({
            'inv_sqrt_n':'mean',
            **{m:['mean','std'] for m in metrics},
        })
    )
    # Flatten MultiIndex columns
    agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in agg.columns.values]

    # Sort for nicer plotting
    agg = agg.sort_values(['alpha','lambda','n_eval']).reset_index(drop=True)

    # Use provided scales if any; else derive unique alphas
    alphas = scales if scales is not None else sorted(agg['alpha'].unique())

    # Plot aggregated (all alphas overlaid, legend by lambda)
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    x_col = 'inv_sqrt_n' if 'inv_sqrt_n' in agg.columns else 'inv_sqrt_n_mean'
    for ax, metric in zip(axes, metrics):
        for alpha in alphas:
            sub = agg[agg['alpha'] == alpha]
            for lam, g in sub.groupby('lambda'):
                mean_col = f'{metric}_mean'
                std_col = f'{metric}_std'
                if mean_col not in g.columns:
                    continue
                yerr = g[std_col] if std_col in g.columns else None
                ax.errorbar(g[x_col], g[mean_col], yerr=yerr, fmt='o-', capsize=3, label=f'λ={lam}', alpha=0.85)
        ax.set_xlabel('1/√n_eval')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs 1/√n (mean±std)')
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    out_path = out_dir / 'size_generalization_comparison.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")

    # Faceted by alpha (rows) with error bars
    n_rows = len(alphas)
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3.5*n_rows), sharex=True)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for i, alpha in enumerate(alphas):
        sub = agg[agg['alpha'] == alpha]
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for lam, g in sub.groupby('lambda'):
                mean_col = f'{metric}_mean'
                std_col = f'{metric}_std'
                if mean_col not in g.columns:
                    continue
                yerr = g[std_col] if std_col in g.columns else None
                ax.errorbar(g[x_col], g[mean_col], yerr=yerr, fmt='o-', capsize=3, label=f'λ={lam}', alpha=0.85)
            ax.set_ylabel(metric if j == 0 else '')
            ax.set_title(f'α={alpha} | {metric}')
            ax.grid(True, alpha=0.3)
            if i == n_rows - 1:
                ax.set_xlabel('1/√n_eval')
    handles, labels = axes[0,0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    out_path = out_dir / 'size_generalization_comparison_by_alpha.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")


def plot_density_agg(density_raw: pd.DataFrame, out_dir: Path) -> None:
    if not {'lambda','p_eval'}.issubset(density_raw.columns):
        raise ValueError("density_raw missing required columns: {'lambda','p_eval'}")
    metrics = [m for m in ['sinkhorn','gw','swd','disp_mean'] if m in density_raw.columns]
    if not metrics:
        print("[warn] No known metric columns in density CSV; skipping density plot.")
        return

    agg = (
        density_raw
        .groupby(['lambda','p_eval'], as_index=False)
        .agg({**{m:['mean','std'] for m in metrics}})
    )
    agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in agg.columns.values]

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        for lam, g in agg.groupby('lambda'):
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            yerr = g[std_col] if std_col in g.columns else None
            ax.errorbar(g['p_eval'], g[mean_col], yerr=yerr, fmt='o-', capsize=3, label=f'λ={lam}', alpha=0.85)
        ax.set_xlabel('p_eval')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs p (mean±std)')
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    out_path = out_dir / 'density_generalization_comparison.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description='Plot lambda generalization from CSVs (no recompute)')
    p.add_argument('--size-raw', type=str, required=True, help='Path to size_generalization_raw.csv')
    p.add_argument('--size-fits', type=str, default='', help='Optional: size_generalization_fits.csv (unused for plotting)')
    p.add_argument('--density-raw', type=str, default='', help='Optional: density_generalization.csv to replot density figure')
    p.add_argument('--out-dir', type=str, required=True, help='Output directory for figures')
    p.add_argument('--alphas', type=str, default='', help='Optional: comma-separated alpha list to order rows')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    size_raw = pd.read_csv(args.size_raw)
    alphas = [float(a) for a in args.alphas.split(',')] if args.alphas else None
    plot_size_agg(size_raw, out_dir, alphas)

    if args.density_raw:
        density_raw = pd.read_csv(args.density_raw)
        plot_density_agg(density_raw, out_dir)


if __name__ == '__main__':
    main()
