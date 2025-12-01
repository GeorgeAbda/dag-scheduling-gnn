#!/usr/bin/env python3
"""
Aggregate alpha_sweep results across different GNP p values and plot
metrics (makespan, active/idle/host total energy) vs p, one curve per model.

Usage Example:
  python scripts/plot_metrics_vs_p.py \
    --inputs \
      0.0:logs/alpha_sweep_p0.0/tasks_0.00/combined_summary.csv \
      0.3:logs/alpha_sweep_p0.3/tasks_0.00/combined_summary.csv \
      0.5:logs/alpha_sweep_p0.5/tasks_0.00/combined_summary.csv \
      0.7:logs/alpha_sweep_p0.7/tasks_0.00/combined_summary.csv \
      0.8:logs/alpha_sweep_p0.8/tasks_0.00/combined_summary.csv \
    --out-dir csv_ablation/gnp_sweep

Notes:
- This expects you ran scripts/eval_models_alpha_sweep.py with --alpha_list 0.0 and --alpha_prime_list 0.0
  so that each out_dir root contains tasks_0.00/combined_summary.csv, with a row per (model, alpha=0.0).
- Model labels are taken from the CSV 'model' column (whatever you passed in --models Label:/path...)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_series(inputs: list[str]):
    """inputs is a list of strings like '0.3:/path/to/combined_summary.csv'.
    Returns: sorted_p_values, models, metrics_by_model = {model: {metric_key: [values aligned to p]}}
    """
    # metrics present in combined_summary.csv written by eval_models_alpha_sweep.py
    metric_keys = ['makespan_mean', 'total_energy_active_mean', 'total_energy_idle_mean', 'total_energy_host_mean']

    # parse p->file mapping
    pairs = []
    for spec in inputs:
        if ':' not in spec:
            raise SystemExit(f"Invalid --inputs entry '{spec}'. Expected 'p_value:/path/to/combined_summary.csv'.")
        p_str, path = spec.split(':', 1)
        try:
            p_val = float(p_str)
        except ValueError:
            raise SystemExit(f"Invalid p value '{p_str}' in --inputs entry '{spec}'.")
        pairs.append((p_val, Path(path)))
    pairs.sort(key=lambda x: x[0])

    # Discover model labels from first file
    models = []
    for _p, fpath in pairs:
        if not fpath.exists():
            raise SystemExit(f"Missing file: {fpath}")
    with pairs[0][1].open('r', newline='') as f:
        r = csv.DictReader(f)
        seen = set()
        for row in r:
            lab = row.get('model')
            if lab and lab not in seen:
                models.append(lab)
                seen.add(lab)
        if not models:
            raise SystemExit(f"No models found in {pairs[0][1]}")

    # Initialize series
    metrics_by_model: dict[str, dict[str, list[float]]] = {lab: {k: [] for k in metric_keys} for lab in models}

    # For each p, read file and load alpha=0.0 rows
    for p, fpath in pairs:
        rows = []
        with fpath.open('r', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
        # filter alpha==0 row per model; if multiple, take first; if none, push nan
        for lab in models:
            row = next((rw for rw in rows if rw.get('model') == lab and float(rw.get('alpha', 'nan')) == 0.0), None)
            for k in metric_keys:
                val = float(row.get(k)) if row and (k in row) else float('nan')
                metrics_by_model[lab][k].append(val)

    sorted_p = [p for p, _ in pairs]
    return sorted_p, models, metrics_by_model


def plot_vs_p(p_values: list[float], models: list[str], series: dict[str, dict[str, list[float]]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ('total_energy_host_mean', 'Total Energy (Host = Active+Idle)'),
        ('total_energy_active_mean', 'Active Energy'),
        ('total_energy_idle_mean', 'Idle Energy'),
        ('makespan_mean', 'Makespan'),
    ]
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for j, (key, label) in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(7.5, 5), dpi=140)
        for i, lab in enumerate(models):
            ys = series.get(lab, {}).get(key, [])
            if len(ys) != len(p_values):
                continue
            ax.plot(p_values, ys, marker='o', color=colors[i % len(colors)], label=lab)
        ax.set_xlabel('GNP p')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs GNP p')
        ax.legend(frameon=False)
        fig.tight_layout()
        fname = {
            'makespan_mean': 'makespan_mean_vs_p.png',
            'total_energy_active_mean': 'total_energy_active_mean_vs_p.png',
            'total_energy_idle_mean': 'total_energy_idle_mean_vs_p.png',
            'total_energy_host_mean': 'total_energy_host_mean_vs_p.png',
        }.get(key, f'{key}_vs_p.png')
        fig.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Aggregate alpha_sweep combined summaries across different p values and plot metrics vs p')
    ap.add_argument('--inputs', nargs='+', required=True, help="List like '0.3:/path/to/combined_summary.csv' ...")
    ap.add_argument('--out-dir', type=str, default='csv_ablation/gnp_sweep')
    args = ap.parse_args()

    p_values, models, series = load_series(args.inputs)
    plot_vs_p(p_values, models, series, Path(args.out_dir))
    print(f"Wrote plots to: {args.out_dir}")


if __name__ == '__main__':
    main()
