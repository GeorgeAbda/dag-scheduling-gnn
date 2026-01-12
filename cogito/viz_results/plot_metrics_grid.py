#!/usr/bin/env python3
"""
Create a grid of Deep RL–style plots comparing three reward settings across multiple metrics.

Scans the csv/ directory for files named:
  <metric>_energy.csv
  <metric>_makespan.csv
  <metric>_energy+makespan.csv
Each file must have columns: Wall time, Step, Value

Usage:
  python scheduler/viz_results/plot_metrics_grid.py \
      --csv-dir csv \
      --out csv/metrics_grid.png \
      --window 7 \
      --xaxis Episode

"""
from __future__ import annotations

import argparse
import glob
import math
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

# Configure matplotlib for publication quality
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['patch.linewidth'] = 0.5

REWARD_KEYS = ["energy", "makespan", "energy+makespan"]
REWARD_LABELS = {
    "energy": "Energy",
    "makespan": "Makespan",
    "energy+makespan": "Energy + Makespan",
}
REWARD_COLORS = {
    # Pastel variants for a softer palette
    "energy": "#aec7e8",          # pastel blue
    "makespan": "#ff9896",        # pastel red
    "energy+makespan": "#98df8a", # pastel green
}


def _read_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"Wall time", "Step", "Value"}
    if not expected.issubset(df.columns):
        raise ValueError(f"{path} missing required columns {expected}")
    df = df.sort_values("Step").reset_index(drop=True)
    steps = df["Step"].to_numpy()
    if len(steps) >= 2:
        diffs = np.diff(steps)
        base = int(reduce(gcd, diffs)) if diffs.size > 0 else int(diffs[0])
        if base <= 0:
            base = int(diffs[0]) if diffs.size > 0 else 1
        df["Episode"] = (df["Step"] // base).astype(int)
        if df["Episode"].min() == 0:
            df["Episode"] = df["Episode"] + 1
    else:
        df["Episode"] = 1
    return df


def _smooth_with_band(df: pd.DataFrame, ycol: str = "Value", window: int = 7) -> Tuple[pd.Series, pd.Series, pd.Series]:
    smooth = df[ycol].rolling(window=window, center=True, min_periods=max(1, window // 2)).mean()
    std = df[ycol].rolling(window=window, center=True, min_periods=max(1, window // 2)).std().fillna(0.0)
    smooth = smooth.fillna(df[ycol])
    lower = (smooth - std).fillna(df[ycol])
    upper = (smooth + std).fillna(df[ycol])
    return smooth, lower, upper


essential_pretty = {
    "idle_energy": "Idle Energy",
    "active_energy": "Active Energy",
    "total_energy": "Total Energy",
    "makespan": "Makespan",
}

# Specific y-labels per metric (fallback to generic "Value")
metric_ylabels = {
    "idle_energy": "Energy (J)",
    "active_energy": "Energy (J)",
    "total_energy": "Energy (J)",
    "makespan": "Time",
}



def discover_metrics(csv_dir: Path) -> List[str]:
    # Restrict to the 5 requested metrics in this specific order
    desired_order = [
        "idle_energy",
        "active_energy",
        "total_energy",
        "makespan",
    ]
    metrics: List[str] = []
    for m in desired_order:
        present = any((csv_dir / f"{m}_{rk}.csv").exists() for rk in REWARD_KEYS)
        if present:
            metrics.append(m)
    return metrics



def plot_grid(csv_dir: Path, out_path: Path, window: int, xaxis: str) -> None:
    # Use clean style for publication
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    metrics = discover_metrics(csv_dir)
    if not metrics:
        raise SystemExit(f"No metric files found in {csv_dir}")

    n = len(metrics)
    # Arrange as 2 columns so we have two panels on top and two on bottom for n=4
    cols = 2
    rows = math.ceil(n / cols)
    # Publication-quality figure size (IEEE column width ~3.5", double column ~7")
    fig, axes = plt.subplots(rows, cols, figsize=(7.5, 6), squeeze=False, dpi=300)
    fig.patch.set_facecolor('white')

    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        title = essential_pretty.get(metric, metric.replace("_", " ").title())
        for rk in REWARD_KEYS:
            path = csv_dir / f"{metric}_{rk}.csv"
            if not path.exists():
                continue
            df = _read_series(path)
            smooth, low, up = _smooth_with_band(df, ycol="Value", window=window)
            ax.plot(df[xaxis], smooth, label=REWARD_LABELS[rk], color=REWARD_COLORS[rk], 
                   linewidth=2.5, alpha=0.9)
            ax.fill_between(df[xaxis], low, up, color=REWARD_COLORS[rk], alpha=0.25, linewidth=0)
        
        ax.set_title(title, fontweight='bold', pad=12)
        ax.set_xlabel(xaxis, fontweight='normal')
        ax.set_ylabel(metric_ylabels.get(metric, "Value"), fontweight='normal')
        
        # Clean grid and spines
        ax.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        
        # Improve tick formatting
        ax.tick_params(axis='both', which='major', labelsize=10, length=4, width=1)
        ax.margins(x=0.02, y=0.05)

    # Hide any unused axes if grid has extra cells
    total_cells = rows * cols
    for k in range(n, total_cells):
        r, c = divmod(k, cols)
        axes[r][c].axis('off')

    # Publication-quality legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(REWARD_KEYS), 
                  frameon=True, fancybox=False, shadow=False, 
                  bbox_to_anchor=(0.5, 0.98), fontsize=11,
                  columnspacing=1.5, handlelength=2.0)
    
    # Tight layout with proper spacing for publication
    plt.tight_layout(rect=(0, 0, 1, 0.92), pad=2.0, h_pad=3.0, w_pad=2.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Saved metrics grid to: {out_path}")



def main() -> None:
    p = argparse.ArgumentParser(description="Plot grid of metrics across reward variants.")
    p.add_argument("--csv-dir", type=Path, default=Path("csv"))
    p.add_argument("--out", type=Path, default=Path("csv/metrics_grid.png"))
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--xaxis", type=str, choices=["Episode", "Step"], default="Episode")
    args = p.parse_args()

    plot_grid(csv_dir=args.csv_dir, out_path=args.out, window=args.window, xaxis=args.xaxis)


if __name__ == "__main__":
    main()
