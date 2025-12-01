#!/usr/bin/env python3
"""
Create plots comparing models (GNN, NoGlobal, MLP, HeteroGNN, Hetero-NoGlobal) across different metrics.

Scans the csv_ablation/ directory for files with patterns:
  <metric>_gnp_p0{3|8}_baseline_tb_baseline*.csv (GNN)
  <metric>_gnp_p0{3|8}_noglobal_tb_no_global_actor*.csv (NoGlobal)
  <metric>_gnp_p0{3|8}_mlp_tb_mlp_only*.csv (MLP)
  <metric>_gnp_p0{3|8}_hetero_tb_hetero*.csv or <metric>_hetero_tb_hetero*.csv (HeteroGNN)
  <metric>_*_hetero_noglobal_tb_hetero_noglobal*.csv (Hetero-NoGlobal)

Each file must have columns: Wall time, Step, Value

Usage:
  python scheduler/viz_results/plot_models_comparison.py \
      --csv-dir csv_ablation \
      --metric active_energy \
      --out csv_ablation/active_energy_comparison.png \
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

MODEL_KEYS = ["baseline", "noglobal", "mlp", "hetero", "hetero_noglobal"]
MODEL_LABELS = {
    "baseline": "HomoGNN",
    "noglobal": "NoGlobal",
    "mlp": "MLP Only",
    "hetero": "HeteroGNN",
    "hetero_noglobal": "Hetero-NoGlobal",
}
MODEL_COLORS = {
    "baseline": "#1f77b4",        # blue
    "noglobal": "#ff7f0e",        # orange
    "mlp": "#2ca02c",             # green
    "hetero": "#9467bd",          # purple
    "hetero_noglobal": "#d62728",  # red
}

# Metric labels and units
METRIC_LABELS = {
    "active_energy": "Active Energy",
    "idle_energy": "Idle Energy", 
    "makespan": "Makespan",
}

METRIC_YLABELS = {
    "active_energy": "Energy (J)",
    "idle_energy": "Energy (J)",
    "makespan": "Time",
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


def find_model_file(csv_dir: Path, metric: str, model: str) -> Optional[Path]:
    """Find the CSV file for a given metric and model."""
    # Try both p03 and p08 patterns (search recursively under csv_dir)
    patterns = []
    if model == "baseline":
        patterns = [
            f"{metric}_gnp_p08_baseline_tb_baseline*.csv",  # p08 pattern first
            f"{metric}_gnp_p03_baseline_tb_baseline*.csv"   # fallback to p03
        ]
    elif model == "noglobal":
        patterns = [
            f"{metric}_gnp_p08_noglobal_tb_no_global_actor*.csv",
            f"{metric}_gnp_p03_noglobal_tb_no_global_actor*.csv"
        ]
    elif model == "mlp":
        patterns = [
            f"{metric}_gnp_p08_mlp_tb_mlp_only*.csv",
            f"{metric}_gnp_p03_mlp_tb_mlp_only*.csv"
        ]
    elif model == "hetero":
        # Heterogeneous GNN variants (allow naming variations)
        patterns = [
            f"{metric}_gnp_p08_hetero_tb_hetero*.csv",
            f"{metric}_gnp_p03_hetero_tb_hetero*.csv",
            f"{metric}_hetero_tb_hetero*.csv",
            # Additional flexible fallbacks (e.g., 'hetero_08_samearchitecture')
            f"{metric}_hetero_*_tb_hetero*.csv",
            f"{metric}_*_hetero_tb_hetero*.csv",
        ]
    elif model == "hetero_noglobal":
        # Hetero without global actor embedding (allow naming variations)
        patterns = [
            f"{metric}_gnp_p08_hetero_noglobal_tb_hetero_noglobal*.csv",
            f"{metric}_gnp_p03_hetero_noglobal_tb_hetero_noglobal*.csv",
            f"{metric}_gnn_ablation_hetero_noglobal_tb_hetero_noglobal*.csv",
            f"{metric}_hetero_noglobal_tb_hetero_noglobal*.csv",
        ]
    else:
        return None
    
    for pattern in patterns:
        files = sorted(csv_dir.rglob(pattern))
        if files:
            return files[0]  # Return first match
    return None


def plot_metric_comparison(csv_dir: Path, metric: str, out_path: Path, window: int, xaxis: str,
                           models: Optional[List[str]] = None) -> None:
    """Plot comparison of one metric across the configured models."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor('white')
    
    title = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    
    models = MODEL_KEYS if models is None else models
    for model in models:
        path = find_model_file(csv_dir, metric, model)
        if not path:
            print(f"Warning: No file found for {metric} + {model}")
            continue
            
        print(f"Loading {model}: {path}")
        df = _read_series(path)
        smooth, low, up = _smooth_with_band(df, ycol="Value", window=window)
        
        ax.plot(df[xaxis], smooth, label=MODEL_LABELS[model], color=MODEL_COLORS[model], 
               linewidth=2.5, alpha=0.9)
        ax.fill_between(df[xaxis], low, up, color=MODEL_COLORS[model], alpha=0.25, linewidth=0)
    
    ax.set_title(title, fontweight='bold', pad=12, fontsize=14)
    ax.set_xlabel(xaxis, fontweight='normal')
    ax.set_ylabel(METRIC_YLABELS.get(metric, "Value"), fontweight='normal')
    
    # Clean grid and spines
    ax.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Improve tick formatting
    ax.tick_params(axis='both', which='major', labelsize=10, length=4, width=1)
    ax.margins(x=0.02, y=0.05)
    
    # Legend
    ax.legend(loc="best", frameon=True, fancybox=False, shadow=False, fontsize=11)
    
    plt.tight_layout(pad=2.0)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {metric} comparison to: {out_path}")
    plt.close()


def calculate_total_energy(csv_dir: Path, out_path: Path, window: int, xaxis: str,
                           models: Optional[List[str]] = None) -> None:
    """Create total energy plot by combining active + idle energy."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor('white')
    
    models = MODEL_KEYS if models is None else models
    for model in models:
        active_path = find_model_file(csv_dir, "active_energy", model)
        idle_path = find_model_file(csv_dir, "idle_energy", model)
        
        if not active_path or not idle_path:
            print(f"Warning: Missing active or idle energy file for {model}")
            continue
            
        print(f"Loading {model}: {active_path} + {idle_path}")
        
        # Load both files
        active_df = _read_series(active_path)
        idle_df = _read_series(idle_path)
        
        # Align by Step and sum
        merged = pd.merge(active_df, idle_df, on="Step", suffixes=("_active", "_idle"))
        merged["Value"] = merged["Value_active"] + merged["Value_idle"]
        merged["Episode"] = merged["Episode_active"]  # Use active's episode calculation
        
        smooth, low, up = _smooth_with_band(merged, ycol="Value", window=window)
        
        ax.plot(merged[xaxis], smooth, label=MODEL_LABELS[model], color=MODEL_COLORS[model], 
               linewidth=2.5, alpha=0.9)
        ax.fill_between(merged[xaxis], low, up, color=MODEL_COLORS[model], alpha=0.25, linewidth=0)
    
    ax.set_title("Total Energy (Active + Idle)", fontweight='bold', pad=12, fontsize=14)
    ax.set_xlabel(xaxis, fontweight='normal')
    ax.set_ylabel("Energy (J)", fontweight='normal')
    
    # Clean grid and spines
    ax.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Improve tick formatting
    ax.tick_params(axis='both', which='major', labelsize=10, length=4, width=1)
    ax.margins(x=0.02, y=0.05)
    
    # Legend
    ax.legend(loc="best", frameon=True, fancybox=False, shadow=False, fontsize=11)
    
    plt.tight_layout(pad=2.0)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved total energy comparison to: {out_path}")
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Plot model comparison for a specific metric.")
    p.add_argument("--csv-dir", type=Path, default=Path("csv_ablation"))
    p.add_argument("--metric", type=str, required=True, 
                   choices=["active_energy", "idle_energy", "makespan", "total_energy"],
                   help="Metric to plot")
    p.add_argument("--out", type=Path, help="Output path (auto-generated if not provided)")
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--xaxis", type=str, choices=["Episode", "Step"], default="Episode")
    p.add_argument("--exclude-models", nargs="+", choices=MODEL_KEYS, default=[],
                   help="List of model keys to exclude from plotting. Options: baseline, noglobal, mlp, hetero, hetero_noglobal")
    args = p.parse_args()

    # Optionally exclude selected models from plotting (without mutating globals)
    models_to_plot = MODEL_KEYS
    if args.exclude_models:
        models_to_plot = [m for m in MODEL_KEYS if m not in set(args.exclude_models)]
        print(f"Excluding models: {', '.join(args.exclude_models)}")
        print(f"Will plot models: {', '.join(models_to_plot)}")

    if not args.out:
        args.out = args.csv_dir / f"{args.metric}_models_comparison.png"
    
    if args.metric == "total_energy":
        calculate_total_energy(args.csv_dir, args.out, args.window, args.xaxis, models=models_to_plot)
    else:
        plot_metric_comparison(args.csv_dir, args.metric, args.out, args.window, args.xaxis, models=models_to_plot)


if __name__ == "__main__":
    main()
