#!/usr/bin/env python3
"""
Create Deep RL–style training curves from CSV logs in csv/.

- Reads CSVs with columns: Wall time, Step, Value
- Applies smoothing (rolling mean) and shaded region (rolling std)
- Plots multiple algorithms on one figure with publication-style aesthetics

Usage:
    python scheduler/viz_results/plot_deeprl_style.py \
        --energy_csv csv/total_energy_energy.csv \
        --makespan_csv csv/total_energy_makespan.csv \
        --both_csv csv/total_energy_energy+makespan.csv \
        --out csv/energy_training_comparison.png

If no arguments are provided, the script will try to auto-discover the three files by pattern.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, Optional, Tuple
from functools import reduce
from math import gcd

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _read_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure expected columns
    expected = {"Wall time", "Step", "Value"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"CSV {path} must contain columns {expected}, got {df.columns.tolist()}")
    # Sort by step and infer episode index from the dominant step increment
    df = df.sort_values("Step").reset_index(drop=True)
    steps = df["Step"].to_numpy()
    if len(steps) >= 2:
        diffs = np.diff(steps)
        # Use GCD of all diffs as the base increment; fall back to first diff if GCD=0
        base = int(reduce(gcd, diffs)) if diffs.size > 0 else int(diffs[0])
        if base <= 0:
            base = int(diffs[0]) if diffs.size > 0 else 1
        # Derive episode as 1-based integer steps/base
        df["Episode"] = (df["Step"] // base).astype(int)
        # Ensure starts at 1
        min_ep = df["Episode"].min()
        if min_ep == 0:
            df["Episode"] = df["Episode"] + 1
    else:
        # Single row – default to episode 1
        df["Episode"] = 1
    return df


def _smooth_with_band(df: pd.DataFrame, ycol: str = "Value", window: int = 5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    # Rolling mean and std (centered for nicer alignment)
    smooth = df[ycol].rolling(window=window, center=True, min_periods=max(1, window // 2)).mean()
    std = df[ycol].rolling(window=window, center=True, min_periods=max(1, window // 2)).std()
    # Fallback where std may be NaN at the edges
    std = std.fillna(0.0)
    lower = smooth - std
    upper = smooth + std
    # For edge values where mean is NaN, fall back to original values
    smooth = smooth.fillna(df[ycol])
    lower = lower.fillna(df[ycol])
    upper = upper.fillna(df[ycol])
    return smooth, lower, upper


def _autodiscover(default_dir: Path) -> Dict[str, Optional[Path]]:
    patterns = {
        "energy": str(default_dir / "*_energy.csv"),
        "makespan": str(default_dir / "*_makespan.csv"),
        "both": str(default_dir / "*_energy_makespan.csv"),
    }
    found: Dict[str, Optional[Path]] = {k: None for k in patterns}
    for key, pat in patterns.items():
        matches = sorted(glob.glob(pat))
        if matches:
            found[key] = Path(matches[-1])  # pick the most recent match lexicographically
    return found


def plot_curves(
    paths: Dict[str, Path],
    out_path: Path,
    title: str = "Training Comparison",
    x_axis: str = "Episode",
    window: int = 7,
) -> None:
    # Style similar to DRL papers
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)

    # Color palette for three curves
    palette = {
        "Energy": "#1f77b4",      # blue
        "Makespan": "#2ca02c",    # green
        "Energy + Makespan": "#ff7f0e",  # orange
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for label_key, label in [("energy", "Energy"), ("makespan", "Makespan"), ("both", "Energy + Makespan")]:
        path = paths.get(label_key)
        if path is None:
            continue
        df = _read_series(path)
        smooth, low, up = _smooth_with_band(df, ycol="Value", window=window)

        color = palette[label]
        ax.plot(df[x_axis], smooth, label=label, color=color, linewidth=2.2)
        ax.fill_between(df[x_axis], low, up, color=color, alpha=0.18, linewidth=0)

    ax.set_xlabel(x_axis)
    ax.set_ylabel("Total Energy")
    ax.set_title(title)
    ax.legend(frameon=True)

    # Ticks and limits
    ax.margins(x=0.02)
    ax.grid(True, which="major", linestyle="--", alpha=0.35)

    # Tight layout and save
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Deep RL–style curves from training CSVs.")
    parser.add_argument("--energy_csv", type=Path, default=None, help="Path to *_energy.csv")
    parser.add_argument("--makespan_csv", type=Path, default=None, help="Path to *_makespan.csv")
    parser.add_argument("--both_csv", type=Path, default=None, help="Path to *_energy_makespan.csv")
    parser.add_argument("--out", type=Path, default=Path("csv/energy_training_comparison.png"), help="Output image path")
    parser.add_argument("--title", type=str, default="Energy Across Training", help="Plot title")
    parser.add_argument("--window", type=int, default=7, help="Smoothing window size (rolling)")
    parser.add_argument("--xaxis", type=str, choices=["Episode", "Step"], default="Episode", help="X-axis choice")
    args = parser.parse_args()

    default_dir = Path("csv")
    discovered = _autodiscover(default_dir)

    paths: Dict[str, Path] = {}
    # Priority: CLI arg > autodiscover
    if args.energy_csv is not None:
        paths["energy"] = args.energy_csv
    elif discovered["energy"] is not None:
        paths["energy"] = discovered["energy"]  # type: ignore[assignment]

    if args.makespan_csv is not None:
        paths["makespan"] = args.makespan_csv
    elif discovered["makespan"] is not None:
        paths["makespan"] = discovered["makespan"]  # type: ignore[assignment]

    if args.both_csv is not None:
        paths["both"] = args.both_csv
    elif discovered["both"] is not None:
        paths["both"] = discovered["both"]  # type: ignore[assignment]

    if not paths:
        raise SystemExit("No CSV files found. Provide paths via CLI or put files in csv/.")

    plot_curves(paths=paths, out_path=args.out, title=args.title, x_axis=args.xaxis, window=args.window)


if __name__ == "__main__":
    main()
