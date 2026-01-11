#!/usr/bin/env python3
"""
Compare objective correlation between NAL and AL cases
for long-CP and wide specialists using hetero_mean_pareto.csv.

Usage:
    python scripts/analyze_objective_correlation_cases.py
"""

import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_pareto_csv(csv_path: Path) -> pd.DataFrame:
    """Load Pareto CSV and return DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Pareto CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["makespan", "energy"])
    return df


def compute_correlation_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Compute correlation statistics between makespan and energy."""
    makespan = df["makespan"].values
    energy = df["energy"].values

    if len(makespan) < 2:
        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "n_points": len(makespan),
        }

    r, p = pearsonr(makespan, energy)
    return {"pearson_r": r, "pearson_p": p, "n_points": len(makespan)}


def plot_case_comparison(
    longcp_nal: pd.DataFrame,
    longcp_al: pd.DataFrame,
    wide_nal: pd.DataFrame,
    wide_al: pd.DataFrame,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Plot NAL vs AL comparison for long-CP and wide specialists (1x2 figure)."""

    # Publication-quality settings similar to analyze_objective_correlation_training
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "patch.linewidth": 0.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "text.usetex": False,
        }
    )

    sns.set_style("white")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Colors
    color_nal = "#2E86AB"  # blue
    color_al = "#A23B72"  # magenta/purple

    # ---- Long-CP panel ----
    ax = axes[0]

    ax.scatter(
        longcp_nal["makespan"],
        longcp_nal["energy"],
        alpha=0.8,
        s=40,
        c=color_nal,
        edgecolors="white",
        linewidths=0.8,
        marker="o",
        label="NAL",
        zorder=3,
    )
    ax.scatter(
        longcp_al["makespan"],
        longcp_al["energy"],
        alpha=0.8,
        s=40,
        c=color_al,
        edgecolors="white",
        linewidths=0.8,
        marker="s",
        label="AL",
        zorder=3,
    )

    # Correlation stats
    longcp_nal_stats = compute_correlation_stats(longcp_nal)
    longcp_al_stats = compute_correlation_stats(longcp_al)

    textstr = (
        f"NAL: $r = {longcp_nal_stats['pearson_r']:.3f}$\n"
        f"AL:  $r = {longcp_al_stats['pearson_r']:.3f}$"
    )
    props = {
        "boxstyle": "round,pad=0.5",
        "facecolor": "white",
        "edgecolor": "gray",
        "alpha": 0.9,
        "linewidth": 0.8,
    }
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    ax.set_xlabel("Makespan")
    ax.set_ylabel("Active Energy")
    ax.set_title("(a) Long-CP: NAL vs AL", loc="left", fontweight="bold", pad=10)
    ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, trim=True)
    ax.legend(loc="lower right", frameon=True, edgecolor="gray", framealpha=0.9)

    # ---- Wide panel ----
    ax = axes[1]

    ax.scatter(
        wide_nal["makespan"],
        wide_nal["energy"],
        alpha=0.8,
        s=40,
        c=color_nal,
        edgecolors="white",
        linewidths=0.8,
        marker="o",
        label="NAL",
        zorder=3,
    )
    ax.scatter(
        wide_al["makespan"],
        wide_al["energy"],
        alpha=0.8,
        s=40,
        c=color_al,
        edgecolors="white",
        linewidths=0.8,
        marker="s",
        label="AL",
        zorder=3,
    )

    wide_nal_stats = compute_correlation_stats(wide_nal)
    wide_al_stats = compute_correlation_stats(wide_al)

    textstr = (
        f"NAL: $r = {wide_nal_stats['pearson_r']:.3f}$\n"
        f"AL:  $r = {wide_al_stats['pearson_r']:.3f}$"
    )
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    ax.set_xlabel("Makespan")
    ax.set_ylabel("Active Energy")
    ax.set_title("(b) Wide: NAL vs AL", loc="left", fontweight="bold", pad=10)
    ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, trim=True)
    ax.legend(loc="lower right", frameon=True, edgecolor="gray", framealpha=0.9)

    plt.tight_layout(pad=0.5)

    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(str(pdf_path), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved NAL vs AL comparison to: {out_path}")
    print(f"Saved PDF version to: {pdf_path}")


def main() -> None:
    # Define input paths (mean Pareto only)
    longcp_nal_path = (
        Path(PROJECT_ROOT)
        / "logs/NAL_case/long_cp_specialist_traj/ablation/per_variant/hetero/hetero_mean_pareto.csv"
    )
    wide_nal_path = (
        Path(PROJECT_ROOT)
        / "logs/NAL_case/wide_specialist_traj/ablation/per_variant/hetero/hetero_mean_pareto.csv"
    )

    longcp_al_path = (
        Path(PROJECT_ROOT)
        / "logs/AL_case/long_cp_specialist_traj_aligned/ablation/per_variant/hetero/hetero_mean_pareto.csv"
    )
    wide_al_path = (
        Path(PROJECT_ROOT)
        / "logs/AL_case/wide_specialist_traj_aligned/ablation/per_variant/hetero/hetero_mean_pareto.csv"
    )

    out_dir = Path(PROJECT_ROOT) / "logs/objective_correlation_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading hetero_mean_pareto.csv for NAL and AL (long-CP and wide)...")

    longcp_nal = load_pareto_csv(longcp_nal_path)
    wide_nal = load_pareto_csv(wide_nal_path)
    longcp_al = load_pareto_csv(longcp_al_path)
    wide_al = load_pareto_csv(wide_al_path)

    print(f"  Long-CP NAL: {len(longcp_nal)} checkpoints")
    print(f"  Wide NAL:    {len(wide_nal)} checkpoints")
    print(f"  Long-CP AL:  {len(longcp_al)} checkpoints")
    print(f"  Wide AL:     {len(wide_al)} checkpoints")

    out_path = out_dir / "objective_correlation_NAL_vs_AL.png"
    plot_case_comparison(longcp_nal, longcp_al, wide_nal, wide_al, out_path=out_path, dpi=300)

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
