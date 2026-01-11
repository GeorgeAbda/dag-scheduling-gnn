#!/usr/bin/env python3
"""Create a single 2x2 figure combining AL and NAL objective correlation panels.

Subplots:
    (a) AL Long-CP
    (b) AL Wide
    (c) NAL Long-CP
    (d) NAL Wide

Usage:
    python scripts/analyze_objective_correlation_combined.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

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
    if not csv_path.exists():
        raise FileNotFoundError(f"Pareto CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["makespan", "energy"])
    return df


def load_case_data(longcp_dir: Path, wide_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pareto_types = ["mean", "cvar", "worst"]
    longcp_dfs = []
    wide_dfs = []

    for ptype in pareto_types:
        longcp_path = longcp_dir / f"hetero_{ptype}_pareto.csv"
        wide_path = wide_dir / f"hetero_{ptype}_pareto.csv"

        if longcp_path.exists():
            df_l = load_pareto_csv(longcp_path)
            df_l["pareto_type"] = ptype
            longcp_dfs.append(df_l)

        if wide_path.exists():
            df_w = load_pareto_csv(wide_path)
            df_w["pareto_type"] = ptype
            wide_dfs.append(df_w)

    longcp_df = pd.concat(longcp_dfs, ignore_index=True) if longcp_dfs else pd.DataFrame()
    wide_df = pd.concat(wide_dfs, ignore_index=True) if wide_dfs else pd.DataFrame()
    return longcp_df, wide_df


def compute_correlation_stats(df: pd.DataFrame) -> Dict[str, float]:
    makespan = df["makespan"].values
    energy = df["energy"].values
    if len(makespan) < 2:
        return {"pearson_r": np.nan, "pearson_p": np.nan, "n_points": len(makespan)}
    r, p = pearsonr(makespan, energy)
    return {"pearson_r": r, "pearson_p": p, "n_points": len(makespan)}


def plot_combined(
    nal_longcp: pd.DataFrame,
    nal_wide: pd.DataFrame,
    al_longcp: pd.DataFrame,
    al_wide: pd.DataFrame,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Create 2x2 figure: (a) AL Long-CP, (b) AL Wide, (c) NAL Long-CP, (d) NAL Wide."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
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

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    color_longcp = "#2E86AB"
    color_wide = "#A23B72"
    color_fit = "#F18F01"
    markers = {"mean": "o", "cvar": "s", "worst": "^"}

    def _plot_panel(ax, df, color, panel_label, title_suffix):
        for ptype, marker in markers.items():
            mask = df.get("pareto_type", "mean") == ptype
            if isinstance(mask, pd.Series) and mask.any():
                ax.scatter(
                    df.loc[mask, "makespan"],
                    df.loc[mask, "energy"],
                    alpha=0.7,
                    s=35,
                    c=color,
                    edgecolors="white",
                    linewidths=0.8,
                    marker=marker,
                    zorder=3,
                    label=f"{ptype.capitalize()}",
                )

        stats = compute_correlation_stats(df) if not df.empty else {"pearson_r": np.nan}
        if not np.isnan(stats["pearson_r"]) and len(df) > 1:
            from scipy import stats as sp_stats

            x = df["makespan"].values
            y = df["energy"].values
            slope, intercept, _, _, _ = sp_stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                color=color_fit,
                linewidth=2,
                linestyle="--",
                alpha=0.9,
                zorder=2,
                label="Linear fit",
            )

        textstr = f"$r = {stats['pearson_r']:.3f}$"
        props = {
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.9,
            "linewidth": 0.8,
        }
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="top", bbox=props)

        ax.set_xlabel("Makespan")
        ax.set_ylabel("Active Energy")
        ax.set_title(f"({panel_label}) {title_suffix}", loc="left", fontweight="bold", pad=8)
        ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, trim=True)
        ax.legend(loc="lower right", frameon=True, edgecolor="gray", framealpha=0.9)

    # (a) AL Long-CP
    _plot_panel(axes[0, 0], al_longcp, color_longcp, "a", "AL Long-CP")

    # (b) AL Wide
    _plot_panel(axes[0, 1], al_wide, color_wide, "b", "AL Wide")

    # (c) NAL Long-CP
    _plot_panel(axes[1, 0], nal_longcp, color_longcp, "c", "NAL Long-CP")

    # (d) NAL Wide
    _plot_panel(axes[1, 1], nal_wide, color_wide, "d", "NAL Wide")

    plt.tight_layout(pad=0.5)

    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(str(pdf_path), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved combined NAL/AL objective correlation figure to: {out_path}")
    print(f"Saved PDF version to: {pdf_path}")


def main() -> None:
    out_dir = Path(PROJECT_ROOT) / "logs/objective_correlation_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # NAL case
    nal_longcp_dir = (
        Path(PROJECT_ROOT)
        / "logs/NAL_case/long_cp_specialist_traj/ablation/per_variant/hetero"
    )
    nal_wide_dir = (
        Path(PROJECT_ROOT)
        / "logs/NAL_case/wide_specialist_traj/ablation/per_variant/hetero"
    )

    print("Loading NAL case (long-CP and wide, mean/cvar/worst Pareto)...")
    nal_longcp_df, nal_wide_df = load_case_data(nal_longcp_dir, nal_wide_dir)
    print(f"  NAL Long-CP: {len(nal_longcp_df)} Pareto checkpoints")
    print(f"  NAL Wide:    {len(nal_wide_df)} Pareto checkpoints")

    # AL case
    al_longcp_dir = (
        Path(PROJECT_ROOT)
        / "logs/AL_case/long_cp_specialist_traj_aligned/ablation/per_variant/hetero"
    )
    al_wide_dir = (
        Path(PROJECT_ROOT)
        / "logs/AL_case/wide_specialist_traj_aligned/ablation/per_variant/hetero"
    )

    print("\nLoading AL case (long-CP and wide, mean/cvar/worst Pareto)...")
    al_longcp_df, al_wide_df = load_case_data(al_longcp_dir, al_wide_dir)
    print(f"  AL Long-CP: {len(al_longcp_df)} Pareto checkpoints")
    print(f"  AL Wide:    {len(al_wide_df)} Pareto checkpoints")

    out_path = out_dir / "objective_correlation_NAL_AL_combined.png"
    plot_combined(nal_longcp_df, nal_wide_df, al_longcp_df, al_wide_df, out_path=out_path, dpi=300)

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
