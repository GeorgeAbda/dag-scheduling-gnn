#!/usr/bin/env python3
"""
Analyze objective correlation during training for wide and long-cp specialists.

This script reads the Pareto CSV files from training and computes/visualizes:
1. Correlation between makespan and energy objectives during training
2. Trajectory of objectives over training iterations
3. Variance and convergence patterns

Usage:
    python scripts/analyze_objective_correlation_training.py
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
    """Load Pareto CSV and return DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Pareto CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    # Remove any empty rows
    df = df.dropna(subset=['makespan', 'energy'])
    return df


def compute_correlation_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Compute correlation statistics between makespan and energy."""
    makespan = df['makespan'].values
    energy = df['energy'].values
    
    if len(makespan) < 2:
        return {
            'pearson_r': np.nan,
            'pearson_p': np.nan,
            'makespan_mean': np.nan,
            'makespan_std': np.nan,
            'energy_mean': np.nan,
            'energy_std': np.nan,
            'n_points': 0,
        }
    
    r, p = pearsonr(makespan, energy)
    
    return {
        'pearson_r': r,
        'pearson_p': p,
        'makespan_mean': np.mean(makespan),
        'makespan_std': np.std(makespan),
        'energy_mean': np.mean(energy),
        'energy_std': np.std(energy),
        'n_points': len(makespan),
    }


def plot_objective_correlation_comparison(
    longcp_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Create a publication-quality comparison plot showing objective correlation."""
    
    # Publication-quality settings (NeurIPS/ICML/ICLR style)
    plt.rcParams.update({
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
        "text.usetex": False,  # Set to True if LaTeX is available
    })
    
    # Use white background for publication
    sns.set_style("white")
    
    # Create figure with specific aspect ratio
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))  # Two-column format
    
    # Compute correlations
    longcp_stats = compute_correlation_stats(longcp_df)
    wide_stats = compute_correlation_stats(wide_df)
    
    # Color scheme: professional and colorblind-friendly
    color_longcp = '#2E86AB'  # Blue
    color_wide = '#A23B72'    # Purple-magenta
    color_fit = '#F18F01'     # Orange
    
    # Plot 1: Long-CP specialist
    ax = axes[0]
    
    # Scatter plot with different markers for different Pareto types
    markers = {'mean': 'o', 'cvar': 's', 'worst': '^'}
    for ptype, marker in markers.items():
        mask = longcp_df.get('pareto_type', 'mean') == ptype
        if mask.any():
            ax.scatter(
                longcp_df.loc[mask, 'makespan'],
                longcp_df.loc[mask, 'energy'],
                alpha=0.7,
                s=40,
                c=color_longcp,
                edgecolors='white',
                linewidths=0.8,
                marker=marker,
                zorder=3,
                label=f'{ptype.capitalize()}'
            )
    
    # Add regression line with confidence interval
    if not np.isnan(longcp_stats['pearson_r']) and len(longcp_df) > 1:
        from scipy import stats
        x = longcp_df['makespan'].values
        y = longcp_df['energy'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=color_fit, linewidth=2, 
                linestyle='--', alpha=0.9, zorder=2, label='Linear fit')
    
    ax.set_xlabel('Makespan', fontweight='normal')
    ax.set_ylabel('Active Energy', fontweight='normal')
    ax.set_title('(a) Long-CP Specialist', loc='left', fontweight='bold', pad=10)
    
    # Add correlation annotation in a box
    textstr = f'$r = {longcp_stats["pearson_r"]:.3f}$'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9, linewidth=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    # Add legend
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='gray', framealpha=0.9, fontsize=8)
    
    # Clean grid
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, trim=True)
    
    # Plot 2: Wide specialist
    ax = axes[1]
    
    # Scatter plot with different markers for different Pareto types
    for ptype, marker in markers.items():
        mask = wide_df.get('pareto_type', 'mean') == ptype
        if mask.any():
            ax.scatter(
                wide_df.loc[mask, 'makespan'],
                wide_df.loc[mask, 'energy'],
                alpha=0.7,
                s=40,
                c=color_wide,
                edgecolors='white',
                linewidths=0.8,
                marker=marker,
                zorder=3,
                label=f'{ptype.capitalize()}'
            )
    
    # Add regression line
    if not np.isnan(wide_stats['pearson_r']) and len(wide_df) > 1:
        from scipy import stats
        x = wide_df['makespan'].values
        y = wide_df['energy'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=color_fit, linewidth=2, 
                linestyle='--', alpha=0.9, zorder=2, label='Linear fit')
    
    ax.set_xlabel('Makespan', fontweight='normal')
    ax.set_ylabel('Active Energy', fontweight='normal')
    ax.set_title('(b) Wide Specialist', loc='left', fontweight='bold', pad=10)
    
    # Add correlation annotation
    textstr = f'$r = {wide_stats["pearson_r"]:.3f}$'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9, linewidth=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    # Add legend
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='gray', framealpha=0.9, fontsize=8)
    
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, trim=True)
    
    plt.tight_layout(pad=0.5)
    
    # Save in multiple formats for publication
    fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight', facecolor='white')
    pdf_path = out_path.with_suffix('.pdf')
    fig.savefig(str(pdf_path), bbox_inches='tight', facecolor='white')
    
    plt.close(fig)
    print(f"Saved objective correlation comparison to: {out_path}")
    print(f"Saved PDF version to: {pdf_path}")


def plot_training_trajectory(
    longcp_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Plot how objectives evolve during training - publication quality."""
    
    # Publication settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 1.0,
        "text.usetex": False,
    })
    sns.set_style("white")
    
    # Create figure - single column format for better readability
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
    
    # Color scheme
    color_longcp = '#2E86AB'
    color_wide = '#A23B72'
    
    # Convert steps to millions for readability
    longcp_steps_m = longcp_df['global_step'].values / 1e6
    wide_steps_m = wide_df['global_step'].values / 1e6
    
    # Long-CP: Makespan over training
    ax = axes[0, 0]
    ax.plot(longcp_steps_m, longcp_df['makespan'], 
            marker='o', markersize=3, alpha=0.8, color=color_longcp, 
            linewidth=1.5, markeredgewidth=0.5, markeredgecolor='white')
    ax.set_xlabel('Training Steps (M)', fontweight='normal')
    ax.set_ylabel('Makespan', fontweight='normal')
    ax.set_title('(a) Long-CP: Makespan', loc='left', fontweight='bold', pad=8)
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, trim=True)
    
    # Long-CP: Energy over training
    ax = axes[0, 1]
    ax.plot(longcp_steps_m, longcp_df['energy'], 
            marker='o', markersize=3, alpha=0.8, color=color_longcp, 
            linewidth=1.5, markeredgewidth=0.5, markeredgecolor='white')
    ax.set_xlabel('Training Steps (M)', fontweight='normal')
    ax.set_ylabel('Active Energy', fontweight='normal')
    ax.set_title('(b) Long-CP: Energy', loc='left', fontweight='bold', pad=8)
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, trim=True)
    
    # Wide: Makespan over training
    ax = axes[1, 0]
    ax.plot(wide_steps_m, wide_df['makespan'], 
            marker='s', markersize=3, alpha=0.8, color=color_wide, 
            linewidth=1.5, markeredgewidth=0.5, markeredgecolor='white')
    ax.set_xlabel('Training Steps (M)', fontweight='normal')
    ax.set_ylabel('Makespan', fontweight='normal')
    ax.set_title('(c) Wide: Makespan', loc='left', fontweight='bold', pad=8)
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, trim=True)
    
    # Wide: Energy over training
    ax = axes[1, 1]
    ax.plot(wide_steps_m, wide_df['energy'], 
            marker='s', markersize=3, alpha=0.8, color=color_wide, 
            linewidth=1.5, markeredgewidth=0.5, markeredgecolor='white')
    ax.set_xlabel('Training Steps (M)', fontweight='normal')
    ax.set_ylabel('Active Energy', fontweight='normal')
    ax.set_title('(d) Wide: Energy', loc='left', fontweight='bold', pad=8)
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, trim=True)
    
    plt.tight_layout(pad=0.5)
    
    fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight', facecolor='white')
    pdf_path = out_path.with_suffix('.pdf')
    fig.savefig(str(pdf_path), bbox_inches='tight', facecolor='white')
    
    plt.close(fig)
    print(f"Saved training trajectory to: {out_path}")
    print(f"Saved PDF version to: {pdf_path}")


def plot_variance_comparison(
    longcp_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    out_path: Path,
    window_size: int = 5,
    dpi: int = 300,
) -> None:
    """Plot rolling variance to show convergence patterns - publication quality."""
    
    # Publication settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "text.usetex": False,
    })
    sns.set_style("white")
    
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    
    # Color scheme
    color_longcp = '#2E86AB'
    color_wide = '#A23B72'
    
    # Convert to millions
    longcp_steps_m = longcp_df['global_step'].values / 1e6
    wide_steps_m = wide_df['global_step'].values / 1e6
    
    # Compute rolling variance for makespan
    ax = axes[0]
    if len(longcp_df) >= window_size:
        longcp_var = longcp_df['makespan'].rolling(window=window_size, min_periods=1).std()
        ax.plot(longcp_steps_m, longcp_var, 
                label='Long-CP', color=color_longcp, linewidth=2, alpha=0.85)
    if len(wide_df) >= window_size:
        wide_var = wide_df['makespan'].rolling(window=window_size, min_periods=1).std()
        ax.plot(wide_steps_m, wide_var, 
                label='Wide', color=color_wide, linewidth=2, alpha=0.85)
    ax.set_xlabel('Training Steps (M)', fontweight='normal')
    ax.set_ylabel(f'Rolling Std (w={window_size})', fontweight='normal')
    ax.set_title('(a) Makespan Variability', loc='left', fontweight='bold', pad=8)
    ax.legend(frameon=True, fancybox=False, edgecolor='gray', framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, trim=True)
    
    # Compute rolling variance for energy
    ax = axes[1]
    if len(longcp_df) >= window_size:
        longcp_var = longcp_df['energy'].rolling(window=window_size, min_periods=1).std()
        ax.plot(longcp_steps_m, longcp_var, 
                label='Long-CP', color=color_longcp, linewidth=2, alpha=0.85)
    if len(wide_df) >= window_size:
        wide_var = wide_df['energy'].rolling(window=window_size, min_periods=1).std()
        ax.plot(wide_steps_m, wide_var, 
                label='Wide', color=color_wide, linewidth=2, alpha=0.85)
    ax.set_xlabel('Training Steps (M)', fontweight='normal')
    ax.set_ylabel(f'Rolling Std (w={window_size})', fontweight='normal')
    ax.set_title('(b) Energy Variability', loc='left', fontweight='bold', pad=8)
    ax.legend(frameon=True, fancybox=False, edgecolor='gray', framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, trim=True)
    
    plt.tight_layout(pad=0.5)
    
    fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight', facecolor='white')
    pdf_path = out_path.with_suffix('.pdf')
    fig.savefig(str(pdf_path), bbox_inches='tight', facecolor='white')
    
    plt.close(fig)
    print(f"Saved variance comparison to: {out_path}")
    print(f"Saved PDF version to: {pdf_path}")


def print_summary_statistics(longcp_df: pd.DataFrame, wide_df: pd.DataFrame) -> None:
    """Print summary statistics for both specialists."""
    
    print("\n" + "="*80)
    print("OBJECTIVE CORRELATION ANALYSIS: LONG-CP vs WIDE SPECIALISTS")
    print("="*80)
    
    longcp_stats = compute_correlation_stats(longcp_df)
    wide_stats = compute_correlation_stats(wide_df)
    
    print("\n--- LONG-CP SPECIALIST ---")
    print(f"  Pareto checkpoints: {longcp_stats['n_points']}")
    print(f"  Makespan: mean={longcp_stats['makespan_mean']:.4f}, std={longcp_stats['makespan_std']:.6f}")
    print(f"  Energy:   mean={longcp_stats['energy_mean']:.4f}, std={longcp_stats['energy_std']:.6f}")
    print(f"  Pearson correlation (makespan vs energy): r={longcp_stats['pearson_r']:.4f}, p={longcp_stats['pearson_p']:.2e}")
    
    print("\n--- WIDE SPECIALIST ---")
    print(f"  Pareto checkpoints: {wide_stats['n_points']}")
    print(f"  Makespan: mean={wide_stats['makespan_mean']:.4f}, std={wide_stats['makespan_std']:.6f}")
    print(f"  Energy:   mean={wide_stats['energy_mean']:.4f}, std={wide_stats['energy_std']:.6f}")
    print(f"  Pearson correlation (makespan vs energy): r={wide_stats['pearson_r']:.4f}, p={wide_stats['pearson_p']:.2e}")
    
    print("\n--- INTERPRETATION ---")
    
    # Correlation comparison
    if not np.isnan(longcp_stats['pearson_r']) and not np.isnan(wide_stats['pearson_r']):
        if abs(longcp_stats['pearson_r']) > abs(wide_stats['pearson_r']):
            stronger = "Long-CP"
            weaker = "Wide"
            diff = abs(longcp_stats['pearson_r']) - abs(wide_stats['pearson_r'])
        else:
            stronger = "Wide"
            weaker = "Long-CP"
            diff = abs(wide_stats['pearson_r']) - abs(longcp_stats['pearson_r'])
        
        print(f"  {stronger} specialist shows STRONGER objective coupling (Δr = {diff:.3f})")
        print(f"  → {stronger} training creates tighter makespan-energy dependency")
        print(f"  → {weaker} objectives are more decoupled during training")
    
    # Variance comparison
    if longcp_stats['makespan_std'] < wide_stats['makespan_std']:
        print(f"\n  Long-CP has LOWER makespan variance ({longcp_stats['makespan_std']:.6f} vs {wide_stats['makespan_std']:.6f})")
        print(f"  → Long-CP converges to more consistent makespan solutions")
    else:
        print(f"\n  Wide has LOWER makespan variance ({wide_stats['makespan_std']:.6f} vs {longcp_stats['makespan_std']:.6f})")
        print(f"  → Wide converges to more consistent makespan solutions")
    
    if longcp_stats['energy_std'] < wide_stats['energy_std']:
        print(f"  Long-CP has LOWER energy variance ({longcp_stats['energy_std']:.6f} vs {wide_stats['energy_std']:.6f})")
        print(f"  → Long-CP converges to more consistent energy solutions")
    else:
        print(f"  Wide has LOWER energy variance ({wide_stats['energy_std']:.6f} vs {longcp_stats['energy_std']:.6f})")
        print(f"  → Wide converges to more consistent energy solutions")
    
    print("\n" + "="*80 + "\n")


def main():
    # Define paths - load ALL Pareto variants (mean, cvar, worst)
    longcp_dir = Path(PROJECT_ROOT) / "logs/long_cp_specialist_traj_aligned/ablation/per_variant/hetero"
    wide_dir = Path(PROJECT_ROOT) / "logs/wide_specialist_traj_aligned/ablation/per_variant/hetero"
    
    out_dir = Path(PROJECT_ROOT) / "logs/objective_correlation_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data from all Pareto files
    print("Loading ALL Pareto CSV files (mean, cvar, worst)...")
    
    pareto_types = ['mean', 'cvar', 'worst']
    longcp_dfs = []
    wide_dfs = []
    
    for ptype in pareto_types:
        longcp_path = longcp_dir / f"hetero_{ptype}_pareto.csv"
        wide_path = wide_dir / f"hetero_{ptype}_pareto.csv"
        
        if longcp_path.exists():
            df = load_pareto_csv(longcp_path)
            df['pareto_type'] = ptype
            longcp_dfs.append(df)
            print(f"  Long-CP {ptype}: {len(df)} checkpoints")
        
        if wide_path.exists():
            df = load_pareto_csv(wide_path)
            df['pareto_type'] = ptype
            wide_dfs.append(df)
            print(f"  Wide {ptype}:    {len(df)} checkpoints")
    
    # Concatenate all Pareto fronts
    longcp_df = pd.concat(longcp_dfs, ignore_index=True) if longcp_dfs else pd.DataFrame()
    wide_df = pd.concat(wide_dfs, ignore_index=True) if wide_dfs else pd.DataFrame()
    
    print(f"\n  Total Long-CP: {len(longcp_df)} Pareto checkpoints")
    print(f"  Total Wide:    {len(wide_df)} Pareto checkpoints")
    
    # Print statistics
    print_summary_statistics(longcp_df, wide_df)
    
    # Generate plots
    print("Generating visualizations...")
    
    plot_objective_correlation_comparison(
        longcp_df, wide_df,
        out_dir / "objective_correlation_comparison.png",
        dpi=300
    )
    
    plot_training_trajectory(
        longcp_df, wide_df,
        out_dir / "training_trajectory.png",
        dpi=300
    )
    
    plot_variance_comparison(
        longcp_df, wide_df,
        out_dir / "variance_comparison.png",
        window_size=5,
        dpi=300
    )
    
    print(f"\nAll outputs saved to: {out_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
