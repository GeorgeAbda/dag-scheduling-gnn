#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_grid(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    a = df['alpha'].to_numpy()
    g = int(np.sqrt(len(df)))
    if g * g != len(df):
        raise ValueError(f"Grid is not square: {csv_path} length={len(df)}")
    X = x.reshape(g, g)
    Y = y.reshape(g, g)
    A = a.reshape(g, g)
    return X, Y, A


def transform_axes(X: np.ndarray, Y: np.ndarray, mode: str, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    if mode == 'identity':
        return X, Y
    if mode == 'log10_inv1m':
        Xt = np.log10(1.0 / np.maximum(eps, 1.0 - np.clip(X, 0.0, 1.0)))
        Yt = np.log10(1.0 / np.maximum(eps, 1.0 - np.clip(Y, 0.0, 1.0)))
        return Xt, Yt
    raise ValueError(f"Unknown axis transform: {mode}")


def to_edges(M: np.ndarray) -> np.ndarray:
    mx = np.pad(M, ((0,1),(0,1)), mode='edge')
    mx[1:, :] = 0.5*(mx[1:, :] + mx[:-1, :])
    mx[:, 1:] = 0.5*(mx[:, 1:] + mx[:, :-1])
    return mx


def plot_eaf_panel(ax, X, Y, A, title: str, levels: List[float], axis_mode: str, cmap: str = 'RdBu_r'):
    Xt, Yt = transform_axes(X, Y, axis_mode)
    Xe = to_edges(Xt)
    Ye = to_edges(Yt)
    im = ax.pcolormesh(Xe, Ye, A, shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)
    if levels:
        cs = ax.contour(Xt, Yt, A, levels=levels, colors='k', linewidths=1.0)
        ax.clabel(cs, fmt=lambda v: f"{v:.2g}", inline=True, fontsize=7, colors='k')
    ax.set_title(title)
    ax.set_xlabel("log10(1/(1 - makespan)) (normalized)")
    ax.set_ylabel("log10(1/(1 - active_energy)) (normalized)")
    return im


def area_fraction_vs_level(A: np.ndarray, levels: List[float]) -> np.ndarray:
    flat = A.reshape(-1)
    return np.array([(flat >= lv).mean() for lv in levels], dtype=float)


def main():
    ap = argparse.ArgumentParser(description='Create a multi-panel EAF overview figure: 3 stacked heatmaps + bottom summary line panel.')
    ap.add_argument('--grid', action='append', required=True, help='CSV path to an EAF grid (x,y,alpha). Repeat for each model (3 recommended).')
    ap.add_argument('--name', action='append', help='Name for each grid, same order as --grid.')
    ap.add_argument('--levels', type=str, default='0.5,0.75,0.9', help='Comma-separated contour/summary levels.')
    ap.add_argument('--axis-mode', type=str, default='log10_inv1m', choices=['identity','log10_inv1m'])
    ap.add_argument('--out', required=True, help='Output path (png/svg).')
    ap.add_argument('--dpi', type=int, default=200)
    args = ap.parse_args()

    names = args.name or [f'm{i+1}' for i in range(len(args.grid))]
    if len(names) != len(args.grid):
        raise SystemExit('--name count must match --grid count')

    levels = [float(s.strip()) for s in args.levels.split(',') if s.strip()]

    # Load grids
    grids = [load_grid(Path(p)) for p in args.grid]

    # Figure layout: 3 rows heatmaps + 1 row summary line plot
    rows_hm = len(grids)
    fig = plt.figure(figsize=(8, 2.8*rows_hm + 2.8))
    gs = GridSpec(rows_hm + 1, 1, height_ratios=[1]*rows_hm + [1.0], hspace=0.28)

    # Colorbar shared for heatmaps (attach to last axis)
    last_im = None
    for i, ((X, Y, A), name) in enumerate(zip(grids, names)):
        ax = fig.add_subplot(gs[i, 0])
        im = plot_eaf_panel(ax, X, Y, A, title=name, levels=levels, axis_mode=args.axis_mode, cmap='coolwarm')
        last_im = im

    # Shared colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label('attainment (alpha)')

    # Summary line panel
    ax_sum = fig.add_subplot(gs[rows_hm, 0])
    # Color mapping to match template: blue (mlp), red (baseline), green (hetero)
    template_colors = {
        'mlp_only': '#1f77b4',
        'mlp': '#1f77b4',
        'baseline': '#d62728',
        'hetero': '#2ca02c',
    }
    fallback = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    for idx, ((X, Y, A), name) in enumerate(zip(grids, names)):
        af = area_fraction_vs_level(A, levels)
        lname = str(name).lower()
        color = None
        for key, val in template_colors.items():
            if key in lname:
                color = val
                break
        if color is None:
            color = fallback[idx % len(fallback)]
        ax_sum.plot(levels, af, marker='o', label=name, color=color)
    ax_sum.set_xlabel('attainment level')
    ax_sum.set_ylabel('fraction of grid with alpha ≥ level')
    ax_sum.set_xlim(min(levels), max(levels))
    ax_sum.set_ylim(0, 1)
    ax_sum.grid(True, linestyle='--', alpha=0.3)
    ax_sum.legend(frameon=False, ncols=min(3, len(names)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    print(f"[done] Wrote {out_path}")


if __name__ == '__main__':
    main()
