#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


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
        # t = log10(1/(1 - x)) with clipping to avoid inf at x=1
        Xt = np.log10(1.0 / np.maximum(eps, 1.0 - np.clip(X, 0.0, 1.0)))
        Yt = np.log10(1.0 / np.maximum(eps, 1.0 - np.clip(Y, 0.0, 1.0)))
        return Xt, Yt
    raise ValueError(f"Unknown axis transform: {mode}")


def plot_single(ax, X, Y, A, levels: List[float], cmap='inferno', axis_mode='log10_inv1m', title="", contour_colors='white'):
    Xt, Yt = transform_axes(X, Y, axis_mode)
    # pcolormesh expects cell corners; convert grid centers to edges by padding
    # Here we assume X,Y form a rectilinear grid; approximate edges
    def to_edges(M):
        # average adjacent points to get midpoints; extrapolate edges
        mx = np.pad(M, ((0,1),(0,1)), mode='edge')
        mx[1:, :] = 0.5*(mx[1:, :] + mx[:-1, :])
        mx[:, 1:] = 0.5*(mx[:, 1:] + mx[:, :-1])
        return mx
    Xe = to_edges(Xt)
    Ye = to_edges(Yt)
    im = ax.pcolormesh(Xe, Ye, A, shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)
    if levels:
        cs = ax.contour(Xt, Yt, A, levels=levels, colors=contour_colors, linewidths=1.2)
        ax.clabel(cs, fmt=lambda v: f"{v:.2g}", inline=True, fontsize=8, colors='k')
    ax.set_title(title)
    ax.set_xlabel("log10(1/(1 - makespan)) (normalized)")
    ax.set_ylabel("log10(1/(1 - active_energy)) (normalized)")
    return im


def plot_ratio(ax, X, Y, A, B, axis_mode='log10_inv1m', title="ratio A/B", vmin=0.5, vmax=2.0):
    Xt, Yt = transform_axes(X, Y, axis_mode)
    ratio = (A + 1e-6) / (B + 1e-6)
    ratio = np.clip(ratio, 1e-6, 1e6)
    def to_edges(M):
        mx = np.pad(M, ((0,1),(0,1)), mode='edge')
        mx[1:, :] = 0.5*(mx[1:, :] + mx[:-1, :])
        mx[:, 1:] = 0.5*(mx[:, 1:] + mx[:, :-1])
        return mx
    Xe = to_edges(Xt)
    Ye = to_edges(Yt)
    im = ax.pcolormesh(Xe, Ye, ratio, shading='auto', cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.contour(Xt, Yt, ratio, levels=[1.0], colors='k', linewidths=1.2)
    ax.set_title(title)
    ax.set_xlabel("log10(1/(1 - makespan)) (normalized)")
    ax.set_ylabel("log10(1/(1 - active_energy)) (normalized)")
    return im


def main():
    ap = argparse.ArgumentParser(description='Plot EAF grids as heatmaps with contours and optional ratio panels.')
    ap.add_argument('--grid', action='append', help='CSV path to an EAF grid (x,y,alpha). Repeatable.')
    ap.add_argument('--name', action='append', help='Name for each grid, same order as --grid.')
    ap.add_argument('--levels', type=str, default='0.5,0.75,0.9', help='Comma-separated contour levels.')
    ap.add_argument('--axis-mode', type=str, default='log10_inv1m', choices=['identity','log10_inv1m'], help='Axis transform for visualization.')
    ap.add_argument('--out', required=True, help='Output image path (png/svg).')
    ap.add_argument('--dpi', type=int, default=180)
    ap.add_argument('--ratio', action='store_true', help='If two grids are given, also draw a ratio panel A/B.')
    ap.add_argument('--cmap', type=str, default='coolwarm', help='Colormap for regions (e.g., coolwarm, RdBu_r, magma, inferno).')
    args = ap.parse_args()

    if not args.grid or len(args.grid) == 0:
        raise SystemExit('--grid must be provided at least once')
    names = args.name or [f'g{i+1}' for i in range(len(args.grid))]
    if len(names) != len(args.grid):
        raise SystemExit('--name count must match --grid count')

    levels = [float(s.strip()) for s in args.levels.split(',') if s.strip()]

    # Load all grids
    grids = [load_grid(Path(p)) for p in args.grid]
    # Basic sanity: ensure same X,Y across grids
    X0, Y0, _ = grids[0]
    for i, (Xi, Yi, _) in enumerate(grids[1:], start=1):
        if Xi.shape != X0.shape or Yi.shape != Y0.shape:
            raise SystemExit(f'Grid {i} shape differs from grid 0')
        if not (np.allclose(Xi, X0) and np.allclose(Yi, Y0)):
            # We proceed but warn; plotting still okay but ratio may be invalid
            print(f'[warn] X/Y coordinates differ between grids 0 and {i}')

    n = len(grids)
    make_ratio = args.ratio and n == 2

    cols = 2 if make_ratio else min(n, 3)
    rows = int(np.ceil(n / cols))
    if make_ratio:
        rows = max(rows, 2)

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

    # Plot individual grids
    for idx, ((X, Y, A), name) in enumerate(zip(grids, names)):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        im = plot_single(
            ax, X, Y, A,
            levels=levels,
            cmap=args.cmap,
            axis_mode=args.axis_mode,
            title=f"{name} EAF",
            contour_colors='k',
        )
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('attainment (alpha)')

    # Optional ratio panel occupying last cell
    if make_ratio:
        X, Y, A = grids[0]
        _, _, B = grids[1]
        ax = axes[-1][-1]
        imr = plot_ratio(ax, X, Y, A, B, axis_mode=args.axis_mode, title=f"ratio {names[0]}/{names[1]}")
        cb = fig.colorbar(imr, ax=ax)
        cb.set_label('ratio A/B')

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    print(f"[done] Wrote {out_path}")


if __name__ == '__main__':
    main()
