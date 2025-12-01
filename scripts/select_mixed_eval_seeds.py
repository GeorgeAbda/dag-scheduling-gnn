#!/usr/bin/env python3
"""
Select a mixed set of representative eval seeds: k from 'wide' and k from 'longcp',
using a simple k-center greedy in standardized feature space.

Also produces diagnostic plots:
- PCA scatter highlighting selected seeds per domain
- Feature distributions per domain with selected seed markers

Usage:
  python -m scripts.select_mixed_eval_seeds \
    --wide-dir runs/datasets/wide/representativeness \
    --longcp-dir runs/datasets/longcp/representativeness \
    --k-per-domain 5 \
    --out-dir runs/datasets/mixed/representativeness

Outputs in out-dir:
- mixed_selected_eval_seeds.json
- pca_mixed_selected.png
- features_mixed_selected.png

Requirements: numpy, pandas, scikit-learn, matplotlib, seaborn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


FEAT_COLS = [
    "tasks",
    "edges",
    "width_peak",
    "depth_avg",
    "Pbar",
    "burstiness",
    "cp_frac",
]


def _load_features(base_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(base_dir / "train_features.csv")
    if "seed" not in df.columns:
        raise ValueError(f"Missing 'seed' column in {base_dir/'train_features.csv'}")
    missing = [c for c in FEAT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature cols {missing} in {base_dir/'train_features.csv'}")
    return df.copy()


def _zscore(df: pd.DataFrame, feat_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    mu = df[feat_cols].mean()
    sd = df[feat_cols].std().replace(0, 1.0)
    z = (df[feat_cols] - mu) / sd
    return z, mu, sd


def _kcenter_greedy(X: np.ndarray, k: int) -> List[int]:
    n = X.shape[0]
    if k >= n:
        return list(range(n))
    # Start with point farthest from the mean for determinism
    mu = X.mean(axis=0, keepdims=True)
    d_mu = np.linalg.norm(X - mu, axis=1)
    centers = [int(np.argmax(d_mu))]
    min_dists = np.linalg.norm(X - X[centers[0]], axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(min_dists))
        centers.append(nxt)
        d = np.linalg.norm(X - X[nxt], axis=1)
        min_dists = np.minimum(min_dists, d)
    # Keep original order for reproducibility by ascending index
    centers = sorted(set(centers))
    return centers


def _coverage_stats(X: np.ndarray, centers_idx: List[int]) -> Dict[str, object]:
    C = X[centers_idx]
    # distances to nearest selected center
    dists = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)  # [N,K]
    min_d = dists.min(axis=1)
    assign = dists.argmin(axis=1)
    cluster_sizes = [int((assign == j).sum()) for j in range(len(centers_idx))]
    return {
        "radius": float(min_d.max()),
        "avg_min_dist": float(min_d.mean()),
        "cluster_sizes": cluster_sizes,
    }


def _plot_pca(df_all: pd.DataFrame, feat_cols: List[str], selected: Dict[str, List[int]], out_path: Path) -> None:
    X = df_all[feat_cols].to_numpy(dtype=float)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-9)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)
    df_plot = df_all.copy()
    df_plot["pc1"] = Z[:, 0]
    df_plot["pc2"] = Z[:, 1]

    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df_plot, x="pc1", y="pc2", hue="domain", alpha=0.25, s=25)
    # highlight selected seeds
    colors = {"wide": "tab:orange", "longcp": "tab:blue"}
    for dom, seeds in selected.items():
        if not seeds:
            continue
        sub = df_plot[df_plot["seed"].isin(seeds)]
        plt.scatter(sub["pc1"], sub["pc2"], s=120, facecolors='none', edgecolors=colors.get(dom, 'black'), linewidths=2, label=f"{dom} selected")
    plt.title("PCA: selected seeds highlighted")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_features(df_all: pd.DataFrame, feat_cols: List[str], selected: Dict[str, List[int]], out_path: Path) -> None:
    plt.figure(figsize=(14, 10))
    n = len(feat_cols)
    for i, col in enumerate(feat_cols, 1):
        ax = plt.subplot(int(np.ceil(n/3)), 3, i)
        sns.kdeplot(data=df_all, x=col, hue="domain", common_norm=False, fill=True, alpha=0.3, linewidth=1)
        # Add vertical markers for selected seeds per domain
        for dom, seeds in selected.items():
            vals = df_all[(df_all["domain"] == dom) & (df_all["seed"].isin(seeds))][col].tolist()
            for v in vals:
                ax.axvline(v, color=("tab:orange" if dom == "wide" else "tab:blue"), linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_title(col)
        ax.grid(True, alpha=0.2)
    plt.suptitle("Feature distributions with selected seed markers", y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide-dir", type=Path, default=Path("runs/datasets/wide/representativeness"))
    ap.add_argument("--longcp-dir", type=Path, default=Path("runs/datasets/longcp/representativeness"))
    ap.add_argument("--k-per-domain", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/datasets/mixed/representativeness"))
    args = ap.parse_args()

    wide_df = _load_features(args.wide_dir)
    long_df = _load_features(args.longcp_dir)

    wide_df = wide_df[["seed"] + FEAT_COLS].copy()
    long_df = long_df[["seed"] + FEAT_COLS].copy()

    # Standardize per domain
    wide_z, wide_mu, wide_sd = _zscore(wide_df, FEAT_COLS)
    long_z, long_mu, long_sd = _zscore(long_df, FEAT_COLS)

    # Select K per domain via k-center greedy
    k = int(max(1, args.k_per_domain))
    wide_idx = _kcenter_greedy(wide_z.to_numpy(dtype=float), k)
    long_idx = _kcenter_greedy(long_z.to_numpy(dtype=float), k)

    wide_selected = [int(wide_df.iloc[i]["seed"]) for i in wide_idx]
    long_selected = [int(long_df.iloc[i]["seed"]) for i in long_idx]

    # Coverage stats per domain (in z-space)
    wide_cov = _coverage_stats(wide_z.to_numpy(dtype=float), wide_idx)
    long_cov = _coverage_stats(long_z.to_numpy(dtype=float), long_idx)

    # Combined JSON output
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = {
        "selected_eval_seeds": [*wide_selected, *long_selected],
        "by_domain": {
            "wide": {
                "selected": wide_selected,
                "coverage": wide_cov,
            },
            "longcp": {
                "selected": long_selected,
                "coverage": long_cov,
            },
        },
        "feature_names": FEAT_COLS,
        "notes": "Per-domain z-score; k-center greedy k={} per domain; coverage computed in z-space".format(k),
    }
    with (out_dir / "mixed_selected_eval_seeds.json").open("w") as f:
        json.dump(out_json, f, indent=2)

    # Plots
    df_all = pd.concat([
        wide_df.assign(domain="wide"),
        long_df.assign(domain="longcp"),
    ], ignore_index=True)
    selected_map = {"wide": wide_selected, "longcp": long_selected}

    _plot_pca(df_all, FEAT_COLS, selected_map, out_dir / "pca_mixed_selected.png")
    _plot_features(df_all, FEAT_COLS, selected_map, out_dir / "features_mixed_selected.png")

    print("[done] Wrote:")
    print(" -", out_dir / "mixed_selected_eval_seeds.json")
    print(" -", out_dir / "pca_mixed_selected.png")
    print(" -", out_dir / "features_mixed_selected.png")


if __name__ == "__main__":
    main()
