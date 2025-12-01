#!/usr/bin/env python3
"""
Geometric analysis of dataset feature distributions (longcp vs wide).
- Loads train_features.csv for both domains and their selected_eval_seeds.
- Verifies queue-free assumption from runs/analysis/*/summary.csv.
- Computes standardized deltas, PCA projections, kNN density, and Mahalanobis distances.
- Exports:
  - PCA scatter with convex hulls per domain and highlighted eval seeds
  - CSV listing boundary seeds (lowest density / largest Mahalanobis) per domain
  - JSON summary of queue_free proportions and IFR ranges

Usage:
  python -m scripts.geometry_oob_analysis \
      --out-dir runs/representativeness/geometry

Dependencies: numpy, pandas, scikit-learn, matplotlib, seaborn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns


def _load_domain_features(base_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(base_dir / "train_features.csv")
    df["domain"] = base_dir.parent.name  # longcp or wide
    return df


def _load_selected_eval(path_json: Path) -> List[int]:
    with open(path_json, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "selected_eval_seeds" in data:
        return [int(x) for x in data["selected_eval_seeds"]]
    return []


def _zscore(df: pd.DataFrame, feat_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    mu = df[feat_cols].mean()
    sd = df[feat_cols].std().replace(0, 1.0)
    z = (df[feat_cols] - mu) / sd
    return z, mu, sd


def _mahalanobis(x: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    diff = x - mu
    return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))


def _cov_inv(X: np.ndarray) -> np.ndarray:
    cov = np.cov(X, rowvar=False)
    # Add a small ridge for numerical stability
    eps = 1e-6
    cov += np.eye(cov.shape[0]) * eps
    return np.linalg.pinv(cov)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    # Andrew's monotone chain in 2D
    if points.shape[0] < 3:
        return points
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1])
    return hull


def _plot_pca(df_all: pd.DataFrame, feat_cols: List[str], eval_map: Dict[str, List[int]], out_dir: Path) -> None:
    pca = PCA(n_components=2, random_state=0)
    X = df_all[feat_cols].values
    Z = pca.fit_transform(X)
    df_plot = df_all.copy()
    df_plot["pc1"] = Z[:, 0]
    df_plot["pc2"] = Z[:, 1]

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df_plot, x="pc1", y="pc2", hue="domain", s=24, alpha=0.7)
    # Draw convex hulls per domain
    for dom, grp in df_plot.groupby("domain"):
        pts = grp[["pc1", "pc2"]].to_numpy()
        hull = _convex_hull(pts)
        if hull.shape[0] >= 3:
            plt.fill(hull[:, 0], hull[:, 1], alpha=0.08, label=f"{dom}-hull")
    # Overlay eval seeds
    for dom, seeds in eval_map.items():
        mask = (df_plot["domain"] == dom) & (df_plot["seed"].isin(seeds))
        sub = df_plot[mask]
        if not sub.empty:
            plt.scatter(sub["pc1"], sub["pc2"], s=60, marker="x", label=f"{dom}-eval", linewidths=1.5, c="black")
    plt.title("PCA of train features (with convex hulls and eval seeds)")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "pca_hulls_eval.png", dpi=180)
    plt.close()


def _knn_density(df: pd.DataFrame, feat_cols: List[str], k: int = 10) -> np.ndarray:
    X = df[feat_cols].values
    nbrs = NearestNeighbors(n_neighbors=min(k, max(2, X.shape[0] - 1)), algorithm="auto").fit(X)
    dists, _ = nbrs.kneighbors(X)
    # Use average distance to k-th neighbor as inverse density proxy
    dens = 1.0 / (dists[:, -1] + 1e-8)
    return dens


def _queue_free_summary(longcp_sum: Path, wide_sum: Path) -> Dict[str, Dict[str, float]]:
    def stats(csv_path: Path) -> Dict[str, float]:
        df = pd.read_csv(csv_path)
        df_tr = df[df["split"] == "train"]
        qf = df_tr["queue_free"].astype(float)
        return {
            "n_train": int(df_tr.shape[0]),
            "queue_free_frac": float((qf == 1.0).mean()),
            "qf_counts": float((qf == 1.0).sum()),
            "ifr_mem_min": float(df_tr["IFR_mem"].min()),
            "ifr_mem_max": float(df_tr["IFR_mem"].max()),
            "ifr_cpu_min": float(df_tr["IFR_cpu"].min()),
            "ifr_cpu_max": float(df_tr["IFR_cpu"].max()),
        }
    return {
        "longcp": stats(longcp_sum),
        "wide": stats(wide_sum),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("runs/representativeness/geometry"))
    args = p.parse_args()

    long_base = Path("runs/datasets/longcp/representativeness")
    wide_base = Path("runs/datasets/wide/representativeness")
    long_df = _load_domain_features(long_base)
    wide_df = _load_domain_features(wide_base)

    # Harmonize columns and concat
    feat_cols = [c for c in long_df.columns if c in ["tasks","edges","width_peak","depth_avg","Pbar","burstiness","cp_frac"]]
    assert feat_cols, "No feature columns found"
    df_all = pd.concat([long_df[feat_cols + ["seed","domain"]], wide_df[feat_cols + ["seed","domain"]]], ignore_index=True)

    # Selected eval seeds
    eval_map = {
        "longcp": _load_selected_eval(long_base / "selected_eval_seeds.json"),
        "wide": _load_selected_eval(wide_base / "selected_eval_seeds.json"),
    }

    # PCA plot with convex hulls and eval seeds
    _plot_pca(df_all, feat_cols, eval_map, args.out_dir)

    # Density and Mahalanobis per domain
    outputs: List[Dict[str, object]] = []
    for dom in ["longcp", "wide"]:
        sub = df_all[df_all["domain"] == dom].reset_index(drop=True)
        X = sub[feat_cols].to_numpy(dtype=float)
        knn_d = _knn_density(sub, feat_cols, k=10)
        mu = X.mean(axis=0)
        cov_inv = _cov_inv(X)
        maha = _mahalanobis(X, mu, cov_inv)
        sub = sub.assign(knn_density=knn_d, mahalanobis=maha)
        # Rank low-density (tail) and high-distance
        sub["dens_rank"] = sub["knn_density"].rank(method="average", ascending=True)  # low density -> rank 1
        sub["maha_rank"] = sub["mahalanobis"].rank(method="average", ascending=False) # high distance -> rank 1
        sub["oob_score"] = 0.5 * (sub["dens_rank"] / len(sub)) + 0.5 * (sub["maha_rank"] / len(sub))
        # Collect top 10 likely boundary seeds
        top = sub.sort_values("oob_score", ascending=True).head(10)
        for _, r in top.iterrows():
            outputs.append({
                "domain": dom,
                "seed": int(r["seed"]),
                "knn_density": float(r["knn_density"]),
                "mahalanobis": float(r["mahalanobis"]),
                "oob_score": float(r["oob_score"]),
            })
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outputs).to_csv(out_dir / "boundary_seeds.csv", index=False)

    # Queue-free summary
    qf = _queue_free_summary(
        Path("runs/analysis/longcp/summary.csv"),
        Path("runs/analysis/wide/summary.csv"),
    )
    with (out_dir / "queue_free_summary.json").open("w") as f:
        json.dump(qf, f, indent=2)

    print("[done] Wrote:")
    print(" -", out_dir / "pca_hulls_eval.png")
    print(" -", out_dir / "boundary_seeds.csv")
    print(" -", out_dir / "queue_free_summary.json")


if __name__ == "__main__":
    main()
